"""I/O utilities for LIF network parameters"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing import Literal
from optimisation.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]
OptimisableParams = Literal["weights", "scaling_factors", None]


class ConductanceLIFNetwork_IO(nn.Module):
    """Base class for LIF network with I/O functionality and parameter management."""

    def __init__(
        self,
        weights: FloatArray,
        cell_type_indices: IntArray,
        cell_params: list[dict],
        synapse_params: list[dict],
        surrgrad_scale: float,
        scaling_factors: FloatArray | None = None,
        weights_FF: FloatArray | None = None,
        cell_type_indices_FF: IntArray | None = None,
        cell_params_FF: list[dict] | None = None,
        synapse_params_FF: list[dict] | None = None,
        scaling_factors_FF: FloatArray | None = None,
        optimisable: OptimisableParams = None,
        use_tqdm: bool = True,
    ):
        """
        Initialize the conductance-based LIF network with explicit parameters.

        Args:
            weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            cell_type_indices (IntArray): Array of shape (n_neurons,) with cell type indices (0, 1, 2, ...).
            cell_params (list[dict]): List of cell type parameter dicts. Each dict contains:
                - 'name' (str): Cell type name (e.g., 'excitatory', 'inhibitory')
                - 'cell_id' (int): Cell type ID (0, 1, 2, ...)
                - 'tau_mem' (float): Membrane time constant (ms)
                - 'theta' (float): Spike threshold voltage (mV)
                - 'U_reset' (float): Reset potential after spike (mV)
                - 'E_L' (float): Leak reversal potential (mV)
                - 'g_L' (float): Leak conductance (nS)
                - 'tau_ref' (float): Refractory period (ms)
            synapse_params (list[dict]): List of synapse parameter dicts. Each dict contains:
                - 'name' (str): Synapse type name (e.g., 'AMPA', 'NMDA', 'GABA_A')
                - 'synapse_id' (int): Unique synapse type ID (0, 1, 2, ...)
                - 'cell_id' (int): Presynaptic cell type ID this synapse belongs to
                - 'tau_rise' (float): Synaptic rise time constant (ms)
                - 'tau_decay' (float): Synaptic decay time constant (ms)
                - 'E_syn' (float): Synaptic reversal potential (mV)
                - 'g_bar' (float): Maximum synaptic conductance (nS)
            surrgrad_scale (float): Scale parameter for surrogate gradient fast sigmoid function.
            scaling_factors (FloatArray | None): Matrix of shape (n_cell_types, n_cell_types) for recurrent scaling (voxel^-1).
                If None, no scaling is applied (identity scaling).
            weights_FF (FloatArray | None): Feedforward weight matrix of shape (n_inputs, n_neurons) or None.
            cell_type_indices_FF (IntArray | None): Array of feedforward cell type indices.
            cell_params_FF (list[dict] | None): List of feedforward cell type parameter dicts (same structure as cell_params).
            synapse_params_FF (list[dict] | None): List of feedforward synapse parameter dicts (same structure as synapse_params).
            scaling_factors_FF (FloatArray | None): Matrix of shape (n_cell_types_FF, n_cell_types) for feedforward scaling (voxel^-1).
                If None, no scaling is applied (identity scaling).
            optimisable (OptimisableParams): What to optimise during training. Options:
                - "weights": Optimise connection weights (weights and weights_FF)
                - "scaling_factors": Optimise scaling factors (scaling_factors and scaling_factors_FF)
                - None: Don't optimise anything (all parameters are fixed) [default]
            use_tqdm (bool): Whether to display tqdm progress bar during forward pass. Default is True.
        """
        super(ConductanceLIFNetwork_IO, self).__init__()

        # Store optimisation mode
        self.optimisable = optimisable

        # Store tqdm preference
        self.use_tqdm = use_tqdm

        # =================================
        # PARAMETER VALIDATION & ASSERTIONS
        # =================================

        self._validate(
            weights=weights,
            cell_type_indices=cell_type_indices,
            cell_params=cell_params,
            synapse_params=synapse_params,
            scaling_factors=scaling_factors,
            surrgrad_scale=surrgrad_scale,
            weights_FF=weights_FF,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params_FF=cell_params_FF,
            synapse_params_FF=synapse_params_FF,
            scaling_factors_FF=scaling_factors_FF,
        )

        # ====================================================
        # FIXED PARAMETERS (NON-TRAINABLE - STORED IN BUFFERS)
        # ====================================================

        self.n_cell_types = len(cell_params)
        self.n_synapse_types = len(synapse_params)
        self.n_cell_types_FF = (
            len(cell_params_FF) if cell_params_FF is not None else None
        )
        self.n_synapse_types_FF = (
            len(synapse_params_FF) if synapse_params_FF is not None else None
        )

        self.n_neurons = weights.shape[0]
        self.n_inputs = weights_FF.shape[0] if weights_FF is not None else 0
        self.n_cell_types = self.n_cell_types
        self.n_synapse_types = self.n_synapse_types

        # Store cell and synapse parameter dictionaries
        self.cell_params = cell_params
        self.synapse_params = synapse_params

        if cell_params_FF is not None:
            self.cell_params_FF = cell_params_FF
            self.synapse_params_FF = synapse_params_FF
            self.n_cell_types_FF = self.n_cell_types_FF
            self.n_synapse_types_FF = len(synapse_params_FF)
        else:
            self.cell_params_FF = None
            self.synapse_params_FF = None
            self.n_cell_types_FF = None
            self.n_synapse_types_FF = None

        # Register weights and cell type indices for recurrent connections
        self._register_parameter_or_buffer(
            "weights", weights, trainable=(self.optimisable == "weights")
        )
        self._register_parameter_or_buffer(
            "cell_type_indices", cell_type_indices, trainable=False
        )

        # Create and register weights mask (always non-trainable)
        weights_mask = weights != 0
        self.register_buffer("weights_mask", torch.from_numpy(weights_mask))

        # Create and register mapping from synapse_id to cell_id
        synapse_to_cell_mapping = np.zeros(self.n_synapse_types, dtype=np.int64)
        for synapse in synapse_params:
            synapse_to_cell_mapping[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id", torch.from_numpy(synapse_to_cell_mapping)
        )

        # Register feedforward structure (if provided)
        if weights_FF is not None:
            self._register_parameter_or_buffer(
                "weights_FF", weights_FF, trainable=(self.optimisable == "weights")
            )
            self._register_parameter_or_buffer(
                "cell_type_indices_FF", cell_type_indices_FF, trainable=False
            )

            # Create and register feedforward weights mask (always non-trainable)
            weights_mask_FF = weights_FF != 0
            self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

            # Create and register mapping from feedforward synapse_id to cell_id
            synapse_to_cell_mapping_FF = np.zeros(
                self.n_synapse_types_FF, dtype=np.int64
            )
            for synapse in synapse_params_FF:
                synapse_to_cell_mapping_FF[synapse["synapse_id"]] = synapse["cell_id"]
            self.register_buffer(
                "synapse_to_cell_id_FF", torch.from_numpy(synapse_to_cell_mapping_FF)
            )
        else:
            self.register_buffer("weights_FF", None)
            self.register_buffer("cell_type_indices_FF", None)
            self.register_buffer("synapse_to_cell_id_FF", None)
            self.register_buffer("weights_mask_FF", None)

        # Create neuron-indexed arrays from cell parameters
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)

        # Register physiological parameters as neuron-indexed buffers
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Create synapse parameter arrays (shape: n_synapse_types)
        synapse_param_arrays = self._create_synapse_param_arrays(synapse_params)

        # Register synapse parameters as buffers
        for param_name, param_array in synapse_param_arrays.items():
            self.register_buffer(param_name, param_array)

        # Create feedforward synapse parameter arrays (if provided)
        if synapse_params_FF is not None:
            synapse_param_arrays_FF = self._create_synapse_param_arrays(
                synapse_params_FF
            )

            # Register feedforward synapse parameters as buffers with _FF suffix
            for param_name, param_array in synapse_param_arrays_FF.items():
                self.register_buffer(f"{param_name}_FF", param_array)

        # ===========================================================
        # OPTIMISABLE PARAMETERS (TRAINABLE - STORED AS nn.Parameter)
        # ===========================================================

        # Register scaling factors (trainable if optimising scaling_factors)
        if scaling_factors is not None:
            self._register_parameter_or_buffer(
                "scaling_factors",
                scaling_factors,
                trainable=(self.optimisable == "scaling_factors"),
            )
        else:
            self.register_buffer("scaling_factors", None)

        if scaling_factors_FF is not None:
            self._register_parameter_or_buffer(
                "scaling_factors_FF",
                scaling_factors_FF,
                trainable=(self.optimisable == "scaling_factors"),
            )
        else:
            self.register_buffer("scaling_factors_FF", None)

        # ======================================================================
        # HYPERPARAMETERS (CONFIGURATION VALUES - STORED AS INSTANCE ATTRIBUTES)
        # ======================================================================

        self.surrgrad_scale = surrgrad_scale

    def _register_parameter_or_buffer(
        self,
        name: str,
        value: np.ndarray | torch.Tensor | None,
        requires_grad: bool = False,
    ) -> None:
        """Register a parameter or buffer, converting from numpy if needed.

        Args:
            name (str): Name of the parameter/buffer
            value (np.ndarray | torch.Tensor | None): Value to register
            requires_grad (bool): Whether to register as parameter (True) or buffer (False)
        """
        if value is None:
            return

        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            # Only convert to float if not already an integer type
            if value.dtype.is_floating_point:
                value = value.float()
            else:
                # Convert integer types to short (int16) for memory efficiency
                # This is sufficient for cell type indices (max ~255 types)
                value = value.short()

        if requires_grad:
            self.register_parameter(name, torch.nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def _create_neuron_param_arrays(
        self,
        cell_params: list[dict],
        cell_type_indices: IntArray,
    ) -> dict[str, torch.Tensor]:
        """
        Create neuron-indexed parameter arrays from cell-type-specific parameters.

        This method creates arrays of shape (n_neurons,) where each neuron's parameters
        are determined by its cell type. The cell_type_indices array maps each neuron
        to its cell type, enabling efficient broadcasted operations later.

        Args:
            cell_params (list[dict]): List of cell parameter dicts, each containing:
                - 'cell_id': Cell type ID (must be 0-indexed and contiguous)
                - 'name': Cell type name
                - Physiological parameters (tau_mem, theta, U_reset, E_L, g_L, tau_ref)
            cell_type_indices (IntArray): Array of shape (n_neurons,) mapping each neuron to its cell type index.
            n_neurons (int): Total number of neurons.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping parameter names to torch tensors of shape (n_neurons,).
                Each tensor contains the parameter value for each neuron based on its cell type.
        """
        # Required physiological parameters for conductance-based LIF neurons
        required_param_names = [
            "tau_mem",  # Membrane time constant (ms)
            "theta",  # Spike threshold voltage (mV)
            "U_reset",  # Reset potential after spike (mV)
            "E_L",  # Leak reversal potential (mV)
            "g_L",  # Leak conductance (nS)
            "tau_ref",  # Refractory period (ms)
        ]

        # Extract n_cell_types from the list of dicts
        n_cell_types = len(cell_params)

        neuron_params = {}

        # Initialize parameter lookup array
        for param_name in required_param_names:
            param_lookup = np.zeros(n_cell_types, dtype=np.float32)
            for cell in cell_params:
                cell_id = cell["cell_id"]
                param_lookup[cell_id] = cell[param_name]

            # Map each neuron to its parameter value using cell_type_indices
            param_array = param_lookup[cell_type_indices]
            neuron_params[param_name] = torch.from_numpy(param_array)

        # Compute derived parameter: capacitance (C_m = tau_mem * g_L)
        neuron_params["C_m"] = neuron_params["tau_mem"] * neuron_params["g_L"]

        return neuron_params

    def _create_synapse_param_arrays(
        self,
        synapse_params: list[dict],
    ) -> dict[str, torch.Tensor]:
        """
        Create synapse parameter arrays from synapse-type-specific parameters.

        This method creates arrays of shape (n_synapse_types,) for each synapse parameter.
        Unlike neuron parameters, synapse parameters are indexed only by synapse type,
        not by individual neurons.

        Args:
            synapse_params (list[dict]): List of synapse parameter dicts, each containing:
                - 'synapse_id': Synapse type ID (must be 0-indexed and contiguous)
                - 'name': Synapse type name (e.g., 'AMPA', 'NMDA', 'GABA_A')
                - 'cell_id': Presynaptic cell type this synapse belongs to
                - 'tau_rise': Synaptic rise time constant (ms)
                - 'tau_decay': Synaptic decay time constant (ms)
                - 'E_syn': Synaptic reversal potential (mV)
                - 'g_bar': Maximum synaptic conductance (nS)

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping parameter names to torch tensors of shape (n_synapse_types,).
                Each tensor contains the parameter value for each synapse type.
        """
        # Required synapse parameters for conductance-based synapses
        required_param_names = [
            "tau_rise",  # Synaptic rise time constant (ms)
            "tau_decay",  # Synaptic decay time constant (ms)
            "E_syn",  # Synaptic reversal potential (mV)
            "g_bar",  # Maximum synaptic conductance (nS)
        ]

        # Extract n_synapse_types from the list of dicts
        # Note: synapse_id validation already done in __init__
        n_synapse_types = len(synapse_params)

        synapse_param_arrays = {}

        # Create lookup array directly indexed by synapse_id (no sorting needed)
        for param_name in required_param_names:
            param_lookup = np.zeros(n_synapse_types, dtype=np.float32)

            # Directly index by synapse_id - each synapse_id is used as array index
            for synapse in synapse_params:
                synapse_id = synapse["synapse_id"]
                param_lookup[synapse_id] = synapse[param_name]

            # Store as tensor of shape (n_synapse_types,)
            synapse_param_arrays[param_name] = torch.from_numpy(param_lookup)

        return synapse_param_arrays

    @property
    def device(self):
        """Get the device the model is on"""
        return self.weights.device

    @property
    def scaled_weights(self) -> torch.Tensor:
        """
        Get the recurrent weights scaled by the scaling_factors matrix and masked by connectivity.

        The scaling_factors matrix is of shape (n_cell_types, n_cell_types) where
        scaling_factors[i, j] scales connections from cell type i to cell type j.

        If scaling_factors is None, returns the raw weights without scaling.

        The weights_mask ensures that zero connections remain zero and do not accumulate gradients.

        Returns:
            torch.Tensor: Scaled and masked recurrent weight matrix of shape (n_neurons, n_neurons).
        """
        if self.scaling_factors is None:
            return self.weights * self.weights_mask

        source_types = self.cell_type_indices[:, None]  # shape (n_neurons, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors[
            source_types, target_types
        ]  # (n_neurons, n_neurons)
        return self.weights * scaling_matrix * self.weights_mask

    @property
    def scaled_weights_FF(self) -> torch.Tensor:
        """
        Get the feedforward weights scaled by the scaling_factors_FF matrix and masked by connectivity.

        The scaling_factors_FF matrix is of shape (n_cell_types_FF, n_cell_types) where
        scaling_factors_FF[i, j] scales connections from input cell type i to cell type j.

        If scaling_factors_FF is None, returns the raw feedforward weights without scaling.

        The weights_mask_FF ensures that zero connections remain zero and do not accumulate gradients.

        Returns:
            torch.Tensor: Scaled and masked feedforward weight matrix of shape (n_inputs, n_neurons).
        """
        if self.weights_FF is None:
            raise ValueError("weights_FF must be provided for feedforward scaling.")

        if self.scaling_factors_FF is None:
            return self.weights_FF * self.weights_mask_FF

        input_types = self.cell_type_indices_FF[:, None]  # shape (n_inputs, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors_FF[
            input_types, target_types
        ]  # (n_inputs, n_neurons)
        return self.weights_FF * scaling_matrix * self.weights_mask_FF

    @property
    def cell_typed_weights(self) -> torch.Tensor:
        """
        Create a tiered structure for the recurrent weight matrix based on synapse types.

        Each synapse layer contains the weights that belong presynaptically to the
        cell_id associated with that synapse type.

        Returns:
            torch.Tensor: Tiered weight matrix of shape
                (n_synapse_types, n_neurons, n_neurons).
        """
        # Initialize a zero tensor for tiered weights
        n_tiers = self.n_synapse_types
        tiered_weights = torch.zeros(
            (n_tiers, self.n_neurons, self.n_neurons),
            dtype=self.weights.dtype,
            device=self.weights.device,
        )

        # For each synapse type, get its associated presynaptic cell_id
        # and mask weights from neurons of that cell type
        for synapse_id in range(n_tiers):
            cell_id = self.synapse_to_cell_id[synapse_id]
            # Create mask for neurons belonging to this cell type
            mask = self.cell_type_indices == cell_id  # Shape: (n_neurons,)
            # Apply mask to presynaptic dimension (rows of weight matrix)
            tiered_weights[synapse_id] = mask[:, None] * self.scaled_weights

        return tiered_weights

    @property
    def cell_typed_weights_FF(self) -> torch.Tensor:
        """
        Create a tiered structure for the feedforward weight matrix based on synapse types.

        Each synapse layer contains the weights that belong presynaptically to the
        cell_id associated with that feedforward synapse type.

        Returns:
            torch.Tensor: Tiered weight matrix of shape
                (n_synapse_types_FF, n_inputs, n_neurons).
        """
        if self.weights_FF is None or self.cell_type_indices_FF is None:
            raise ValueError(
                "Feedforward weights and cell type indices must be provided."
            )

        # Initialize a zero tensor for tiered weights
        n_tiers = self.n_synapse_types_FF
        tiered_weights_FF = torch.zeros(
            (n_tiers, self.n_inputs, self.n_neurons),
            dtype=self.weights_FF.dtype,
            device=self.weights_FF.device,
        )

        # For each feedforward synapse type, get its associated presynaptic cell_id
        # and mask weights from inputs of that cell type
        for synapse_id in range(n_tiers):
            cell_id = self.synapse_to_cell_id_FF[synapse_id]
            # Create mask for inputs belonging to this cell type
            mask = self.cell_type_indices_FF == cell_id  # Shape: (n_inputs,)
            # Apply mask to presynaptic dimension (rows of weight matrix)
            tiered_weights_FF[synapse_id] = mask[:, None] * self.scaled_weights_FF

        return tiered_weights_FF

    @property
    def spike_fn(self):
        """
        Get the surrogate gradient spike function with the current scale parameter.

        Returns:
            Callable: Partial function for SurrGradSpike with configured scale.
        """
        return lambda x: SurrGradSpike.apply(x, self.surrgrad_scale)

    def _validate(
        self,
        weights: FloatArray,
        cell_type_indices: IntArray,
        cell_params: list[dict],
        synapse_params: list[dict],
        scaling_factors: FloatArray | None,
        surrgrad_scale: float,
        weights_FF: FloatArray | None,
        cell_type_indices_FF: IntArray | None,
        cell_params_FF: list[dict] | None,
        synapse_params_FF: list[dict] | None,
        scaling_factors_FF: FloatArray | None,
    ) -> tuple[int, int, int | None, int | None]:
        """
        Validate all input parameters and return extracted dimensions.

        Args:
            weights: Recurrent weight matrix
            cell_type_indices: Neuron-to-cell-type mapping
            cell_params: List of cell parameter dicts
            synapse_params: List of synapse parameter dicts
            scaling_factors: Cell-type-to-cell-type scaling matrix (optional)
            surrgrad_scale: Surrogate gradient scale
            weights_FF: Feedforward weight matrix (optional)
            cell_type_indices_FF: Input-to-cell-type mapping (optional)
            cell_params_FF: List of feedforward cell parameter dicts (optional)
            synapse_params_FF: List of feedforward synapse parameter dicts (optional)
            scaling_factors_FF: Input-cell-type-to-cell-type scaling matrix (optional)

        Returns:
            tuple: (n_cell_types, n_synapse_types, n_cell_types_FF, n_synapse_types_FF)
                where FF values are None if feedforward is not provided
        """
        # Extract key dimensions
        n_neurons = weights.shape[0] if weights.ndim == 2 else 0
        n_inputs = (
            weights_FF.shape[0]
            if weights_FF is not None and weights_FF.ndim == 2
            else 0
        )

        # ========================================
        # RECURRENT CELL PARAMS VALIDATION
        # ========================================
        assert isinstance(cell_params, list), "cell_params must be a list"
        assert len(cell_params) > 0, "cell_params must not be empty"
        assert all(isinstance(p, dict) for p in cell_params), (
            "All cell_params entries must be dicts"
        )

        # Extract n_cell_types from cell_params by finding max cell_id
        cell_ids = [params["cell_id"] for params in cell_params]
        n_cell_types = max(cell_ids) + 1

        # Ensure cell_ids are 0-indexed, contiguous, and complete
        cell_ids_array = np.array(cell_ids, dtype=np.int32)
        expected_cell_ids = np.arange(n_cell_types, dtype=np.int32)
        assert np.array_equal(np.sort(cell_ids_array), expected_cell_ids), (
            f"cell_ids in cell_params must be 0-indexed and contiguous [0, 1, ..., {n_cell_types - 1}]. "
            f"Found: {sorted(cell_ids)}, Expected: {expected_cell_ids.tolist()}"
        )

        # Ensure no duplicate cell_ids
        assert len(cell_ids) == len(np.unique(cell_ids_array)), (
            f"Duplicate cell_ids found in cell_params: {cell_ids}"
        )

        # ========================================
        # RECURRENT SYNAPSE PARAMS VALIDATION
        # ========================================
        assert isinstance(synapse_params, list), "synapse_params must be a list"
        assert len(synapse_params) > 0, "synapse_params must not be empty"
        assert all(isinstance(p, dict) for p in synapse_params), (
            "All synapse_params entries must be dicts"
        )

        # Extract synapse_ids and validate
        synapse_ids = [params["synapse_id"] for params in synapse_params]
        n_synapse_types = max(synapse_ids) + 1

        # Ensure synapse_ids are 0-indexed, contiguous, and complete
        synapse_ids_array = np.array(synapse_ids, dtype=np.int32)
        expected_synapse_ids = np.arange(n_synapse_types, dtype=np.int32)
        assert np.array_equal(np.sort(synapse_ids_array), expected_synapse_ids), (
            f"synapse_ids in synapse_params must be 0-indexed and contiguous [0, 1, ..., {n_synapse_types - 1}]. "
            f"Found: {sorted(synapse_ids)}, Expected: {expected_synapse_ids.tolist()}"
        )

        # Ensure no duplicate synapse_ids
        assert len(synapse_ids) == len(np.unique(synapse_ids_array)), (
            f"Duplicate synapse_ids found in synapse_params: {synapse_ids}"
        )

        # ========================================
        # RECURRENT WEIGHTS VALIDATION
        # ========================================
        assert weights.ndim == 2, "Recurrent weights must be 2D matrix"
        assert weights.shape[0] == weights.shape[1], (
            "Recurrent weights must be square matrix"
        )
        assert n_neurons > 0, "Number of neurons must be positive"

        # ========================================
        # RECURRENT CELL TYPE INDICES VALIDATION
        # ========================================
        assert cell_type_indices.ndim == 1, "Cell type indices must be 1D array"
        assert cell_type_indices.shape[0] == n_neurons, (
            f"Cell type indices length ({cell_type_indices.shape[0]}) must match number of neurons ({n_neurons})"
        )
        assert np.all(cell_type_indices >= 0), (
            "All cell type indices must be non-negative"
        )
        assert np.all(cell_type_indices < n_cell_types), (
            f"All cell type indices must be less than n_cell_types ({n_cell_types})"
        )

        # ========================================
        # RECURRENT SCALING FACTORS VALIDATION
        # ========================================
        if scaling_factors is not None:
            assert scaling_factors.ndim == 2, "Scaling factors must be 2D matrix"
            assert scaling_factors.shape == (n_cell_types, n_cell_types), (
                f"Scaling factors shape {scaling_factors.shape} must be ({n_cell_types}, {n_cell_types})"
            )
            assert np.all(scaling_factors > 0), "All scaling factors must be positive"

        # ========================================
        # FEEDFORWARD VALIDATION (ALL-OR-NOTHING)
        # ========================================
        ff_args = [
            weights_FF,
            cell_type_indices_FF,
            cell_params_FF,
            synapse_params_FF,
        ]
        ff_provided = [arg is not None for arg in ff_args]

        if any(ff_provided):
            assert all(ff_provided), (
                "If any feedforward argument is provided, all of "
                "weights_FF, cell_type_indices_FF, cell_params_FF, "
                "and synapse_params_FF must be provided."
            )

            # Feedforward weights validation
            assert weights_FF.ndim == 2, "Feedforward weights must be 2D matrix"
            assert weights_FF.shape[1] == n_neurons, (
                f"Feedforward weights output dimension ({weights_FF.shape[1]}) must match number of neurons ({n_neurons})"
            )

            # Feedforward cell type indices validation
            assert cell_type_indices_FF.ndim == 1, (
                "Feedforward cell type indices must be 1D array"
            )
            assert cell_type_indices_FF.shape[0] == n_inputs, (
                f"Feedforward cell type indices length ({cell_type_indices_FF.shape[0]}) must match number of inputs ({n_inputs})"
            )

            # Feedforward cell params validation
            assert isinstance(cell_params_FF, list), "cell_params_FF must be a list"
            assert len(cell_params_FF) > 0, "cell_params_FF must not be empty"
            assert all(isinstance(p, dict) for p in cell_params_FF), (
                "All cell_params_FF entries must be dicts"
            )

            # Extract n_cell_types_FF from cell_params_FF
            cell_ids_FF = [params["cell_id"] for params in cell_params_FF]
            n_cell_types_FF = max(cell_ids_FF) + 1

            # Ensure feedforward cell_ids are 0-indexed, contiguous, and complete
            cell_ids_FF_array = np.array(cell_ids_FF, dtype=np.int32)
            expected_cell_ids_FF = np.arange(n_cell_types_FF, dtype=np.int32)
            assert np.array_equal(np.sort(cell_ids_FF_array), expected_cell_ids_FF), (
                f"cell_ids in cell_params_FF must be 0-indexed and contiguous [0, 1, ..., {n_cell_types_FF - 1}]. "
                f"Found: {sorted(cell_ids_FF)}, Expected: {expected_cell_ids_FF.tolist()}"
            )

            # Ensure no duplicate cell_ids in feedforward
            assert len(cell_ids_FF) == len(np.unique(cell_ids_FF_array)), (
                f"Duplicate cell_ids found in cell_params_FF: {cell_ids_FF}"
            )

            # Feedforward cell type indices range validation
            assert np.all(cell_type_indices_FF >= 0), (
                "All feedforward cell type indices must be non-negative"
            )
            assert np.all(cell_type_indices_FF < n_cell_types_FF), (
                f"All feedforward cell type indices must be less than n_cell_types_FF ({n_cell_types_FF})"
            )

            # Feedforward synapse params validation
            assert isinstance(synapse_params_FF, list), (
                "synapse_params_FF must be a list"
            )
            assert len(synapse_params_FF) > 0, "synapse_params_FF must not be empty"
            assert all(isinstance(p, dict) for p in synapse_params_FF), (
                "All synapse_params_FF entries must be dicts"
            )

            # Extract synapse_ids_FF and validate
            synapse_ids_FF = [params["synapse_id"] for params in synapse_params_FF]
            n_synapse_types_FF = max(synapse_ids_FF) + 1

            # Ensure feedforward synapse_ids are 0-indexed, contiguous, and complete
            synapse_ids_FF_array = np.array(synapse_ids_FF, dtype=np.int32)
            expected_synapse_ids_FF = np.arange(n_synapse_types_FF, dtype=np.int32)
            assert np.array_equal(
                np.sort(synapse_ids_FF_array), expected_synapse_ids_FF
            ), (
                f"synapse_ids in synapse_params_FF must be 0-indexed and contiguous [0, 1, ..., {n_synapse_types_FF - 1}]. "
                f"Found: {sorted(synapse_ids_FF)}, Expected: {expected_synapse_ids_FF.tolist()}"
            )

            # Ensure no duplicate synapse_ids in feedforward
            assert len(synapse_ids_FF) == len(np.unique(synapse_ids_FF_array)), (
                f"Duplicate synapse_ids found in synapse_params_FF: {synapse_ids_FF}"
            )

            # Feedforward scaling factors validation (optional)
            if scaling_factors_FF is not None:
                assert scaling_factors_FF.ndim == 2, (
                    "Feedforward scaling factors must be 2D matrix"
                )
                assert scaling_factors_FF.shape == (n_cell_types_FF, n_cell_types), (
                    f"Feedforward scaling factors shape {scaling_factors_FF.shape} must be ({n_cell_types_FF}, {n_cell_types})"
                )
                assert np.all(scaling_factors_FF > 0), (
                    "All feedforward scaling factors must be positive"
                )
        else:
            n_cell_types_FF = None
            n_synapse_types_FF = None

        # ========================================
        # HYPERPARAMETER VALIDATION
        # ========================================
        assert isinstance(surrgrad_scale, (int, float)), (
            "surrgrad_scale must be numeric"
        )
        assert surrgrad_scale > 0, "Surrogate gradient scale must be positive"

    def _validate_forward(
        self,
        n_steps: int,
        dt: float,
        inputs: FloatArray | None,
        initial_v: FloatArray | None,
        initial_g: FloatArray | None,
        initial_g_FF: FloatArray | None,
    ) -> None:
        """Validate the inputs to the forward method."""

        # Determine batch size
        batch_size = inputs.shape[0] if inputs is not None else 1

        assert isinstance(dt, float), "dt must be a float."

        # Validate inputs if provided
        if inputs is not None:
            assert inputs.ndim == 3, (
                "inputs must have 3 dimensions (batch_size, n_steps, n_inputs)."
            )
            assert inputs.shape[0] == batch_size, (
                "inputs batch size must match batch_size."
            )
            assert inputs.shape[1] == n_steps, "inputs must have n_steps time steps."
            assert inputs.shape[2] == self.n_inputs, (
                "inputs must match the number of feedforward inputs."
            )

        # Validate initial conductances if provided
        if initial_g is not None:
            assert initial_g.ndim == 4, (
                "initial_g must have 4 dimensions (batch_size, n_neurons, 2, n_synapse_types)."
            )
            assert initial_g.shape[0] == batch_size, (
                "initial_g batch size must match batch_size."
            )
            assert initial_g.shape[1] == self.n_neurons, (
                "initial_g must match n_neurons."
            )
            assert initial_g.shape[2] == 2, (
                "initial_g must have 2 for rise/decay components."
            )
            assert initial_g.shape[3] == self.n_synapse_types, (
                "initial_g must match the number of synapse types."
            )

        # Validate initial feedforward conductances if provided
        if initial_g_FF is not None:
            assert initial_g_FF.ndim == 4, (
                "initial_g_FF must have 4 dimensions (batch_size, n_neurons, 2, n_synapse_types_FF)."
            )
            assert initial_g_FF.shape[0] == batch_size, (
                "initial_g_FF batch size must match batch_size."
            )
            assert initial_g_FF.shape[1] == self.n_neurons, (
                "initial_g_FF must match n_neurons."
            )
            assert initial_g_FF.shape[2] == 2, (
                "initial_g_FF must have 2 for rise/decay components."
            )
            assert initial_g_FF.shape[3] == self.n_synapse_types_FF, (
                "initial_g_FF must match the number of feedforward synapse types."
            )

        # Validate initial membrane potentials if provided
        if initial_v is not None:
            assert initial_v.ndim == 2, (
                "initial_v must have 2 dimensions (batch_size, n_neurons)."
            )
            assert initial_v.shape[1] == self.n_neurons, (
                "initial_v must match n_neurons."
            )
            assert initial_v.shape[0] == batch_size, (
                "initial_v batch size must match batch_size."
            )
