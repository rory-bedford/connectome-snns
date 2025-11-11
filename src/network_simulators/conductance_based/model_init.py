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
        dt: float,
        weights: FloatArray,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        scaling_factors: FloatArray | None = None,
        scaling_factors_FF: FloatArray | None = None,
        optimisable: OptimisableParams = None,
        use_tqdm: bool = True,
    ):
        """
        Initialize the conductance-based LIF network with explicit parameters.

        Args:
            dt (float): The timestep in milliseconds (ms).
            weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            weights_FF (FloatArray): Feedforward weight matrix of shape (n_inputs, n_neurons).
            cell_type_indices (IntArray): Array of shape (n_neurons,) with cell type indices (0, 1, 2, ...).
            cell_type_indices_FF (IntArray): Array of feedforward cell type indices.
            cell_params (list[dict]): List of cell type parameter dicts. Each dict contains:
                - 'name' (str): Cell type name (e.g., 'excitatory', 'inhibitory')
                - 'cell_id' (int): Cell type ID (0, 1, 2, ...)
                - 'tau_mem' (float): Membrane time constant (ms)
                - 'theta' (float): Spike threshold voltage (mV)
                - 'U_reset' (float): Reset potential after spike (mV)
                - 'E_L' (float): Leak reversal potential (mV)
                - 'g_L' (float): Leak conductance (nS)
                - 'tau_ref' (float): Refractory period (ms)
            cell_params_FF (list[dict]): List of feedforward cell type parameter dicts (same structure as cell_params).
            synapse_params (list[dict]): List of synapse parameter dicts. Each dict contains:
                - 'name' (str): Synapse type name (e.g., 'AMPA', 'NMDA', 'GABA_A')
                - 'synapse_id' (int): Unique synapse type ID (0, 1, 2, ...)
                - 'cell_id' (int): Presynaptic cell type ID this synapse belongs to
                - 'tau_rise' (float): Synaptic rise time constant (ms)
                - 'tau_decay' (float): Synaptic decay time constant (ms)
                - 'E_syn' (float): Synaptic reversal potential (mV)
                - 'g_bar' (float): Maximum synaptic conductance (nS)
            synapse_params_FF (list[dict]): List of feedforward synapse parameter dicts (same structure as synapse_params).
            surrgrad_scale (float): Scale parameter for surrogate gradient fast sigmoid function.
            scaling_factors (FloatArray | None): Matrix of shape (n_cell_types, n_cell_types) for recurrent scaling (voxel^-1).
                If None, no scaling is applied (identity scaling).
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
            dt=dt,
            weights=weights,
            weights_FF=weights_FF,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=cell_params,
            cell_params_FF=cell_params_FF,
            synapse_params=synapse_params,
            synapse_params_FF=synapse_params_FF,
            scaling_factors=scaling_factors,
            scaling_factors_FF=scaling_factors_FF,
            surrgrad_scale=surrgrad_scale,
        )

        # ====================================================
        # FIXED PARAMETERS (NON-TRAINABLE - STORED IN BUFFERS)
        # ====================================================

        self.n_cell_types = len(cell_params)
        self.n_synapse_types = len(synapse_params)
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)

        self.n_neurons = weights.shape[0]
        self.n_inputs = weights_FF.shape[0]
        self.n_cell_types = self.n_cell_types
        self.n_synapse_types = self.n_synapse_types

        # Store cell and synapse parameter dictionaries
        self.cell_params = cell_params
        self.synapse_params = synapse_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF
        self.n_cell_types_FF = self.n_cell_types_FF
        self.n_synapse_types_FF = len(synapse_params_FF)

        # Register weights and cell type indices for recurrent connections
        self._register_parameter_or_buffer(
            "weights", weights, trainable=(self.optimisable == "weights")
        )
        self._register_parameter_or_buffer(
            "cell_type_indices", cell_type_indices, trainable=False
        )

        # Register feedforward weights and cell type indices
        self._register_parameter_or_buffer(
            "weights_FF", weights_FF, trainable=(self.optimisable == "weights")
        )
        self._register_parameter_or_buffer(
            "cell_type_indices_FF", cell_type_indices_FF, trainable=False
        )

        # Create and register weights masks (always non-trainable)
        weights_mask = weights != 0
        weights_mask_FF = weights_FF != 0
        self.register_buffer("weights_mask", torch.from_numpy(weights_mask))
        self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

        # Create and register mapping from synapse_id to cell_id for recurrent
        synapse_to_cell_mapping = np.zeros(self.n_synapse_types, dtype=np.int64)
        for synapse in synapse_params:
            synapse_to_cell_mapping[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id", torch.from_numpy(synapse_to_cell_mapping)
        )

        # Create and register mapping from synapse_id to cell_id for feedforward
        synapse_to_cell_mapping_FF = np.zeros(self.n_synapse_types_FF, dtype=np.int64)
        for synapse in synapse_params_FF:
            synapse_to_cell_mapping_FF[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id_FF", torch.from_numpy(synapse_to_cell_mapping_FF)
        )

        # Create boolean masks: cell_to_synapse_mask[cell_type] gives boolean mask for synapse types
        # Masks cover both recurrent and feedforward synapses (concatenated)
        total_synapse_types = self.n_synapse_types + self.n_synapse_types_FF

        # Get unique cell type IDs from the actual parameters (might not be contiguous)
        recurrent_cell_types = sorted(
            set(synapse["cell_id"] for synapse in synapse_params)
        )
        max_recurrent_cell_type = (
            max(recurrent_cell_types) if recurrent_cell_types else -1
        )

        cell_to_synapse_mask = torch.zeros(
            max_recurrent_cell_type + 1, total_synapse_types, dtype=torch.bool
        )
        for cell_type in recurrent_cell_types:
            cell_to_synapse_mask[cell_type, : self.n_synapse_types] = (
                self.synapse_to_cell_id == cell_type
            )
        self.register_buffer("cell_to_synapse_mask", cell_to_synapse_mask)

        # Same for feedforward cell types
        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))
        max_ff_cell_type = max(ff_cell_types) if ff_cell_types else -1

        cell_to_synapse_mask_FF = torch.zeros(
            max_ff_cell_type + 1, total_synapse_types, dtype=torch.bool
        )
        for cell_type in ff_cell_types:
            cell_to_synapse_mask_FF[cell_type, self.n_synapse_types :] = (
                self.synapse_to_cell_id_FF == cell_type
            )
        self.register_buffer("cell_to_synapse_mask_FF", cell_to_synapse_mask_FF)

        # Precompute cell type masks for efficient indexing
        cell_type_masks = []
        for cell_type in recurrent_cell_types:
            cell_type_masks.append(self.cell_type_indices == cell_type)
        self.cell_type_masks = cell_type_masks

        cell_type_masks_FF = []
        for cell_type in ff_cell_types:
            cell_type_masks_FF.append(self.cell_type_indices_FF == cell_type)
        self.cell_type_masks_FF = cell_type_masks_FF

        # Create neuron-indexed arrays from cell parameters
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)

        # Register physiological parameters as neuron-indexed buffers
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Create synapse parameter arrays (shape: n_synapse_types)
        synapse_param_arrays = self._create_synapse_param_arrays(synapse_params)

        # Create feedforward synapse parameter arrays
        synapse_param_arrays_FF = self._create_synapse_param_arrays(synapse_params_FF)

        # Combine recurrent and feedforward synapse parameters immediately
        for param_name in synapse_param_arrays.keys():
            recurrent_param = synapse_param_arrays[param_name]
            ff_param = synapse_param_arrays_FF[param_name]
            combined_param = torch.cat([recurrent_param, ff_param], dim=-1)
            self.register_buffer(param_name, combined_param)

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

        # Initialize timestep-dependent parameters
        self.set_timestep(dt)

    def _register_parameter_or_buffer(
        self, name: str, value: torch.Tensor | np.ndarray, trainable: bool = False
    ) -> None:
        """
        Register a parameter as either trainable (nn.Parameter) or fixed (buffer).

        Args:
            name (str): Parameter name.
            value (torch.Tensor | np.ndarray): Parameter value.
            trainable (bool): Whether to make this parameter trainable.
        """
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            # Only convert to float if not already an integer type
            if value.dtype.is_floating_point:
                value = value.float()
            else:
                # Convert integer types to int (int32) for indexing operations
                value = value.int()

        if trainable:
            self.register_parameter(name, nn.Parameter(value))
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
    def spike_fn(self):
        """
        Get the surrogate gradient spike function with the current scale parameter.

        Returns:
            Callable: Partial function for SurrGradSpike with configured scale.
        """

        def spike_function(x):
            return SurrGradSpike.apply(x, self.surrgrad_scale)

        return spike_function

    def set_timestep(self, dt: float) -> None:
        """
        Set the simulation timestep for the network.

        Args:
            dt (float): The timestep in milliseconds (ms).
        """
        assert isinstance(dt, float), "dt must be a float"
        assert dt > 0, "Timestep dt must be positive"

        # Convert dt to tensor and register as buffer
        self.register_buffer("dt", torch.tensor(dt, device=self.device))

        # Precompute decay factors using combined arrays and register as buffers
        tau_syn = torch.stack(
            (self.tau_rise, self.tau_decay), dim=0
        )  # Shape (2, n_synapse_types + n_synapse_types_FF)
        self.register_buffer("tau_syn", tau_syn)

        alpha = torch.exp(
            -self.dt / self.tau_syn
        )  # Shape (2, n_synapse_types + n_synapse_types_FF)
        self.register_buffer("alpha", alpha)

        beta = torch.exp(-self.dt / self.tau_mem)  # Shape (n_neurons,)
        self.register_buffer("beta", beta)

        # Stack g_bar with its negative using combined arrays
        g_scale = torch.stack(
            [-self.g_bar, self.g_bar], dim=0
        )  # Shape (2, n_synapse_types + n_synapse_types_FF)

        # Normalize by peak using combined arrays
        norm_peak = (self.tau_decay / self.tau_rise) ** (
            self.tau_rise / (self.tau_decay - self.tau_rise)
        )
        g_scale = g_scale / norm_peak
        self.register_buffer("g_scale", g_scale)

    def _validate(
        self,
        dt: float,
        weights: FloatArray,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        scaling_factors: FloatArray | None,
        scaling_factors_FF: FloatArray | None,
        surrgrad_scale: float,
    ) -> tuple[int, int, int, int]:
        """
        Validate all input parameters and return extracted dimensions.

        Args:
            dt: Simulation timestep in milliseconds
            weights: Recurrent weight matrix
            weights_FF: Feedforward weight matrix
            cell_type_indices: Neuron-to-cell-type mapping
            cell_type_indices_FF: Input-to-cell-type mapping
            cell_params: List of cell parameter dicts
            cell_params_FF: List of feedforward cell parameter dicts
            synapse_params: List of synapse parameter dicts
            synapse_params_FF: List of feedforward synapse parameter dicts
            scaling_factors: Cell-type-to-cell-type scaling matrix (optional)
            scaling_factors_FF: Input-cell-type-to-cell-type scaling matrix (optional)
            surrgrad_scale: Surrogate gradient scale

        Returns:
            tuple: (n_cell_types, n_synapse_types, n_cell_types_FF, n_synapse_types_FF)
        """
        # ========================================
        # TIMESTEP VALIDATION
        # ========================================
        assert isinstance(dt, (int, float)), "dt must be numeric (int or float)"
        assert dt > 0, f"Simulation timestep dt must be positive, got {dt}"
        assert dt <= 100.0, (
            f"Simulation timestep dt seems unusually large ({dt} ms), please check units"
        )

        # Extract key dimensions
        n_neurons = weights.shape[0] if weights.ndim == 2 else 0
        n_inputs = weights_FF.shape[0] if weights_FF.ndim == 2 else 0

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
        # FEEDFORWARD PARAMETERS VALIDATION
        # ========================================

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
        assert isinstance(synapse_params_FF, list), "synapse_params_FF must be a list"
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
        assert np.array_equal(np.sort(synapse_ids_FF_array), expected_synapse_ids_FF), (
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

        # ========================================
        # HYPERPARAMETER VALIDATION
        # ========================================
        assert isinstance(surrgrad_scale, (int, float)), (
            "surrgrad_scale must be numeric"
        )
        assert surrgrad_scale > 0, "Surrogate gradient scale must be positive"

        return n_cell_types, n_synapse_types, n_cell_types_FF, n_synapse_types_FF

    def _validate_forward(
        self,
        input_spikes: torch.Tensor,
        initial_v: torch.Tensor | None,
        initial_g: torch.Tensor | None,
        initial_g_FF: torch.Tensor | None,
    ) -> None:
        """Validate the inputs to the forward method."""

        # Validate input_spikes (now required, not nullable)
        assert input_spikes is not None, "input_spikes cannot be None"
        assert isinstance(input_spikes, torch.Tensor), (
            "input_spikes must be a torch.Tensor"
        )
        assert input_spikes.dtype == torch.bool, (
            f"input_spikes must be torch.bool, but got {input_spikes.dtype}"
        )
        assert input_spikes.device == self.device, (
            f"input_spikes must be on device {self.device}, but got {input_spikes.device}"
        )
        assert input_spikes.ndim == 3, (
            "input_spikes must have 3 dimensions (batch_size, n_steps, n_inputs)."
        )

        # Determine batch size and n_steps from input_spikes
        batch_size = input_spikes.shape[0]

        assert input_spikes.shape[2] == self.n_inputs, (
            "input_spikes must match the number of feedforward inputs."
        )

        # Validate initial conductances if provided
        if initial_g is not None:
            assert isinstance(initial_g, torch.Tensor), (
                "initial_g must be a torch.Tensor"
            )
            assert initial_g.dtype == torch.float32, (
                f"initial_g must be torch.float32, but got {initial_g.dtype}"
            )
            assert initial_g.device == self.device, (
                f"initial_g must be on device {self.device}, but got {initial_g.device}"
            )
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
            assert isinstance(initial_g_FF, torch.Tensor), (
                "initial_g_FF must be a torch.Tensor"
            )
            assert initial_g_FF.dtype == torch.float32, (
                f"initial_g_FF must be torch.float32, but got {initial_g_FF.dtype}"
            )
            assert initial_g_FF.device == self.device, (
                f"initial_g_FF must be on device {self.device}, but got {initial_g_FF.device}"
            )
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
            assert isinstance(initial_v, torch.Tensor), (
                "initial_v must be a torch.Tensor"
            )
            assert initial_v.dtype == torch.float32, (
                f"initial_v must be torch.float32, but got {initial_v.dtype}"
            )
            assert initial_v.device == self.device, (
                f"initial_v must be on device {self.device}, but got {initial_v.device}"
            )
            assert initial_v.ndim == 2, (
                "initial_v must have 2 dimensions (batch_size, n_neurons)."
            )
            assert initial_v.shape[1] == self.n_neurons, (
                "initial_v must match n_neurons."
            )
            assert initial_v.shape[0] == batch_size, (
                "initial_v batch size must match batch_size."
            )
