"""I/O utilities for LIF network parameters"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from optimisation.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class ConductanceLIFNetwork_IO(nn.Module):
    """Base class for LIF network with I/O functionality and parameter management."""

    def __init__(
        self,
        weights: FloatArray,
        cell_type_indices: IntArray,
        cell_params: list[dict],
        synapse_params: list[dict],
        scaling_factors: FloatArray,
        surrgrad_scale: float,
        weights_FF: FloatArray | None = None,
        cell_type_indices_FF: IntArray | None = None,
        cell_params_FF: list[dict] | None = None,
        synapse_params_FF: list[dict] | None = None,
        scaling_factors_FF: FloatArray | None = None,
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
                - 'reversal_potential' (float): Synaptic reversal potential (mV)
                - 'g_bar' (float): Maximum synaptic conductance (nS)
            scaling_factors (FloatArray): Matrix of shape (n_cell_types, n_cell_types) for recurrent scaling (voxel^-1).
            surrgrad_scale (float): Scale parameter for surrogate gradient fast sigmoid function.
            weights_FF (FloatArray | None): Feedforward weight matrix of shape (n_inputs, n_neurons) or None.
            cell_type_indices_FF (IntArray | None): Array of feedforward cell type indices.
            cell_params_FF (list[dict] | None): List of feedforward cell type parameter dicts (same structure as cell_params).
            synapse_params_FF (list[dict] | None): List of feedforward synapse parameter dicts (same structure as synapse_params).
            scaling_factors_FF (FloatArray | None): Matrix of shape (n_cell_types_FF, n_cell_types) for feedforward scaling (voxel^-1).
        """
        super(ConductanceLIFNetwork_IO, self).__init__()

        # =================================
        # PARAMETER VALIDATION & ASSERTIONS
        # =================================

        # Extract key dimensions for validation
        n_neurons = weights.shape[0] if weights.ndim == 2 else 0
        n_inputs = (
            weights_FF.shape[0]
            if weights_FF is not None and weights_FF.ndim == 2
            else 0
        )

        # Validate cell_params structure
        assert isinstance(cell_params, list), "cell_params must be a list"
        assert len(cell_params) > 0, "cell_params must not be empty"
        assert all(isinstance(p, dict) for p in cell_params), (
            "All cell_params entries must be dicts"
        )

        # Extract n_cell_types from cell_params by finding max cell_id
        cell_ids = [params["cell_id"] for params in cell_params]
        n_cell_types = max(cell_ids) + 1

        # Validate synapse_params structure
        assert isinstance(synapse_params, list), "synapse_params must be a list"
        assert len(synapse_params) > 0, "synapse_params must not be empty"
        assert all(isinstance(p, dict) for p in synapse_params), (
            "All synapse_params entries must be dicts"
        )

        # Recurrent weights validation
        assert weights.ndim == 2, "Recurrent weights must be 2D matrix"
        assert weights.shape[0] == weights.shape[1], (
            "Recurrent weights must be square matrix"
        )
        assert n_neurons > 0, "Number of neurons must be positive"

        # Cell type indices validation
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

        # Scaling factors validation (recurrent)
        assert scaling_factors.ndim == 2, "Scaling factors must be 2D matrix"
        assert scaling_factors.shape == (n_cell_types, n_cell_types), (
            f"Scaling factors shape {scaling_factors.shape} must be ({n_cell_types}, {n_cell_types})"
        )
        assert np.all(scaling_factors > 0), "All scaling factors must be positive"

        # Feedforward weights validation (all-or-nothing)
        ff_args = [
            weights_FF,
            cell_type_indices_FF,
            cell_params_FF,
            synapse_params_FF,
            scaling_factors_FF,
        ]
        ff_provided = [arg is not None for arg in ff_args]
        if any(ff_provided):
            assert all(ff_provided), (
                "If any feedforward argument is provided, all of "
                "weights_FF, cell_type_indices_FF, cell_params_FF, synapse_params_FF, and scaling_factors_FF must be provided."
            )
            assert weights_FF.ndim == 2, "Feedforward weights must be 2D matrix"
            assert weights_FF.shape[1] == n_neurons, (
                f"Feedforward weights output dimension ({weights_FF.shape[1]}) must match number of neurons ({n_neurons})"
            )
            assert cell_type_indices_FF.ndim == 1, (
                "Feedforward cell type indices must be 1D array"
            )
            assert cell_type_indices_FF.shape[0] == n_inputs, (
                f"Feedforward cell type indices length ({cell_type_indices_FF.shape[0]}) must match number of inputs ({n_inputs})"
            )

            # Validate feedforward cell_params structure
            assert isinstance(cell_params_FF, list), "cell_params_FF must be a list"
            assert len(cell_params_FF) > 0, "cell_params_FF must not be empty"
            assert all(isinstance(p, dict) for p in cell_params_FF), (
                "All cell_params_FF entries must be dicts"
            )

            # Extract n_cell_types_FF from cell_params_FF
            cell_ids_FF = [params["cell_id"] for params in cell_params_FF]
            n_cell_types_FF = max(cell_ids_FF) + 1

            # Validate feedforward synapse_params structure
            assert isinstance(synapse_params_FF, list), (
                "synapse_params_FF must be a list"
            )
            assert len(synapse_params_FF) > 0, "synapse_params_FF must not be empty"
            assert all(isinstance(p, dict) for p in synapse_params_FF), (
                "All synapse_params_FF entries must be dicts"
            )

            assert scaling_factors_FF.ndim == 2, (
                "Feedforward scaling factors must be 2D matrix"
            )
            assert scaling_factors_FF.shape == (n_cell_types_FF, n_cell_types), (
                f"Feedforward scaling factors shape {scaling_factors_FF.shape} must be ({n_cell_types_FF}, {n_cell_types})"
            )
            assert np.all(scaling_factors_FF > 0), (
                "All feedforward scaling factors must be positive"
            )

        # Hyperparameter validation
        assert isinstance(surrgrad_scale, (int, float)), (
            "surrgrad_scale must be numeric"
        )
        assert surrgrad_scale > 0, "Surrogate gradient scale must be positive"

        # ====================================================
        # FIXED PARAMETERS (NON-TRAINABLE - STORED IN BUFFERS)
        # ====================================================

        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.n_cell_types = n_cell_types

        # Store cell and synapse parameter dictionaries
        self.cell_params = cell_params
        self.synapse_params = synapse_params

        if cell_params_FF is not None:
            self.cell_params_FF = cell_params_FF
            self.synapse_params_FF = synapse_params_FF
            self.n_cell_types_FF = n_cell_types_FF
        else:
            self.cell_params_FF = None
            self.synapse_params_FF = None
            self.n_cell_types_FF = None

        # Register weights and cell type indices for recurrent connections
        self.register_buffer("weights", torch.from_numpy(weights).float())
        self.register_buffer("cell_type_indices", torch.from_numpy(cell_type_indices))

        # Register feedforward structure (if provided)
        if weights_FF is not None:
            self.register_buffer("weights_FF", torch.from_numpy(weights_FF).float())
            self.register_buffer(
                "cell_type_indices_FF",
                torch.from_numpy(cell_type_indices_FF),
            )
        else:
            self.register_buffer("weights_FF", None)
            self.register_buffer("cell_type_indices_FF", None)

        # TODO: Create neuron-indexed arrays and synapse arrays from the parameter dictionaries
        # This will be implemented in the next step

        # ===========================================================
        # OPTIMIZABLE PARAMETERS (TRAINABLE - STORED AS nn.Parameter)
        # ===========================================================

        # Convert scaling factors to tensors and register as trainable parameters
        self.scaling_factors = nn.Parameter(
            torch.tensor(scaling_factors, dtype=torch.float32)
        )
        if scaling_factors_FF is not None:
            self.scaling_factors_FF = nn.Parameter(
                torch.tensor(scaling_factors_FF, dtype=torch.float32)
            )
        else:
            self.register_buffer("scaling_factors_FF", None)

        # ======================================================================
        # HYPERPARAMETERS (CONFIGURATION VALUES - STORED AS INSTANCE ATTRIBUTES)
        # ======================================================================

        self.surrgrad_scale = surrgrad_scale

    def _create_neuron_param_arrays(
        self,
        physiology_params: dict[str, dict[str, float]],
        cell_types: list[str],
        cell_type_indices: IntArray,
        n_neurons: int,
    ) -> dict[str, torch.Tensor]:
        """
        Create neuron-indexed parameter arrays from cell-type-specific parameters.

        Args:
            physiology_params (dict[str, dict[str, float]]): Nested dict {cell_type: {param_name: value}}.
            cell_types (list[str]): List of cell type names.
            cell_type_indices (IntArray): Array mapping each neuron to its cell type index.
            n_neurons (int): Total number of neurons.

        Returns:
            dict[str, torch.Tensor]: Dictionary of parameter names to torch tensors of shape (n_neurons,).
        """
        # Required physiological parameters for LIF neurons):
        required_param_names = [
            "tau_mem",  # Membrane time constant
            "R",  # Membrane resistance
            "U_rest",  # Resting potential
            "theta",  # Spike threshold
            "U_reset",  # Reset potential
        ]

        neuron_params = {}

        for param_name in required_param_names:
            # Build a lookup array: parameter value for each cell type
            param_lookup = np.array(
                [physiology_params[ct][param_name] for ct in cell_types],
                dtype=np.float32,
            )

            # Map each neuron to its parameter value using cell_type_indices
            param_array = param_lookup[cell_type_indices]
            neuron_params[param_name] = torch.from_numpy(param_array)

        return neuron_params

    def _create_synapse_param_arrays(
        self,
        physiology_params: dict[str, dict[str, float]],
        cell_types: list[str],
        cell_type_indices: IntArray,
        n_neurons: int,
        physiology_params_FF: dict[str, dict[str, float]] | None,
        cell_types_FF: list[str] | None,
        cell_type_indices_FF: IntArray | None,
        n_inputs: int,
    ) -> dict[str, torch.Tensor]:
        """
        Create synapse parameter arrays for tau_syn and tau_syn_FF.

        Args:
            physiology_params (dict[str, dict[str, float]]): Nested dict {cell_type: {param_name: value}}.
            cell_types (list[str]): List of cell type names.
            cell_type_indices (IntArray): Array mapping each neuron to its cell type index.
            n_neurons (int): Total number of neurons.
            physiology_params_FF (dict[str, dict[str, float]] | None): Nested dict for feedforward input cell types.
            cell_types_FF (list[str] | None): List of feedforward cell type names.
            cell_type_indices_FF (IntArray | None): Array mapping each input to its cell type index.
            n_inputs (int): Total number of inputs.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tau_syn and tau_syn_FF arrays.
        """
        synapse_params = {}

        # Create tau_syn for recurrent connections
        # tau_syn_lookup: (n_cell_types,)
        tau_syn_lookup = np.array(
            [physiology_params[ct]["tau_syn"] for ct in cell_types],
            dtype=np.float32,
        )
        # tau_syn_array: (n_cell_types, n_neurons)
        tau_syn_array = tau_syn_lookup[:, np.newaxis] * np.ones(
            (len(cell_types), n_neurons), dtype=np.float32
        )
        synapse_params["tau_syn"] = torch.from_numpy(tau_syn_array)

        # Create tau_syn_FF for feedforward connections (if provided)
        if physiology_params_FF is not None and cell_types_FF is not None:
            # tau_syn_FF_lookup: (n_cell_types_FF,)
            tau_syn_FF_lookup = np.array(
                [physiology_params_FF[ct]["tau_syn"] for ct in cell_types_FF],
                dtype=np.float32,
            )
            # tau_syn_FF_array: (n_cell_types_FF, n_neurons)
            tau_syn_FF_array = tau_syn_FF_lookup[:, np.newaxis] * np.ones(
                (len(cell_types_FF), n_neurons), dtype=np.float32
            )
            synapse_params["tau_syn_FF"] = torch.from_numpy(tau_syn_FF_array)

        return synapse_params

    @property
    def device(self):
        """Get the device the model is on"""
        return self.weights.device

    @property
    def scaled_weights(self) -> torch.Tensor:
        """
        Get the recurrent weights scaled by the scaling_factors matrix.

        The scaling_factors matrix is of shape (n_cell_types, n_cell_types) where
        scaling_factors[i, j] scales connections from cell type i to cell type j.

        Returns:
            torch.Tensor: Scaled recurrent weight matrix of shape (n_neurons, n_neurons).
        """
        source_types = self.cell_type_indices[:, None]  # shape (n_neurons, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors[
            source_types, target_types
        ]  # (n_neurons, n_neurons)
        return self.weights * scaling_matrix

    @property
    def scaled_weights_FF(self) -> torch.Tensor:
        """
        Get the feedforward weights scaled by the scaling_factors_FF matrix.

        The scaling_factors_FF matrix is of shape (n_cell_types_FF, n_cell_types) where
        scaling_factors_FF[i, j] scales connections from input cell type i to cell type j.

        Returns:
            torch.Tensor: Scaled feedforward weight matrix of shape (n_inputs, n_neurons).
        """
        if self.weights_FF is None:
            raise ValueError("weights_FF must be provided for feedforward scaling.")
        input_types = self.cell_type_indices_FF[:, None]  # shape (n_inputs, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors_FF[
            input_types, target_types
        ]  # (n_inputs, n_neurons)
        return self.weights_FF * scaling_matrix

    @property
    def cell_typed_weights(self) -> torch.Tensor:
        """
        Create a tiered structure for the recurrent weight matrix based on cell types.

        Each input row of `n_neurons` is placed on the level according to its index
        in `cell_type_indices`.

        Returns:
            torch.Tensor: Tiered weight matrix of shape
                (n_cell_types, n_neurons, n_neurons).
        """
        # Initialize a zero tensor for tiered weights
        n_tiers = self.n_cell_types
        tiered_weights = torch.zeros(
            (n_tiers, self.n_neurons, self.n_neurons),
            dtype=self.weights.dtype,
            device=self.weights.device,
        )

        # Create a boolean mask for tiering
        mask = self.cell_type_indices[None, :] == torch.arange(
            n_tiers, device=self.device
        ).view(-1, 1)  # Shape: (n_cell_types, n_neurons)

        # Broadcast the mask to tiered_weights and assign scaled weights
        tiered_weights = mask[:, :, None] * self.scaled_weights[None, :, :]

        return tiered_weights

    @property
    def cell_typed_weights_FF(self) -> torch.Tensor:
        """
        Create a tiered structure for the feedforward weight matrix based on input cell types.

        Each input row of `n_inputs` is placed on the level according to its index
        in `cell_type_indices_FF`.

        Returns:
            torch.Tensor: Tiered weight matrix of shape
                (n_cell_types_FF, n_inputs, n_neurons).
        """
        if self.weights_FF is None or self.cell_type_indices_FF is None:
            raise ValueError(
                "Feedforward weights and cell type indices must be provided."
            )

        # Initialize a zero tensor for tiered weights
        n_tiers = self.n_cell_types_FF
        tiered_weights_FF = torch.zeros(
            (n_tiers, self.n_inputs, self.n_neurons),
            dtype=self.weights_FF.dtype,
            device=self.weights_FF.device,
        )

        # Create a boolean mask for tiering
        mask = self.cell_type_indices_FF[None, :] == torch.arange(
            n_tiers, device=self.device
        ).view(-1, 1)  # Shape: (n_cell_types_FF, n_inputs)

        # Broadcast the mask to tiered_weights_FF and assign scaled feedforward weights
        tiered_weights_FF = mask[:, :, None] * self.scaled_weights_FF[None, :, :]

        return tiered_weights_FF

    @property
    def spike_fn(self):
        """
        Get the surrogate gradient spike function with the current scale parameter.

        Returns:
            Callable: Partial function for SurrGradSpike with configured scale.
        """
        return lambda x: SurrGradSpike.apply(x, self.surrgrad_scale)
