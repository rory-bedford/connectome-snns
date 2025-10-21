"""I/O utilities for LIF network parameters"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from optimisation.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class CurrentLIFNetwork_IO(nn.Module):
    """Base class for LIF network with I/O functionality and parameter management."""

    def __init__(
        self,
        weights: FloatArray,
        cell_types: list[str],
        cell_type_indices: IntArray,
        physiology_params: dict[str, dict[str, float]],
        scaling_factors: FloatArray,
        surrgrad_scale: float,
        weights_FF: FloatArray | None = None,
        cell_types_FF: list[str] | None = None,
        cell_type_indices_FF: IntArray | None = None,
        scaling_factors_FF: FloatArray | None = None,
    ):
        """
        Initialize the LIF network with explicit parameters.

        Args:
            weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            cell_types (list[str]): List of cell type names (e.g., ["excitatory", "inhibitory"]).
            cell_type_indices (IntArray): Array of shape (n_neurons,) with cell type indices (0, 1, 2, ...).
            physiology_params (dict[str, dict[str, float]]): Nested dict {cell_type: {param_name: value}}.
            scaling_factors (FloatArray): Matrix of shape (n_cell_types, n_cell_types) for recurrent scaling.
            surrgrad_scale (float): Scale parameter for surrogate gradient function.
            weights_FF (FloatArray | None): Feedforward weight matrix of shape (n_inputs, n_neurons) or None.
            cell_types_FF (list[str] | None): List of feedforward cell type names.
            cell_type_indices_FF (IntArray | None): Array of feedforward cell type indices.
            scaling_factors_FF (FloatArray | None): Matrix of shape (n_cell_types_FF, n_cell_types) for feedforward scaling.
        """
        super(CurrentLIFNetwork_IO, self).__init__()

        # =================================
        # PARAMETER VALIDATION & ASSERTIONS
        # =================================

        # Extract key dimensions for validation
        n_neurons = weights.shape[0] if weights.ndim == 2 else 0
        n_cell_types = len(cell_types)
        n_inputs = (
            weights_FF.shape[0]
            if weights_FF is not None and weights_FF.ndim == 2
            else 0
        )
        n_cell_types_FF = len(cell_types_FF) if cell_types_FF is not None else 0

        # Basic type and shape validation
        assert isinstance(cell_types, list), "cell_types must be a list"
        assert len(cell_types) > 0, "cell_types must not be empty"
        assert all(isinstance(ct, str) for ct in cell_types), (
            "All cell_types must be strings"
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

        # Physiology parameters validation
        assert isinstance(physiology_params, dict), (
            "physiology_params must be a dictionary"
        )
        for cell_type in cell_types:
            assert cell_type in physiology_params, (
                f"Missing physiology parameters for cell type '{cell_type}'"
            )
            assert isinstance(physiology_params[cell_type], dict), (
                f"Physiology parameters for '{cell_type}' must be a dictionary"
            )

        # Feedforward weights validation (all-or-nothing)
        ff_args = [weights_FF, cell_types_FF, cell_type_indices_FF, scaling_factors_FF]
        ff_provided = [arg is not None for arg in ff_args]
        if any(ff_provided):
            assert all(ff_provided), (
                "If any feedforward argument is provided, all of "
                "weights_FF, cell_types_FF, cell_type_indices_FF, and scaling_factors_FF must be provided."
            )
            assert weights_FF.ndim == 2, "Feedforward weights must be 2D matrix"
            # Updated asserts to use n_inputs and n_cell_types_FF for readability
            assert weights_FF.shape[1] == n_neurons, (
                f"Feedforward weights output dimension ({weights_FF.shape[1]}) must match number of neurons ({n_neurons})"
            )
            assert cell_type_indices_FF.ndim == 1, (
                "Feedforward cell type indices must be 1D array"
            )
            assert cell_type_indices_FF.shape[0] == n_inputs, (
                f"Feedforward cell type indices length ({cell_type_indices_FF.shape[0]}) must match number of inputs ({n_inputs})"
            )
            assert isinstance(cell_types_FF, list), "cell_types_FF must be a list"
            assert len(cell_types_FF) > 0, (
                "cell_types_FF must not be empty when provided"
            )
            assert all(isinstance(ct, str) for ct in cell_types_FF), (
                "All cell_types_FF must be strings"
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

        # Create neuron-indexed arrays for physiological parameters
        neuron_params = self._create_neuron_param_arrays(
            physiology_params, cell_types, cell_type_indices, n_neurons
        )

        # Register network structure (connectivity matrices and dimensions)
        self.register_buffer("weights", torch.from_numpy(weights).float())

        # Register feedforward structure (if provided)
        if weights_FF is not None:
            self.register_buffer("weights_FF", torch.from_numpy(weights_FF).float())
            if cell_type_indices_FF is not None:
                self.register_buffer(
                    "cell_type_indices_FF",
                    torch.from_numpy(cell_type_indices_FF).long(),
                )
        else:
            self.weights_FF = None
            self.cell_type_indices_FF = None

        # Register cell type indices for efficient scaling
        self.register_buffer(
            "cell_type_indices", torch.from_numpy(cell_type_indices).long()
        )
        if weights_FF is not None and cell_type_indices_FF is not None:
            self.register_buffer(
                "cell_type_indices_FF", torch.from_numpy(cell_type_indices_FF).long()
            )

        # Register physiological parameters as neuron-indexed arrays
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # ===========================================================
        # OPTIMIZABLE PARAMETERS (TRAINABLE - STORED AS nn.Parameter)
        # ===========================================================

        # Convert scaling factors to tensors and register as trainable parameters
        scaling_factors_tensor = torch.tensor(scaling_factors, dtype=torch.float32)
        self.scaling_factors = nn.Parameter(scaling_factors_tensor)

        if scaling_factors_FF is not None:
            scaling_factors_FF_tensor = torch.tensor(
                scaling_factors_FF, dtype=torch.float32
            )
            self.scaling_factors_FF = nn.Parameter(scaling_factors_FF_tensor)
        else:
            self.scaling_factors_FF = None

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
        # Required physiological parameters for LIF neurons:
        required_param_names = [
            "tau_mem",  # Membrane time constant
            "tau_syn",  # Synaptic time constant
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

    @property
    def device(self):
        """Get the device the model is on"""
        return self.recurrent_weights.device

    @property
    def scaled_weights(self) -> torch.Tensor:
        """
        Get the recurrent weights scaled by the scaling_factors matrix.

        The scaling_factors matrix is of shape (n_cell_types, n_cell_types) where
        scaling_factors[i, j] scales connections from cell type i to cell type j.

        Returns:
            torch.Tensor: Scaled recurrent weight matrix.
        """
        source_types = self.cell_type_indices[:, None]  # shape (n_neurons, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors[source_types, target_types]
        return self.weights * scaling_matrix

    @property
    def scaled_weights_FF(self) -> torch.Tensor:
        """
        Get the feedforward weights scaled by the scaling_factors_FF matrix.

        The scaling_factors_FF matrix is of shape (n_cell_types_FF, n_cell_types) where
        scaling_factors_FF[i, j] scales connections from input cell type i to cell type j.

        Returns:
            torch.Tensor: Scaled feedforward weight matrix.
        """
        if self.weights_FF is None:
            raise ValueError("weights_FF must be provided for feedforward scaling.")
        if (
            not hasattr(self, "cell_type_indices_FF")
            or self.cell_type_indices_FF is None
        ):
            raise ValueError(
                "cell_type_indices_FF must be provided for feedforward scaling."
            )
        if self.scaling_factors_FF is None:
            raise ValueError(
                "scaling_factors_FF must be provided for feedforward scaling."
            )

        input_types = self.cell_type_indices_FF[:, None]  # shape (n_inputs, 1)
        target_types = self.cell_type_indices[None, :]  # shape (1, n_neurons)
        scaling_matrix = self.scaling_factors_FF[input_types, target_types]
        return self.weights_FF * scaling_matrix

    @property
    def spike_fn(self):
        """
        Get the surrogate gradient spike function with the current scale parameter.

        Returns:
            Callable: Partial function for SurrGradSpike with configured scale.
        """
        return lambda x: SurrGradSpike.apply(x, self.surrgrad_scale)
