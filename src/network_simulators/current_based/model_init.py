"""I/O utilities for recurrent current-based LIF network parameters"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from typing import Literal

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]
OptimisableParams = Literal[
    "weights",
    "scaling_factors",
    "scaling_factors_recurrent",
    "scaling_factors_feedforward",
    None,
]


class CurrentLIFNetwork_IO(nn.Module):
    """Base class for recurrent current-based LIF network with I/O functionality and parameter management.

    Uses natural units: membrane potential varies from 0 (reset) to 1 (threshold), resistance = 1.
    """

    def __init__(
        self,
        dt: float,
        weights: FloatArray,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors: FloatArray | None = None,
        scaling_factors_FF: FloatArray | None = None,
        optimisable: OptimisableParams = None,
        connectome_mask: FloatArray | None = None,
        feedforward_mask: FloatArray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = True,
    ):
        """
        Initialize the recurrent current-based LIF network with explicit parameters.

        Args:
            dt (float): The timestep in milliseconds (ms).
            weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            weights_FF (FloatArray): Feedforward weight matrix of shape (n_inputs, n_neurons).
            cell_type_indices (IntArray): Array of shape (n_neurons,) with postsynaptic cell type indices (0, 1, 2, ...).
            cell_type_indices_FF (IntArray): Array of feedforward/input cell type indices.
            batch_size (int): Number of parallel simulations to run (batch dimension).
            cell_params (list[dict]): List of postsynaptic cell type parameter dicts. Each dict contains:
                - 'name' (str): Cell type name (e.g., 'excitatory', 'inhibitory')
                - 'cell_id' (int): Cell type ID (0, 1, 2, ...)
                - 'tau_mem' (float): Membrane time constant (ms)
                - 'tau_ref' (float): Refractory period (ms)
            synapse_params (list[dict]): List of recurrent synapse parameter dicts. Each dict contains:
                - 'name' (str): Synapse type name (e.g., 'cholinergic', 'gabaergic')
                - 'synapse_id' (int): Synapse type ID (0, 1, 2, ...)
                - 'cell_id' (int): Presynaptic cell type ID that produces this synapse
                - 'tau_syn' (float): Synaptic time constant (ms)
                - 'type' (str): 'excitatory' or 'inhibitory' (determines current sign)
            synapse_params_FF (list[dict]): List of feedforward synapse parameter dicts (same structure as synapse_params).
            surrgrad_scale (float): Scale parameter for surrogate gradient fast sigmoid function.
            scaling_factors (FloatArray | None): Matrix of shape (n_cell_types, n_cell_types) for recurrent scaling.
                If None, initialized to ones (no scaling).
            scaling_factors_FF (FloatArray | None): Matrix of shape (n_cell_types_FF, n_cell_types) for feedforward scaling.
                If None, initialized to ones (no scaling).
            optimisable (OptimisableParams): What to optimise during training. Options:
                - "weights": Optimise connection weights (weights and weights_FF)
                - "scaling_factors": Optimise both recurrent and feedforward scaling factors
                - "scaling_factors_recurrent": Optimise only recurrent scaling factors
                - "scaling_factors_feedforward": Optimise only feedforward scaling factors
                - None: Don't optimise anything (all parameters are fixed) [default]
            connectome_mask (FloatArray | None): Boolean mask for recurrent connections of shape (n_neurons, n_neurons).
                Required if optimisable="weights". If None, computed from weights != 0.
            feedforward_mask (FloatArray | None): Boolean mask for feedforward connections of shape (n_inputs, n_neurons).
                Required if optimisable="weights". If None, computed from weights_FF != 0.
            track_variables (bool): Whether to accumulate and return internal state variables (v, I_syn, I_syn_rec) over time.
                When False (default), only spikes are returned and memory usage is minimized.
                When True, all variables are tracked and returned as a dict for analysis/visualization.
            track_gradients (bool): Whether to track gradients for all intermediate variables.
            track_batch_idx (int | None): Which batch index to track when track_variables=True.
                If None (default), tracks all batch elements. If an integer, only tracks that specific
                batch index, reducing memory usage by a factor of batch_size. Useful for visualization
                where only one example is needed. Ignored when track_variables=False.
            use_tqdm (bool): Whether to display tqdm progress bar during forward pass. Default is True.
        """
        super(CurrentLIFNetwork_IO, self).__init__()

        # Store optimisation mode
        self.optimisable = optimisable

        # Store batch size and tracking preference
        self.batch_size = batch_size
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
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
        # For current-based model, n_cell_types_FF is inferred from cell_type_indices_FF
        self.n_cell_types_FF = int(cell_type_indices_FF.max()) + 1
        self.n_synapse_types_FF = len(synapse_params_FF)

        self.n_neurons = weights.shape[0]
        self.n_inputs = weights_FF.shape[0]

        # Store cell parameter dictionaries
        self.cell_params = cell_params
        self.synapse_params = synapse_params
        self.synapse_params_FF = synapse_params_FF

        # Validate and prepare weights masks
        if self.optimisable == "weights":
            assert connectome_mask is not None, (
                "connectome_mask is required when optimisable='weights'"
            )
            assert feedforward_mask is not None, (
                "feedforward_mask is required when optimisable='weights'"
            )
            weights_mask = connectome_mask
            weights_mask_FF = feedforward_mask
        else:
            # If not optimizing weights, derive masks from non-zero entries if not provided
            weights_mask = (
                connectome_mask if connectome_mask is not None else (weights != 0)
            )
            weights_mask_FF = (
                feedforward_mask if feedforward_mask is not None else (weights_FF != 0)
            )

        # Create and register weights masks (always non-trainable)
        self.register_buffer("weights_mask", torch.from_numpy(weights_mask))
        self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

        # Register weights with log parameterization if optimising
        if self.optimisable == "weights":
            # Extract only non-masked weights and store as flat 1D log-space arrays
            eps = 1e-8
            non_masked_weights = weights[weights_mask]
            non_masked_weights_FF = weights_FF[weights_mask_FF]

            # Convert to log space
            log_weights_flat = np.log(non_masked_weights + eps)
            log_weights_FF_flat = np.log(non_masked_weights_FF + eps)

            self._register_parameter_or_buffer(
                "log_weights_flat", log_weights_flat, trainable=True
            )
            self._register_parameter_or_buffer(
                "log_weights_FF_flat", log_weights_FF_flat, trainable=True
            )
        else:
            # Store linear-space parameters with internal names to avoid conflict with @property
            self._register_parameter_or_buffer(
                "_weights_buffer", weights, trainable=False
            )
            self._register_parameter_or_buffer(
                "_weights_FF_buffer", weights_FF, trainable=False
            )

        # Register cell type indices (always non-trainable)
        self._register_parameter_or_buffer(
            "cell_type_indices", cell_type_indices, trainable=False
        )
        self._register_parameter_or_buffer(
            "cell_type_indices_FF", cell_type_indices_FF, trainable=False
        )

        # Create neuron-indexed arrays from cell parameters
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)

        # Register physiological parameters as neuron-indexed buffers
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Create synapse parameter arrays for recurrent connections
        synapse_param_arrays = self._create_synapse_param_arrays(synapse_params)

        # Register recurrent synapse parameters as buffers
        for param_name, param_array in synapse_param_arrays.items():
            self.register_buffer(param_name, param_array)

        # Create synapse parameter arrays for feedforward connections
        synapse_param_arrays_FF = self._create_synapse_param_arrays(synapse_params_FF)

        # Register feedforward synapse parameters as buffers
        for param_name, param_array in synapse_param_arrays_FF.items():
            self.register_buffer(param_name + "_FF", param_array)

        # Create synapse-to-cell and cell-to-synapse mapping arrays
        self._create_synapse_to_cell_mappings(synapse_params)
        self._create_synapse_to_cell_mappings_FF(synapse_params_FF)
        self._create_cell_to_synapse_masks(synapse_params)
        self._create_cell_to_synapse_masks_FF(synapse_params_FF)
        self._create_cell_type_masks(synapse_params)
        self._create_cell_type_masks_FF(synapse_params_FF)

        # ===========================================================
        # OPTIMISABLE PARAMETERS (TRAINABLE - STORED AS nn.Parameter)
        # ===========================================================

        # Register scaling factors (trainable based on optimisable mode)
        # Initialize to ones if not provided
        if scaling_factors is None:
            scaling_factors = np.ones(
                (self.n_cell_types, self.n_cell_types), dtype=np.float32
            )

        self._register_parameter_or_buffer(
            "scaling_factors",
            scaling_factors,
            trainable=(
                self.optimisable in ["scaling_factors", "scaling_factors_recurrent"]
            ),
        )

        if scaling_factors_FF is None:
            scaling_factors_FF = np.ones(
                (self.n_cell_types_FF, self.n_cell_types), dtype=np.float32
            )

        self._register_parameter_or_buffer(
            "scaling_factors_FF",
            scaling_factors_FF,
            trainable=(
                self.optimisable in ["scaling_factors", "scaling_factors_feedforward"]
            ),
        )

        # ======================================================================
        # HYPERPARAMETERS (CONFIGURATION VALUES - STORED AS INSTANCE ATTRIBUTES)
        # ======================================================================

        # Convert surrgrad_scale to tensor for JIT compatibility
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )

        # Pre-compute cached weight matrices split by input cell type
        self._create_cached_weights()

        # Initialize timestep-dependent parameters
        self.set_timestep(dt)

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state variables to initial conditions.

        This method resets the membrane potentials (v), recurrent synaptic currents (I_syn),
        and feedforward synaptic currents (I_syn_FF) to their initial states. Call this before
        starting independent simulations or when changing batch size.

        State variables after reset:
        - self.v: Membrane potentials set to 0 (natural units: 0 = reset, 1 = threshold)
        - self.I_syn: Recurrent synaptic currents set to zeros
        - self.I_syn_FF: Feedforward synaptic currents set to zeros

        Args:
            batch_size (int | None): New batch size for state tensors. If None, uses self.batch_size.
                Use this parameter when switching between training (larger batches) and
                visualization (typically batch_size=1).

        Example:
            >>> # Reset for new independent simulation
            >>> model.reset_state()
            >>>
            >>> # Change batch size for visualization
            >>> model.reset_state(batch_size=1)
        """
        if batch_size is not None:
            self.batch_size = batch_size

        # Initialize membrane potentials to 0 (reset potential in natural units)
        # Shape: (batch_size, n_neurons)
        v = torch.zeros(
            (self.batch_size, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("v", v, persistent=False)

        # Initialize recurrent synaptic currents to zeros
        # Shape: (batch_size, n_neurons, n_synapse_types)
        I_syn = torch.zeros(
            (self.batch_size, self.n_neurons, self.n_synapse_types),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("I_syn", I_syn, persistent=False)

        # Initialize feedforward synaptic currents to zeros
        # Shape: (batch_size, n_neurons, n_synapse_types_FF)
        I_syn_FF = torch.zeros(
            (self.batch_size, self.n_neurons, self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("I_syn_FF", I_syn_FF, persistent=False)

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
                - Physiological parameters (tau_mem, tau_ref)
            cell_type_indices (IntArray): Array of shape (n_neurons,) mapping each neuron to its cell type index.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping parameter names to torch tensors of shape (n_neurons,).
                Each tensor contains the parameter value for each neuron based on its cell type.
        """
        # Required physiological parameters for current-based LIF neurons
        required_param_names = [
            "tau_mem",  # Membrane time constant (ms)
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

        return neuron_params

    def _create_synapse_param_arrays(
        self,
        synapse_params: list[dict],
    ) -> dict[str, torch.Tensor]:
        """
        Create synapse-indexed parameter arrays from synapse-type-specific parameters.

        Args:
            synapse_params (list[dict]): List of synapse parameter dicts, each containing:
                - 'synapse_id': Synapse type ID (must be 0-indexed and contiguous)
                - 'name': Synapse type name
                - 'tau_syn': Synaptic time constant (ms)
                - 'type': Synapse type ('excitatory' or 'inhibitory')

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping parameter names to torch tensors of shape (n_synapse_types,).
                Includes 'tau_syn' and 'sign' (1.0 for excitatory, -1.0 for inhibitory)
        """
        required_param_names = [
            "tau_syn",  # Synaptic time constant (ms)
        ]

        n_synapse_types = len(synapse_params)
        synapse_param_dict = {}

        # Initialize parameter lookup array
        for param_name in required_param_names:
            param_lookup = np.zeros(n_synapse_types, dtype=np.float32)
            for synapse in synapse_params:
                synapse_id = synapse["synapse_id"]
                param_lookup[synapse_id] = synapse[param_name]

            synapse_param_dict[param_name] = torch.from_numpy(param_lookup)

        # Create sign array: +1 for excitatory, -1 for inhibitory
        sign_lookup = np.zeros(n_synapse_types, dtype=np.float32)
        for synapse in synapse_params:
            synapse_id = synapse["synapse_id"]
            sign_lookup[synapse_id] = 1.0 if synapse["type"] == "excitatory" else -1.0
        synapse_param_dict["sign"] = torch.from_numpy(sign_lookup)

        return synapse_param_dict

    def _create_synapse_to_cell_mappings(self, synapse_params: list[dict]) -> None:
        """
        Create and register lookup array mapping recurrent synapse IDs to their parent cell type IDs.

        Args:
            synapse_params: List of recurrent synapse parameter dicts with 'synapse_id' and 'cell_id'
        """
        # Create mapping from synapse_id to cell_id for recurrent connections
        synapse_to_cell_mapping = np.zeros(self.n_synapse_types, dtype=np.int64)
        for synapse in synapse_params:
            synapse_to_cell_mapping[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id", torch.from_numpy(synapse_to_cell_mapping)
        )

    def _create_synapse_to_cell_mappings_FF(
        self, synapse_params_FF: list[dict]
    ) -> None:
        """
        Create and register lookup array mapping feedforward synapse IDs to their parent cell type IDs.

        Args:
            synapse_params_FF: List of feedforward synapse parameter dicts with 'synapse_id' and 'cell_id'
        """
        # Create mapping from synapse_id to cell_id for feedforward connections
        synapse_to_cell_mapping_FF = np.zeros(self.n_synapse_types_FF, dtype=np.int64)
        for synapse in synapse_params_FF:
            synapse_to_cell_mapping_FF[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id_FF", torch.from_numpy(synapse_to_cell_mapping_FF)
        )

    def _create_cell_to_synapse_masks(self, synapse_params: list[dict]) -> None:
        """
        Create boolean masks for mapping recurrent cell types to their associated synapse types.

        Args:
            synapse_params: List of recurrent synapse parameter dicts
        """
        # Get unique cell type IDs from recurrent synapses
        recurrent_cell_types = sorted(
            set(synapse["cell_id"] for synapse in synapse_params)
        )
        max_recurrent_cell_type = (
            max(recurrent_cell_types) if recurrent_cell_types else -1
        )

        # Create boolean mask for recurrent cell types -> synapse types
        # Shape: (max_recurrent_cell_type + 1, n_synapse_types)
        cell_to_synapse_mask = torch.zeros(
            max_recurrent_cell_type + 1, self.n_synapse_types, dtype=torch.bool
        )
        for cell_type in recurrent_cell_types:
            cell_to_synapse_mask[cell_type, :] = self.synapse_to_cell_id == cell_type
        self.register_buffer("cell_to_synapse_mask", cell_to_synapse_mask)

    def _create_cell_to_synapse_masks_FF(self, synapse_params_FF: list[dict]) -> None:
        """
        Create boolean masks for mapping feedforward cell types to their associated synapse types.

        Args:
            synapse_params_FF: List of feedforward synapse parameter dicts
        """
        # Get unique cell type IDs from feedforward synapses
        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))
        max_ff_cell_type = max(ff_cell_types) if ff_cell_types else -1

        # Create boolean mask for feedforward cell types -> synapse types
        # Shape: (max_ff_cell_type + 1, n_synapse_types_FF)
        cell_to_synapse_mask_FF = torch.zeros(
            max_ff_cell_type + 1, self.n_synapse_types_FF, dtype=torch.bool
        )
        for cell_type in ff_cell_types:
            cell_to_synapse_mask_FF[cell_type, :] = (
                self.synapse_to_cell_id_FF == cell_type
            )
        self.register_buffer("cell_to_synapse_mask_FF", cell_to_synapse_mask_FF)

    def _create_cell_type_masks(self, synapse_params: list[dict]) -> None:
        """
        Create boolean masks for efficient neuron indexing by cell type (recurrent).

        Creates:
        - cell_type_masks: List of boolean tensors indicating which neurons belong to each recurrent cell type

        Args:
            synapse_params: List of recurrent synapse parameter dicts
        """
        # Get unique recurrent cell types
        recurrent_cell_types = sorted(
            set(synapse["cell_id"] for synapse in synapse_params)
        )

        # Precompute boolean masks for each recurrent cell type
        cell_type_masks = []
        for cell_type in recurrent_cell_types:
            cell_type_masks.append(self.cell_type_indices == cell_type)
        self.cell_type_masks = cell_type_masks

    def _create_cell_type_masks_FF(self, synapse_params_FF: list[dict]) -> None:
        """
        Create boolean masks for efficient input indexing by cell type (feedforward).

        Creates:
        - cell_type_masks_FF: List of boolean tensors indicating which inputs belong to each feedforward cell type

        Args:
            synapse_params_FF: List of feedforward synapse parameter dicts
        """
        # Get unique feedforward cell types
        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))

        # Precompute boolean masks for each feedforward cell type
        cell_type_masks_FF = []
        for cell_type in ff_cell_types:
            cell_type_masks_FF.append(self.cell_type_indices_FF == cell_type)
        self.cell_type_masks_FF = cell_type_masks_FF

    def _create_cached_weights(self) -> None:
        """
        Precompute weight matrices split by input cell type and synapse type.

        For each cell type (recurrent and feedforward), creates weight matrices that are:
        - Non-zero for inputs/neurons of that cell type
        - Zero everywhere else
        - Split by synapse type (using cell_to_synapse_mask)

        This enables efficient computation of currents per synapse type in the forward loop.

        Stores:
        - cached_weights_rec: List of precomputed recurrent weight tensors
        - cached_weights_rec_masks: List of neuron masks (which neurons belong to this cell type)
        - cached_weights_rec_syn_masks: List of synapse masks (which synapse types this cell type produces)
        - cached_weights_ff: List of precomputed feedforward weight tensors
        - cached_weights_ff_masks: List of input masks (which inputs belong to this cell type)
        - cached_weights_ff_syn_masks: List of synapse masks (which synapse types this cell type produces)
        """
        # Store metadata
        self.cached_weights_rec_masks = []
        self.cached_weights_rec_syn_masks = []
        self.cached_weights_ff_masks = []
        self.cached_weights_ff_syn_masks = []

        # Create cached weights for each recurrent cell type
        for k, mask in enumerate(self.cell_type_masks):
            # Get which synapse types this cell type produces
            cell_id = int(self.cell_type_indices[mask][0])  # Get the actual cell_id
            syn_mask = self.cell_to_synapse_mask[cell_id, :]

            if syn_mask.any():
                # Create weight matrix for this cell type
                # Shape: (n_neurons_in_type, n_neurons, n_synapse_types_in_mask)
                # weights[mask, :] has shape (n_neurons_in_type, n_neurons)
                # We need to expand it to include synapse dimension
                weights_for_type = self.weights[mask, :][
                    :, :, None
                ]  # (n_neurons, n_neurons, 1)

                # Only keep the synapse types this cell type produces
                # Broadcast to match synapse types
                n_syn_in_mask = syn_mask.sum().item()
                weights_product = weights_for_type.expand(-1, -1, n_syn_in_mask)

                # Apply scaling factors if not optimizing them
                if self.optimisable not in [
                    "scaling_factors",
                    "scaling_factors_recurrent",
                ]:
                    # scaling_factors has shape (n_cell_types, n_cell_types)
                    # We need to apply it per neuron based on cell_type_indices
                    scaling = self.scaling_factors[
                        cell_id, self.cell_type_indices
                    ]  # (n_neurons,)
                    weights_product = weights_product * scaling[None, :, None]

                # Register as buffer
                self.register_buffer(
                    f"cached_rec_{k}", weights_product, persistent=False
                )
                self.cached_weights_rec_masks.append(mask)
                self.cached_weights_rec_syn_masks.append(syn_mask)

        # Create cached weights for each feedforward cell type
        for k, mask in enumerate(self.cell_type_masks_FF):
            # Get which synapse types this cell type produces
            cell_id = int(self.cell_type_indices_FF[mask][0])  # Get the actual cell_id
            syn_mask = self.cell_to_synapse_mask_FF[cell_id, :]

            if syn_mask.any():
                # Create weight matrix for this cell type
                # Shape: (n_inputs_in_type, n_neurons, n_synapse_types_in_mask)
                # weights_FF[mask, :] has shape (n_inputs_in_type, n_neurons)
                # We need to expand it to include synapse dimension
                weights_for_type = self.weights_FF[mask, :][
                    :, :, None
                ]  # (n_inputs, n_neurons, 1)

                # Only keep the synapse types this cell type produces
                # Broadcast to match synapse types
                n_syn_in_mask = syn_mask.sum().item()
                weights_product = weights_for_type.expand(-1, -1, n_syn_in_mask)

                # Apply scaling factors if not optimizing them
                if self.optimisable not in [
                    "scaling_factors",
                    "scaling_factors_feedforward",
                ]:
                    # scaling_factors_FF has shape (n_cell_types_FF, n_cell_types)
                    # We need to apply it per neuron based on cell_type_indices
                    scaling = self.scaling_factors_FF[
                        cell_id, self.cell_type_indices
                    ]  # (n_neurons,)
                    weights_product = weights_product * scaling[None, :, None]

                # Register as buffer
                self.register_buffer(
                    f"cached_ff_{k}", weights_product, persistent=False
                )
                self.cached_weights_ff_masks.append(mask)
                self.cached_weights_ff_syn_masks.append(syn_mask)

    @property
    def weights(self) -> torch.Tensor:
        """Get recurrent weights in linear space (converts from log if optimising weights)."""
        if self.optimisable == "weights":
            return torch.zeros_like(
                self.weights_mask, dtype=torch.float32
            ).masked_scatter_(self.weights_mask, torch.exp(self.log_weights_flat))
        else:
            return self._buffers["_weights_buffer"]

    @property
    def weights_FF(self) -> torch.Tensor:
        """Get feedforward weights in linear space (converts from log if optimising weights)."""
        if self.optimisable == "weights":
            return torch.zeros_like(
                self.weights_mask_FF, dtype=torch.float32
            ).masked_scatter_(self.weights_mask_FF, torch.exp(self.log_weights_FF_flat))
        else:
            return self._buffers["_weights_FF_buffer"]

    @property
    def device(self):
        """Get the device the model is on"""
        if self.optimisable == "weights":
            return self.log_weights_flat.device
        elif self.optimisable in [
            "scaling_factors",
            "scaling_factors_recurrent",
            "scaling_factors_feedforward",
        ]:
            return self.scaling_factors.device
        else:
            # Get device from weights buffer
            return self._buffers["_weights_buffer"].device

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

        # Precompute decay factors for membrane and synapse
        alpha_syn = torch.exp(-self.dt / self.tau_syn)  # Shape (n_synapse_types,)
        self.register_buffer("alpha_syn", alpha_syn)

        alpha_syn_FF = torch.exp(
            -self.dt / self.tau_syn_FF
        )  # Shape (n_synapse_types_FF,)
        self.register_buffer("alpha_syn_FF", alpha_syn_FF)

        beta_mem = torch.exp(-self.dt / self.tau_mem)  # Shape (n_neurons,)
        self.register_buffer("beta_mem", beta_mem)

        # Initialize internal state variables
        self.reset_state()

    def _validate(
        self,
        dt: float,
        weights: FloatArray,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        scaling_factors: FloatArray | None,
        scaling_factors_FF: FloatArray | None,
        surrgrad_scale: float,
    ) -> None:
        """
        Validate all input parameters.

        Args:
            dt: Simulation timestep in milliseconds
            weights: Recurrent weight matrix
            weights_FF: Feedforward weight matrix
            cell_type_indices: Neuron-to-cell-type mapping
            cell_type_indices_FF: Input-to-cell-type mapping
            cell_params: List of postsynaptic cell parameter dicts
            synapse_params: List of recurrent synapse parameter dicts
            synapse_params_FF: List of feedforward synapse parameter dicts
            scaling_factors: Cell-type-to-cell-type scaling matrix (optional)
            scaling_factors_FF: Input-cell-type-to-cell-type scaling matrix (optional)
            surrgrad_scale: Surrogate gradient scale
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
        # POSTSYNAPTIC CELL PARAMS VALIDATION
        # ========================================
        assert isinstance(cell_params, list), "cell_params must be a list"
        assert len(cell_params) > 0, "cell_params must not be empty"
        assert all(isinstance(p, dict) for p in cell_params), (
            "All cell_params entries must be dicts"
        )

        # Required parameters for current-based LIF
        required_cell_param_names = ["cell_id", "name", "tau_mem", "tau_ref"]
        for cell in cell_params:
            for param_name in required_cell_param_names:
                assert param_name in cell, (
                    f"Missing required parameter '{param_name}' in cell_params entry: {cell}"
                )

        # ========================================
        # RECURRENT SYNAPSE PARAMS VALIDATION
        # ========================================
        assert isinstance(synapse_params, list), "synapse_params must be a list"
        assert len(synapse_params) > 0, "synapse_params must not be empty"
        assert all(isinstance(p, dict) for p in synapse_params), (
            "All synapse_params entries must be dicts"
        )

        # Required parameters for recurrent synapses
        required_synapse_param_names = [
            "synapse_id",
            "name",
            "cell_id",
            "tau_syn",
            "type",
        ]
        for synapse in synapse_params:
            for param_name in required_synapse_param_names:
                assert param_name in synapse, (
                    f"Missing required parameter '{param_name}' in synapse_params entry: {synapse}"
                )
            # Validate synapse type
            assert synapse["type"] in ["excitatory", "inhibitory"], (
                f"Synapse type must be 'excitatory' or 'inhibitory', got '{synapse['type']}' in synapse_params entry: {synapse}"
            )

        # Extract n_synapse_types from synapse_params by finding max synapse_id
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
        # FEEDFORWARD SYNAPSE PARAMS VALIDATION
        # ========================================
        assert isinstance(synapse_params_FF, list), "synapse_params_FF must be a list"
        assert len(synapse_params_FF) > 0, "synapse_params_FF must not be empty"
        assert all(isinstance(p, dict) for p in synapse_params_FF), (
            "All synapse_params_FF entries must be dicts"
        )

        # Required parameters for feedforward synapses
        for synapse in synapse_params_FF:
            for param_name in required_synapse_param_names:
                assert param_name in synapse, (
                    f"Missing required parameter '{param_name}' in synapse_params_FF entry: {synapse}"
                )
            # Validate synapse type
            assert synapse["type"] in ["excitatory", "inhibitory"], (
                f"Synapse type must be 'excitatory' or 'inhibitory', got '{synapse['type']}' in synapse_params_FF entry: {synapse}"
            )

        # Extract n_synapse_types_FF from synapse_params_FF by finding max synapse_id
        synapse_ids_FF = [params["synapse_id"] for params in synapse_params_FF]
        n_synapse_types_FF = max(synapse_ids_FF) + 1

        # Ensure synapse_ids are 0-indexed, contiguous, and complete
        synapse_ids_array = np.array(synapse_ids_FF, dtype=np.int32)
        expected_synapse_ids = np.arange(n_synapse_types_FF, dtype=np.int32)
        assert np.array_equal(np.sort(synapse_ids_array), expected_synapse_ids), (
            f"synapse_ids in synapse_params_FF must be 0-indexed and contiguous [0, 1, ..., {n_synapse_types_FF - 1}]. "
            f"Found: {sorted(synapse_ids_FF)}, Expected: {expected_synapse_ids.tolist()}"
        )

        # Ensure no duplicate synapse_ids
        assert len(synapse_ids_FF) == len(np.unique(synapse_ids_array)), (
            f"Duplicate synapse_ids found in synapse_params_FF: {synapse_ids_FF}"
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
        # RECURRENT WEIGHTS VALIDATION
        # ========================================
        assert weights.ndim == 2, "Recurrent weights must be 2D matrix"
        assert weights.shape[0] == weights.shape[1], (
            "Recurrent weights must be square matrix"
        )
        assert n_neurons > 0, "Number of neurons must be positive"
        # Weights must be positive (sign comes from synapse type)
        assert np.all(weights >= 0), (
            "All recurrent weights must be non-negative. Sign is determined by synapse type ('excitatory' or 'inhibitory')."
        )

        # ========================================
        # CELL TYPE INDICES VALIDATION
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
        # Weights must be positive (sign comes from synapse type)
        assert np.all(weights_FF >= 0), (
            "All feedforward weights must be non-negative. Sign is determined by synapse type ('excitatory' or 'inhibitory')."
        )

        # Feedforward cell type indices validation
        assert cell_type_indices_FF.ndim == 1, (
            "Feedforward cell type indices must be 1D array"
        )
        assert cell_type_indices_FF.shape[0] == n_inputs, (
            f"Feedforward cell type indices length ({cell_type_indices_FF.shape[0]}) must match number of inputs ({n_inputs})"
        )

        # Feedforward cell type indices range validation
        n_cell_types_FF = int(cell_type_indices_FF.max()) + 1
        assert np.all(cell_type_indices_FF >= 0), (
            "All feedforward cell type indices must be non-negative"
        )
        assert np.all(cell_type_indices_FF < n_cell_types_FF), (
            f"All feedforward cell type indices must be less than n_cell_types_FF ({n_cell_types_FF})"
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

    def _validate_forward(self, input_spikes: torch.Tensor) -> None:
        """Validate the inputs to the forward method.

        Args:
            input_spikes: Input spike tensor to validate

        Raises:
            AssertionError: If validation fails
            ValueError: If batch size mismatch
        """
        assert input_spikes is not None, "input_spikes cannot be None"
        assert isinstance(input_spikes, torch.Tensor), (
            "input_spikes must be a torch.Tensor"
        )
        assert input_spikes.device == self.device, (
            f"input_spikes must be on device {self.device}, but got {input_spikes.device}"
        )
        assert input_spikes.ndim == 3, (
            f"input_spikes must have 3 dimensions (batch_size, n_steps, n_inputs), got shape {input_spikes.shape}"
        )
        assert input_spikes.shape[2] == self.n_inputs, (
            f"input_spikes shape[2] must match n_inputs ({self.n_inputs}), got {input_spikes.shape[2]}"
        )

        if input_spikes.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size ({input_spikes.shape[0]}) does not match "
                f"model batch size ({self.batch_size}). Call reset_state(batch_size={input_spikes.shape[0]}) first."
            )
