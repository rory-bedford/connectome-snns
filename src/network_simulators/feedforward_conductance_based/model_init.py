"""I/O utilities for feedforward-only LIF network parameters"""

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


class FeedforwardConductanceLIFNetwork_IO(nn.Module):
    """Base class for feedforward-only LIF network with I/O functionality and parameter management."""

    def __init__(
        self,
        dt: float,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors_FF: FloatArray | None = None,
        optimisable: OptimisableParams = None,
        feedforward_mask: FloatArray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = True,
    ):
        """
        Initialize the feedforward-only conductance-based LIF network with explicit parameters.

        Args:
            dt (float): The timestep in milliseconds (ms).
            weights_FF (FloatArray): Feedforward weight matrix of shape (n_inputs, n_neurons).
            cell_type_indices (IntArray): Array of shape (n_neurons,) with postsynaptic cell type indices (0, 1, 2, ...).
            cell_type_indices_FF (IntArray): Array of feedforward/input cell type indices.
            batch_size (int): Number of parallel simulations to run (batch dimension).
            cell_params (list[dict]): List of postsynaptic cell type parameter dicts. Each dict contains:
                - 'name' (str): Cell type name (e.g., 'excitatory', 'inhibitory')
                - 'cell_id' (int): Cell type ID (0, 1, 2, ...)
                - 'tau_mem' (float): Membrane time constant (ms)
                - 'theta' (float): Spike threshold voltage (mV)
                - 'U_reset' (float): Reset potential after spike (mV)
                - 'E_L' (float): Leak reversal potential (mV)
                - 'g_L' (float): Leak conductance (nS)
                - 'tau_ref' (float): Refractory period (ms)
            cell_params_FF (list[dict]): List of feedforward cell type parameter dicts (same structure as cell_params).
            synapse_params_FF (list[dict]): List of feedforward synapse parameter dicts. Each dict contains:
                - 'name' (str): Synapse type name (e.g., 'AMPA', 'NMDA', 'GABA_A')
                - 'synapse_id' (int): Unique synapse type ID (0, 1, 2, ...)
                - 'cell_id' (int): Presynaptic cell type ID this synapse belongs to
                - 'tau_rise' (float): Synaptic rise time constant (ms)
                - 'tau_decay' (float): Synaptic decay time constant (ms)
                - 'E_syn' (float): Synaptic reversal potential (mV)
                - 'g_bar' (float): Maximum synaptic conductance (nS)
            surrgrad_scale (float): Scale parameter for surrogate gradient fast sigmoid function.
            scaling_factors_FF (FloatArray | None): Matrix of shape (n_cell_types_FF, n_cell_types) for feedforward scaling (voxel^-1).
                If None, no scaling is applied (identity scaling).
            optimisable (OptimisableParams): What to optimise during training. Options:
                - "weights": Optimise connection weights (weights_FF)
                - "scaling_factors": Optimise scaling factors (scaling_factors_FF)
                - None: Don't optimise anything (all parameters are fixed) [default]
            feedforward_mask (FloatArray | None): Boolean mask for feedforward connections of shape (n_inputs, n_neurons).
                Required if optimisable="weights". If None, computed from weights_FF != 0.
            track_variables (bool): Whether to accumulate and return internal state variables (v, g_FF, I) over time.
                When False (default), only spikes are returned and memory usage is minimized.
                When True, all variables are tracked and returned as a dict for analysis/visualization.
            track_gradients (bool): Whether to store intermediate tensors WITH gradient tracking enabled.
                When False (default), intermediate states are not stored for gradient analysis.
                When True, stores v, g_FF, s at each timestep without detaching for gradient debugging.
                Use get_tracked_gradients() after backward() to extract gradient magnitudes.
            track_batch_idx (int | None): Which batch index to track when track_variables=True.
                If None (default), tracks all batch elements. If an integer, only tracks that specific
                batch index, reducing memory usage by a factor of batch_size. Useful for visualization
                where only one example is needed. Ignored when track_variables=False.
            use_tqdm (bool): Whether to display tqdm progress bar during forward pass. Default is True.
        """
        super(FeedforwardConductanceLIFNetwork_IO, self).__init__()

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
            weights_FF=weights_FF,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=cell_params,
            cell_params_FF=cell_params_FF,
            synapse_params_FF=synapse_params_FF,
            scaling_factors_FF=scaling_factors_FF,
            surrgrad_scale=surrgrad_scale,
        )

        # ====================================================
        # FIXED PARAMETERS (NON-TRAINABLE - STORED IN BUFFERS)
        # ====================================================

        self.n_cell_types = len(cell_params)
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)

        self.n_neurons = weights_FF.shape[1]
        self.n_inputs = weights_FF.shape[0]

        # Store cell and synapse parameter dictionaries
        self.cell_params = cell_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF

        # Validate and prepare weights masks
        if self.optimisable == "weights":
            assert feedforward_mask is not None, (
                "feedforward_mask is required when optimisable='weights'"
            )
            weights_mask_FF = feedforward_mask
        else:
            # If not optimizing weights, derive masks from non-zero entries if not provided
            weights_mask_FF = (
                feedforward_mask if feedforward_mask is not None else (weights_FF != 0)
            )

        # Create and register weights masks (always non-trainable)
        self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

        # Register weights with log parameterization if optimising
        if self.optimisable == "weights":
            # Extract only non-masked weights and store as flat 1D log-space arrays
            eps = 1e-8
            non_masked_weights_FF = weights_FF[weights_mask_FF]

            # Convert to log space
            log_weights_FF_flat = np.log(non_masked_weights_FF + eps)

            self._register_parameter_or_buffer(
                "log_weights_FF_flat", log_weights_FF_flat, trainable=True
            )
        else:
            # Store linear-space parameters with internal names to avoid conflict with @property
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

        # Create synapse-to-cell and cell-to-synapse mapping arrays
        self._create_synapse_to_cell_mappings(synapse_params_FF)
        self._create_cell_to_synapse_masks(synapse_params_FF)
        self._create_cell_type_masks(synapse_params_FF)

        # Create neuron-indexed arrays from cell parameters
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)

        # Register physiological parameters as neuron-indexed buffers
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Create feedforward synapse parameter arrays
        synapse_param_arrays_FF = self._create_synapse_param_arrays(synapse_params_FF)

        # Register feedforward synapse parameters
        for param_name, param_tensor in synapse_param_arrays_FF.items():
            self.register_buffer(param_name, param_tensor)

        # ===========================================================
        # OPTIMISABLE PARAMETERS (TRAINABLE - STORED AS nn.Parameter)
        # ===========================================================

        # Register scaling factors with log parameterization if optimising
        # Initialize to ones if not provided
        if scaling_factors_FF is None:
            scaling_factors_FF = np.ones(
                (self.n_cell_types_FF, self.n_cell_types), dtype=np.float32
            )

        # For feedforward-only model, all scaling factor variants mean the same thing
        is_optimizing_scaling_factors = self.optimisable in [
            "scaling_factors",
            "scaling_factors_feedforward",
            "scaling_factors_recurrent",
        ]

        if is_optimizing_scaling_factors:
            # Convert to log space for optimization
            eps = 1e-8
            log_scaling_factors_FF = np.log(scaling_factors_FF + eps)
            self._register_parameter_or_buffer(
                "log_scaling_factors_FF",
                log_scaling_factors_FF,
                trainable=True,
            )
        else:
            self._register_parameter_or_buffer(
                "_scaling_factors_FF_buffer",
                scaling_factors_FF,
                trainable=False,
            )

        # ======================================================================
        # HYPERPARAMETERS (CONFIGURATION VALUES - STORED AS INSTANCE ATTRIBUTES)
        # ======================================================================

        # Convert surrgrad_scale to tensor for JIT compatibility
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )

        # Initialize timestep-dependent parameters
        self.set_timestep(dt)

    def _precompute_weight_products(self) -> None:
        """Precompute weight-related products based on optimization mode.

        Three optimization modes:
        1. optimisable=None (inference): Precompute weights * scaling_factors * g_scale (fully optimized)
        2. optimisable="weights": Precompute scaling_factors * g_scale (weights stay dynamic)
        3. optimisable="scaling_factors": Precompute weights * g_scale (scaling_factors stay dynamic)

        Stores precomputed tensors as buffers using register_buffer() for automatic device management.
        Metadata (masks, indices) stored as regular Python lists since they're small.
        """
        # Store metadata (stays CPU, tiny)
        self.cached_weights_ff_masks = []
        self.cached_weights_ff_syn_masks = []
        self.cached_weights_ff_indices = []  # Only for training modes

        if self.optimisable is None:
            # === INFERENCE MODE: Precompute everything ===
            # Tensors have shape (n_inputs_in_type, n_neurons, 2, n_synapses_in_mask)
            for k in range(len(self.cell_type_masks_FF)):
                mask = self.cell_type_masks_FF[k]
                syn_mask = self.cell_to_synapse_mask_FF[k]
                if syn_mask.any():
                    weights_product = (
                        self.weights_FF[mask, :][:, :, None, None]
                        * self.scaling_factors_FF[k, self.cell_type_indices][
                            None, :, None, None
                        ]
                        * self.g_scale[None, None, :, syn_mask]
                    )
                    self.register_buffer(
                        f"cached_ff_{k}", weights_product, persistent=False
                    )
                    self.cached_weights_ff_masks.append(mask)
                    self.cached_weights_ff_syn_masks.append(syn_mask)

        elif self.optimisable == "weights":
            # === OPTIMIZE WEIGHTS: Precompute scaling_factors * g_scale ===
            # Tensors have shape (n_neurons, 2, n_synapses_in_mask)
            for k in range(len(self.cell_type_masks_FF)):
                mask = self.cell_type_masks_FF[k]
                syn_mask = self.cell_to_synapse_mask_FF[k]
                if syn_mask.any():
                    weights_product = (
                        self.scaling_factors_FF[k, self.cell_type_indices][
                            :, None, None
                        ]
                        * self.g_scale[None, :, syn_mask]
                    )
                    self.register_buffer(
                        f"cached_ff_{k}", weights_product, persistent=False
                    )
                    self.cached_weights_ff_masks.append(mask)
                    self.cached_weights_ff_syn_masks.append(syn_mask)
                    self.cached_weights_ff_indices.append(k)

        elif self.optimisable == "scaling_factors":
            # === OPTIMIZE SCALING_FACTORS: Precompute weights * g_scale ===
            # Tensors have shape (n_inputs_in_type, n_neurons, 2, n_synapses_in_mask)
            for k in range(len(self.cell_type_masks_FF)):
                mask = self.cell_type_masks_FF[k]
                syn_mask = self.cell_to_synapse_mask_FF[k]
                if syn_mask.any():
                    weights_product = (
                        self.weights_FF[mask, :][:, :, None, None]
                        * self.g_scale[None, None, :, syn_mask]
                    )
                    self.register_buffer(
                        f"cached_ff_{k}", weights_product, persistent=False
                    )
                    self.cached_weights_ff_masks.append(mask)
                    self.cached_weights_ff_syn_masks.append(syn_mask)
                    self.cached_weights_ff_indices.append(k)

        # Initialize internal state variables
        self.reset_state()

    def reset_state(self, batch_size: int | None = None) -> None:
        """
        Reset internal state variables to initial conditions.

        This method resets the membrane potentials (v) and feedforward conductances (g_FF)
        to their initial states. Call this before starting independent simulations or when
        changing batch size.

        State variables after reset:
        - self.v: Membrane potentials set to resting potential (v_rest)
        - self.g_FF: Feedforward synaptic conductances set to zeros

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

        # Initialize membrane potentials to resting potential
        # Shape: (batch_size, n_neurons)
        # Explicitly create on self.device to ensure consistency
        v = torch.full(
            (self.batch_size, self.n_neurons),
            fill_value=0.0,
            dtype=torch.float32,
            device=self.device,
        )
        v[:] = self.U_reset.unsqueeze(0)  # Broadcast resting potential
        self.register_buffer("v", v, persistent=False)

        # Initialize feedforward synaptic conductances to zeros
        # Shape: (batch_size, n_neurons, 2, n_synapse_types_FF)
        g_FF = torch.zeros(
            (self.batch_size, self.n_neurons, 2, self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("g_FF", g_FF, persistent=False)

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
    def weights_FF(self) -> torch.Tensor:
        """Get feedforward weights in linear space (converts from log if optimising weights)."""
        if self.optimisable == "weights":
            return torch.zeros_like(
                self.weights_mask_FF, dtype=torch.float32
            ).masked_scatter_(self.weights_mask_FF, torch.exp(self.log_weights_FF_flat))
        else:
            return self._buffers["_weights_FF_buffer"]

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        """Get feedforward scaling factors in linear space (converts from log if optimising scaling factors)."""
        if self.optimisable in [
            "scaling_factors",
            "scaling_factors_feedforward",
            "scaling_factors_recurrent",
        ]:
            return torch.exp(self.log_scaling_factors_FF)
        else:
            return self._buffers["_scaling_factors_FF_buffer"]

    @property
    def device(self):
        """Get the device the model is on"""
        if self.optimisable == "weights":
            return self.log_weights_FF_flat.device
        elif self.optimisable in [
            "scaling_factors",
            "scaling_factors_feedforward",
            "scaling_factors_recurrent",
        ]:
            return self.log_scaling_factors_FF.device
        else:
            return self._buffers["_weights_FF_buffer"].device

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

        # Precompute decay factors using feedforward arrays and register as buffers
        tau_syn = torch.stack(
            (self.tau_rise, self.tau_decay), dim=0
        )  # Shape (2, n_synapse_types_FF)
        self.register_buffer("tau_syn", tau_syn)

        alpha = torch.exp(-self.dt / self.tau_syn)  # Shape (2, n_synapse_types_FF)
        self.register_buffer("alpha", alpha)

        beta = torch.exp(-self.dt / self.tau_mem)  # Shape (n_neurons,)
        self.register_buffer("beta", beta)

        # Stack g_bar with its negative
        g_scale = torch.stack(
            [-self.g_bar, self.g_bar], dim=0
        )  # Shape (2, n_synapse_types_FF)

        # Normalize by peak
        r = self.tau_rise / self.tau_decay
        norm_peak = (r ** (r / (1 - r))) - (r ** (1 / (1 - r)))
        g_scale = g_scale / norm_peak
        self.register_buffer("g_scale", g_scale)

        # Precompute clamping bounds for conductances
        g_mins = torch.stack([g_scale[0, :], torch.zeros_like(g_scale[1, :])], dim=0)
        g_maxs = torch.stack([torch.zeros_like(g_scale[0, :]), g_scale[1, :]], dim=0)
        self.register_buffer("g_mins", g_mins)
        self.register_buffer("g_maxs", g_maxs)

        # Recompute weight products after timestep change
        self._precompute_weight_products()

    def compile_step(self) -> None:
        """JIT compile the appropriate _step method based on optimization mode.

        This should be called after moving the model to the target device and before
        running the forward pass for maximum performance.

        Example:
            >>> model = FeedforwardConductanceLIFNetwork(...)
            >>> model.to('cuda')
            >>> model.compile_step()
            >>> output = model(input_spikes)
        """
        if self.optimisable is None:
            self._step_inference = torch.jit.script(self._step_inference)
        elif self.optimisable == "weights":
            self._step_optimize_weights = torch.jit.script(self._step_optimize_weights)
        elif self.optimisable == "scaling_factors":
            self._step_optimize_scaling_factors = torch.jit.script(
                self._step_optimize_scaling_factors
            )

    def _create_synapse_to_cell_mappings(self, synapse_params_FF: list[dict]) -> None:
        """
        Create and register lookup array mapping synapse IDs to their parent cell type IDs.

        This creates a mapping tensor:
        - synapse_to_cell_id_FF: Maps feedforward synapse_id -> cell_id (shape: [n_synapse_types_FF])

        Each synapse type belongs to a specific presynaptic cell type, and this mapping
        enables fast lookup of which cell type produces a given synapse type.

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

    def _create_cell_to_synapse_masks(self, synapse_params_FF: list[dict]) -> None:
        """
        Create boolean mask for mapping cell types to their associated synapse types.

        Creates a boolean mask tensor that enables efficient lookup of which synapse types
        are produced by each feedforward cell type:

        - cell_to_synapse_mask_FF: Shape [max_ff_cell_type + 1, n_synapse_types_FF]
          For feedforward connections, mask[cell_type, :] indicates which
          feedforward synapse types this cell type produces

        Args:
            synapse_params_FF: List of feedforward synapse parameter dicts
        """
        # Get unique cell type IDs from feedforward synapses
        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))
        max_ff_cell_type = max(ff_cell_types) if ff_cell_types else -1

        # Create boolean mask for feedforward cell types -> synapse types
        cell_to_synapse_mask_FF = torch.zeros(
            max_ff_cell_type + 1, self.n_synapse_types_FF, dtype=torch.bool
        )
        for cell_type in ff_cell_types:
            cell_to_synapse_mask_FF[cell_type, :] = (
                self.synapse_to_cell_id_FF == cell_type
            )
        self.register_buffer("cell_to_synapse_mask_FF", cell_to_synapse_mask_FF)

    def _create_cell_type_masks(self, synapse_params_FF: list[dict]) -> None:
        """
        Create boolean masks for efficient input indexing by cell type.

        Precomputes boolean masks that indicate which inputs belong to each cell type,
        avoiding repeated tensor comparisons during simulation:

        - cell_type_masks_FF: List of boolean tensors, where cell_type_masks_FF[i] has shape [n_inputs]
          and indicates which inputs belong to feedforward cell type i

        These are stored as Python lists (not registered buffers) since the list length
        varies and PyTorch buffers require fixed tensor shapes.

        Args:
            synapse_params_FF: List of feedforward synapse parameter dicts (used to extract cell types)
        """
        # Get unique feedforward cell types
        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))

        # Precompute boolean masks for each feedforward cell type
        cell_type_masks_FF = []
        for cell_type in ff_cell_types:
            cell_type_masks_FF.append(self.cell_type_indices_FF == cell_type)
        self.cell_type_masks_FF = cell_type_masks_FF

    def _validate(
        self,
        dt: float,
        weights_FF: FloatArray,
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
        scaling_factors_FF: FloatArray | None,
        surrgrad_scale: float,
    ) -> tuple[int, int, int]:
        """
        Validate all input parameters and return extracted dimensions.

        Args:
            dt: Simulation timestep in milliseconds
            weights_FF: Feedforward weight matrix
            cell_type_indices: Neuron-to-cell-type mapping
            cell_type_indices_FF: Input-to-cell-type mapping
            cell_params: List of postsynaptic cell parameter dicts
            cell_params_FF: List of feedforward cell parameter dicts
            synapse_params_FF: List of feedforward synapse parameter dicts
            scaling_factors_FF: Input-cell-type-to-cell-type scaling matrix (optional)
            surrgrad_scale: Surrogate gradient scale

        Returns:
            tuple: (n_cell_types, n_cell_types_FF, n_synapse_types_FF)
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
        n_neurons = weights_FF.shape[1] if weights_FF.ndim == 2 else 0
        n_inputs = weights_FF.shape[0] if weights_FF.ndim == 2 else 0

        # ========================================
        # POSTSYNAPTIC CELL PARAMS VALIDATION
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

        return n_cell_types, n_cell_types_FF, n_synapse_types_FF

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
