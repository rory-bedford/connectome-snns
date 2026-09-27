"""I/O utilities for LIF network parameters (projection-based API)."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from connectome_snns.network_simulators.projections import (
    Projection,
    ScalingFactorProjection,
)

# Type aliases
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]
PairKey = Tuple[str, str]  # (source_cell_type_name, target_cell_type_name)


class ConductanceLIFNetwork_IO(nn.Module):
    """Base class for LIF network with I/O functionality and parameter management.

    Connectivity is supplied as ``rec_projections`` (recurrent) and
    ``ff_projections`` (feedforward) — dicts mapping
    ``(source_cell_type_name, target_cell_type_name)`` to a ``Projection``
    object. Each projection owns its own parameters and chooses its
    parametrisation independently (Frozen/ScalingFactor/FullRank/LowRank).

    Constraint: all projections sharing the same source cell type must
    share the same ``caching_mode`` (e.g. all "weights" or all
    "scaling_factors"). Mixed caching modes within a source cell type
    are not supported.
    """

    def __init__(
        self,
        dt: float,
        rec_projections: dict[PairKey, Projection],
        ff_projections: dict[PairKey, Projection],
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: Literal["deterministic", "sample"] = "deterministic",
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
    ):
        """Initialize the conductance-based LIF network.

        Args:
            dt: Simulation timestep (ms).
            rec_projections: ``{(src_ct_name, tgt_ct_name): Projection}`` for
                recurrent connectivity. Source and target cell type names
                must appear in ``cell_params``.
            ff_projections: ``{(src_ct_name, tgt_ct_name): Projection}`` for
                feedforward connectivity. Source names must appear in
                ``cell_params_FF``; target names in ``cell_params``.
            cell_type_indices: Per-neuron cell type ID, shape (n_neurons,).
            cell_type_indices_FF: Per-input cell type ID, shape (n_inputs,).
            cell_params: List of recurrent cell-type configs (each has
                'name', 'cell_id', 'tau_mem', 'theta', 'U_reset', 'E_L',
                'g_L', 'tau_ref').
            cell_params_FF: Same shape for FF input cell types.
            synapse_params: List of recurrent synapse configs ('synapse_id',
                'cell_id', 'name', 'tau_rise', 'tau_decay', 'E_syn',
                'g_bar', optional 'g_clip').
            synapse_params_FF: Same shape for FF synapses.
            surrgrad_scale: Surrogate-gradient scale for spike function.
            batch_size: Number of parallel simulations.
            spike_mode: ``"deterministic"`` (Heaviside + surrogate grad) or
                ``"sample"`` (Bernoulli forward, surrogate grad backward).
            track_variables: Store v/g/I per timestep for analysis.
            track_gradients: Retain grads on per-step tensors for analysis.
            track_batch_idx: If set, only track that batch index.
        """
        super().__init__()

        # Store basics
        self.batch_size = batch_size
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
        self.spike_mode = spike_mode

        # Store cell/synapse param dicts for downstream use
        self.cell_params = cell_params
        self.synapse_params = synapse_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF

        self.n_cell_types = len(cell_params)
        self.n_synapse_types = len(synapse_params)
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)

        self.n_neurons = int(cell_type_indices.shape[0])
        self.n_inputs = int(cell_type_indices_FF.shape[0])

        # Cell type name <-> id mappings (canonical)
        self._rec_name_to_id: dict[str, int] = {
            p["name"]: p["cell_id"] for p in cell_params
        }
        self._rec_id_to_name: dict[int, str] = {
            p["cell_id"]: p["name"] for p in cell_params
        }
        self._ff_name_to_id: dict[str, int] = {
            p["name"]: p["cell_id"] for p in cell_params_FF
        }
        self._ff_id_to_name: dict[int, str] = {
            p["cell_id"]: p["name"] for p in cell_params_FF
        }

        # =================================
        # PARAMETER VALIDATION
        # =================================
        self._validate(
            dt=dt,
            rec_projections=rec_projections,
            ff_projections=ff_projections,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=cell_params,
            cell_params_FF=cell_params_FF,
            synapse_params=synapse_params,
            synapse_params_FF=synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
        )

        # =================================
        # SYNAPSE INFRASTRUCTURE (unchanged)
        # =================================
        self._build_unified_synapse_registry(
            cell_params, synapse_params, cell_params_FF, synapse_params_FF
        )

        # =================================
        # REGISTER PROJECTIONS
        # =================================
        # ModuleDict requires string keys; encode pair as "src__tgt".
        self.rec_projections = nn.ModuleDict(
            {self._pair_to_key(p): proj for p, proj in rec_projections.items()}
        )
        self.ff_projections = nn.ModuleDict(
            {self._pair_to_key(p): proj for p, proj in ff_projections.items()}
        )
        self._rec_pairs: list[PairKey] = list(rec_projections.keys())
        self._ff_pairs: list[PairKey] = list(ff_projections.keys())

        # =================================
        # DERIVE PER-SOURCE-CELL-TYPE MODES
        # =================================
        # Constraint: all (src, *) projections share caching_mode.
        self._rec_mode_per_ct: list[str | None] = self._derive_modes_per_source_ct(
            rec_projections, self.n_cell_types, self._rec_id_to_name, "recurrent"
        )
        self._ff_mode_per_ct: list[str | None] = self._derive_modes_per_source_ct(
            ff_projections, self.n_cell_types_FF, self._ff_id_to_name, "feedforward"
        )

        # Convenience flags (kept for parity with old API for any downstream readers)
        self._has_weights_mode = (
            "weights" in self._rec_mode_per_ct + self._ff_mode_per_ct
        )
        self._has_scaling_mode = (
            "scaling_factors" in self._rec_mode_per_ct + self._ff_mode_per_ct
        )

        # =================================
        # BUILD WEIGHTS MASKS FROM PROJECTIONS
        # =================================
        # Used by the simulator to slice the assembled weight matrices.
        weights_mask = self._assemble_mask_from_projections(
            rec_projections,
            cell_type_indices,
            cell_type_indices,
            self._rec_id_to_name,
            self._rec_id_to_name,
            (self.n_neurons, self.n_neurons),
        )
        weights_mask_FF = self._assemble_mask_from_projections(
            ff_projections,
            cell_type_indices_FF,
            cell_type_indices,
            self._ff_id_to_name,
            self._rec_id_to_name,
            (self.n_inputs, self.n_neurons),
        )
        self.register_buffer("weights_mask", torch.from_numpy(weights_mask))
        self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

        # Cell type indices as buffers (so they move with .to(device))
        self.register_buffer(
            "cell_type_indices", torch.from_numpy(np.asarray(cell_type_indices)).long()
        )
        self.register_buffer(
            "cell_type_indices_FF",
            torch.from_numpy(np.asarray(cell_type_indices_FF)).long(),
        )

        # =================================
        # SYNAPSE & CELL TYPE MASKS (unchanged helpers)
        # =================================
        self._create_synapse_to_cell_mappings_unified()
        self._create_cell_to_synapse_masks_unified()
        self._create_cell_type_masks(synapse_params, synapse_params_FF)

        # Neuron-indexed physiology params
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Unified synapse parameter arrays
        synapse_param_arrays = self._create_unified_synapse_param_arrays()
        for param_name, param_tensor in synapse_param_arrays.items():
            self.register_buffer(param_name, param_tensor)

        # Surrogate gradient scale
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )

        # Initialise timestep-dependent params + per-cell-type cached buffers
        self.set_timestep(dt)

    # ======================================================================
    # Helpers — pair keys, mode derivation, mask assembly
    # ======================================================================

    @staticmethod
    def _pair_to_key(pair: PairKey) -> str:
        return f"{pair[0]}__{pair[1]}"

    @staticmethod
    def _key_to_pair(key: str) -> PairKey:
        src, tgt = key.split("__", 1)
        return (src, tgt)

    @staticmethod
    def _derive_modes_per_source_ct(
        projections: dict[PairKey, Projection],
        n_source_cell_types: int,
        id_to_name: dict[int, str],
        which: str,  # "recurrent" or "feedforward" (for error messages)
    ) -> list[str | None]:
        """Per source cell type, return the unique caching mode of all (src, *) projections.

        Raises if a source cell type's projections have mixed caching modes.
        Cell types with no outgoing projections return None.
        """
        modes: list[str | None] = [None] * n_source_cell_types
        for (src_name, tgt_name), proj in projections.items():
            src_id = next((k for k, v in id_to_name.items() if v == src_name), None)
            if src_id is None:
                raise ValueError(
                    f"{which} projection source '{src_name}' not in cell_params"
                )
            mode = proj.caching_mode
            if modes[src_id] is None:
                modes[src_id] = mode
            elif modes[src_id] != mode:
                raise ValueError(
                    f"{which} source cell type '{src_name}' has mixed projection "
                    f"caching modes ({modes[src_id]!r} and {mode!r}). All "
                    f"projections sharing a source cell type must use the same mode."
                )
        return modes

    @staticmethod
    def _assemble_mask_from_projections(
        projections: dict[PairKey, Projection],
        src_cell_type_indices: IntArray,
        tgt_cell_type_indices: IntArray,
        src_id_to_name: dict[int, str],
        tgt_id_to_name: dict[int, str],
        shape: tuple[int, int],
    ) -> NDArray[np.bool_]:
        """Build a (n_src, n_tgt) bool mask from per-pair projection connectomes.

        The mask is True wherever at least one entry in the corresponding
        projection's weight matrix is non-zero. ScalingFactorProjection
        masks come from the connectome buffer; FullRank/LowRank from the
        explicit mask buffer; FrozenProjection from the weights buffer.
        """
        mask = np.zeros(shape, dtype=bool)
        src_name_to_id = {v: k for k, v in src_id_to_name.items()}
        tgt_name_to_id = {v: k for k, v in tgt_id_to_name.items()}
        for (src_name, tgt_name), proj in projections.items():
            src_id = src_name_to_id[src_name]
            tgt_id = tgt_name_to_id[tgt_name]
            src_idx = np.flatnonzero(src_cell_type_indices == src_id)
            tgt_idx = np.flatnonzero(tgt_cell_type_indices == tgt_id)
            if src_idx.size == 0 or tgt_idx.size == 0:
                continue
            # Source the per-pair mask depending on projection type
            block_mask = _projection_mask_np(proj)
            if block_mask.shape != (src_idx.size, tgt_idx.size):
                raise ValueError(
                    f"Projection ({src_name!r}, {tgt_name!r}) has shape "
                    f"{block_mask.shape}, expected {(src_idx.size, tgt_idx.size)}"
                )
            mask[np.ix_(src_idx, tgt_idx)] = block_mask
        return mask

    # ======================================================================
    # Mode resolution (used by simulator)
    # ======================================================================

    def _resolve_modes(self) -> tuple[list[str | None], list[str | None]]:
        """Return per-cell-type modes for recurrent and FF pathways.

        Now derived from the projection registry rather than ``optimisable``.
        Each cell-type entry is None, "weights", or "scaling_factors".
        """
        # Restrict to cell types that actually appear in the cell-type masks
        # (matches the existing behaviour of the simulator's pathway loop).
        n_rec = len(self.cell_type_masks)
        n_ff = len(self.cell_type_masks_FF)
        rec_modes = list(self._rec_mode_per_ct[:n_rec])
        ff_modes = list(self._ff_mode_per_ct[:n_ff])
        return rec_modes, ff_modes

    # ======================================================================
    # Weight & SF assembly properties
    # ======================================================================

    def _assemble_block_matrix(
        self,
        projections_dict: nn.ModuleDict,
        pairs: list[PairKey],
        src_cell_type_indices: torch.Tensor,
        tgt_cell_type_indices: torch.Tensor,
        src_id_to_name: dict[int, str],
        tgt_id_to_name: dict[int, str],
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Assemble a (n_src, n_tgt) tensor from per-pair Projection.forward()."""
        out = torch.zeros(shape, dtype=dtype, device=device)
        src_name_to_id = {v: k for k, v in src_id_to_name.items()}
        tgt_name_to_id = {v: k for k, v in tgt_id_to_name.items()}
        for pair in pairs:
            proj = projections_dict[self._pair_to_key(pair)]
            src_id = src_name_to_id[pair[0]]
            tgt_id = tgt_name_to_id[pair[1]]
            src_idx = (src_cell_type_indices == src_id).nonzero(as_tuple=True)[0]
            tgt_idx = (tgt_cell_type_indices == tgt_id).nonzero(as_tuple=True)[0]
            if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                continue
            block = proj()  # (n_src_pair, n_tgt_pair)
            # Place block at rows src_idx, columns tgt_idx
            out.index_put_(
                (src_idx[:, None], tgt_idx[None, :]), block, accumulate=False
            )
        return out

    @property
    def weights(self) -> torch.Tensor:
        """Recurrent weight matrix (n_neurons, n_neurons), assembled from projections."""
        return self._assemble_block_matrix(
            self.rec_projections,
            self._rec_pairs,
            self.cell_type_indices,
            self.cell_type_indices,
            self._rec_id_to_name,
            self._rec_id_to_name,
            (self.n_neurons, self.n_neurons),
            torch.float32,
            self.device,
        )

    @property
    def weights_FF(self) -> torch.Tensor:
        """Feedforward weight matrix (n_inputs, n_neurons), assembled from projections."""
        return self._assemble_block_matrix(
            self.ff_projections,
            self._ff_pairs,
            self.cell_type_indices_FF,
            self.cell_type_indices,
            self._ff_id_to_name,
            self._rec_id_to_name,
            (self.n_inputs, self.n_neurons),
            torch.float32,
            self.device,
        )

    @property
    def scaling_factors(self) -> torch.Tensor:
        """Recurrent scaling factor matrix (n_cell_types, n_cell_types).

        Entry (i, j) is the scalar SF for source cell type i → target type j
        (1.0 if the projection is not a ScalingFactorProjection).
        """
        sf = torch.ones(
            (self.n_cell_types, self.n_cell_types),
            dtype=torch.float32,
            device=self.device,
        )
        # Use the first available parameter to seed graph membership for grad flow
        for pair, proj in zip(self._rec_pairs, self.rec_projections.values()):
            if isinstance(proj, ScalingFactorProjection):
                src_id = self._rec_name_to_id[pair[0]]
                tgt_id = self._rec_name_to_id[pair[1]]
                sf[src_id, tgt_id] = proj.kernel_factor()
        return sf

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        """Feedforward scaling factor matrix (n_cell_types_FF, n_cell_types)."""
        sf = torch.ones(
            (self.n_cell_types_FF, self.n_cell_types),
            dtype=torch.float32,
            device=self.device,
        )
        for pair, proj in zip(self._ff_pairs, self.ff_projections.values()):
            if isinstance(proj, ScalingFactorProjection):
                src_id = self._ff_name_to_id[pair[0]]
                tgt_id = self._rec_name_to_id[pair[1]]
                sf[src_id, tgt_id] = proj.kernel_factor()
        return sf

    @property
    def device(self):
        # Use the first projection's device (or a buffer's if no projections trained).
        for proj in self.rec_projections.values():
            for p in proj.parameters():
                return p.device
            for b in proj.buffers():
                return b.device
        for proj in self.ff_projections.values():
            for p in proj.parameters():
                return p.device
            for b in proj.buffers():
                return b.device
        return self.weights_mask.device  # fallback: registered at init

    # ======================================================================
    # _precompute_weight_products (per-source-cell-type, unchanged shape)
    # ======================================================================

    def _precompute_weight_products(self) -> None:
        """Precompute conductance update tensors, factored for efficiency.

        See class docstring + the projection-API REFACTOR_PLAN.md for the
        per-pair caching constraint. Each source cell type has a uniform
        caching mode; we still cache conn_w/kernel per source cell type.
        """
        self.rec_masks = []
        self.rec_syn_masks = []
        self.rec_cell_type_indices = []
        self.ff_masks = []
        self.ff_syn_masks = []
        self.ff_cell_type_indices = []

        rec_modes, ff_modes = self._resolve_modes()

        # Snapshot assembled tensors once for init-time buffer construction.
        weights_now = self.weights.detach()
        weights_FF_now = self.weights_FF.detach()
        sf_now = self.scaling_factors.detach()
        sf_FF_now = self.scaling_factors_FF.detach()

        def _register_cell_type(
            prefix,
            k,
            mask,
            syn_mask,
            weights,
            sf,
            mode,
            mask_list,
            syn_mask_list,
            idx_list,
        ):
            g_kernel = self.g_scale[None, :, syn_mask]  # (1, n_rise, n_syn)
            if mode is None:
                conn_w = (
                    weights[mask, :] * sf[k, self.cell_type_indices][None, :]
                ).detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
            elif mode == "weights":
                conn_w = None
                kernel = (
                    sf[k, self.cell_type_indices][:, None, None] * g_kernel
                ).detach()
            elif mode == "scaling_factors":
                conn_w = weights[mask, :].detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
            elif mode == "fixed":
                # FrozenProjection: weights are constant, no scaling factor.
                conn_w = weights[mask, :].detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
            else:
                raise ValueError(f"Unknown mode {mode!r}")

            if conn_w is not None:
                self.register_buffer(
                    f"connection_weights_{prefix}_{k}", conn_w, persistent=False
                )
            self.register_buffer(
                f"synapse_kernel_{prefix}_{k}", kernel, persistent=False
            )
            mask_list.append(mask)
            syn_mask_list.append(syn_mask)
            idx_list.append(k)

        for ki, k in enumerate(range(len(self.cell_type_masks))):
            mask = self.cell_type_masks[k]
            syn_mask = self.cell_to_synapse_mask[k]
            if syn_mask.any():
                _register_cell_type(
                    "rec",
                    k,
                    mask,
                    syn_mask,
                    weights_now,
                    sf_now,
                    rec_modes[ki],
                    self.rec_masks,
                    self.rec_syn_masks,
                    self.rec_cell_type_indices,
                )
        for ki, k in enumerate(range(len(self.cell_type_masks_FF))):
            mask = self.cell_type_masks_FF[k]
            syn_mask = self.cell_to_synapse_mask_FF[k]
            if syn_mask.any():
                _register_cell_type(
                    "ff",
                    k,
                    mask,
                    syn_mask,
                    weights_FF_now,
                    sf_FF_now,
                    ff_modes[ki],
                    self.ff_masks,
                    self.ff_syn_masks,
                    self.ff_cell_type_indices,
                )

        # Initialise simulation state
        self.reset_state()

    # ======================================================================
    # State / checkpointing
    # ======================================================================

    def reset_state(self, batch_size: int | None = None) -> None:
        """Reset membrane potentials and conductances to initial conditions."""
        if batch_size is not None:
            self.batch_size = batch_size
        v = torch.full(
            (self.batch_size, self.n_neurons),
            fill_value=0.0,
            dtype=torch.float32,
            device=self.device,
        )
        v[:] = self.U_reset.unsqueeze(0)
        self.register_buffer("v", v, persistent=False)
        g = torch.zeros(
            (self.batch_size, self.n_neurons, 2, self.n_unified_synapse_types),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("g", g, persistent=False)

    def get_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"v": self.v.cpu().numpy(), "g": self.g.cpu().numpy()}

    def load_checkpoint_state(self, state: dict[str, np.ndarray]) -> None:
        self.v = torch.from_numpy(state["v"]).to(self.device)
        self.g = torch.from_numpy(state["g"]).to(self.device)

    def _register_parameter_or_buffer(
        self, name: str, value: torch.Tensor | np.ndarray, trainable: bool = False
    ) -> None:
        """Register as nn.Parameter (trainable=True) or buffer."""
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            if value.dtype.is_floating_point:
                value = value.float()
            else:
                value = value.int()
        if trainable:
            self.register_parameter(name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    # ======================================================================
    # Set timestep + decay precomputation
    # ======================================================================

    def set_timestep(self, dt: float) -> None:
        """Set simulation timestep. Recomputes all dt-dependent buffers."""
        assert isinstance(dt, float), "dt must be a float"
        assert dt > 0, "Timestep dt must be positive"

        self.register_buffer("dt", torch.tensor(dt, device=self.device))
        tau_syn = torch.stack((self.tau_rise, self.tau_decay), dim=0)
        self.register_buffer("tau_syn", tau_syn)
        alpha = torch.exp(-self.dt / self.tau_syn)
        self.register_buffer("alpha", alpha)
        beta = torch.exp(-self.dt / self.tau_mem)
        self.register_buffer("beta", beta)
        g_scale = torch.stack([-self.g_bar, self.g_bar], dim=0)
        r = self.tau_rise / self.tau_decay
        norm_peak = (r ** (r / (1 - r))) - (r ** (1 / (1 - r)))
        g_scale = g_scale / norm_peak
        self.register_buffer("g_scale", g_scale)

        self._precompute_weight_products()

    # ======================================================================
    # Synapse infrastructure (unchanged from previous version)
    # ======================================================================

    def _create_neuron_param_arrays(
        self,
        cell_params: list[dict],
        cell_type_indices: IntArray,
    ) -> dict[str, torch.Tensor]:
        required_param_names = [
            "tau_mem",
            "theta",
            "U_reset",
            "E_L",
            "g_L",
            "tau_ref",
        ]
        n_cell_types = len(cell_params)
        neuron_params = {}
        for param_name in required_param_names:
            param_lookup = np.zeros(n_cell_types, dtype=np.float32)
            for cell in cell_params:
                param_lookup[cell["cell_id"]] = cell[param_name]
            param_array = param_lookup[cell_type_indices]
            neuron_params[param_name] = torch.from_numpy(param_array)
        neuron_params["C_m"] = neuron_params["tau_mem"] * neuron_params["g_L"]
        return neuron_params

    def _build_unified_synapse_registry(
        self,
        cell_params: list[dict],
        synapse_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
    ) -> None:
        rec_cell_id_to_name = {p["cell_id"]: p["name"] for p in cell_params}
        ff_cell_id_to_name = {p["cell_id"]: p["name"] for p in cell_params_FF}

        synapse_signature_to_unified_id: dict[tuple[str, str], int] = {}
        unified_synapse_params: list[dict] = []
        unified_id = 0

        rec_synapse_to_unified: dict[int, int] = {}
        for synapse in synapse_params:
            cell_name = rec_cell_id_to_name[synapse["cell_id"]]
            signature = (cell_name, synapse["name"])
            if signature not in synapse_signature_to_unified_id:
                synapse_signature_to_unified_id[signature] = unified_id
                unified_params = synapse.copy()
                unified_params["unified_synapse_id"] = unified_id
                unified_params["signature"] = signature
                unified_synapse_params.append(unified_params)
                unified_id += 1
            rec_synapse_to_unified[synapse["synapse_id"]] = (
                synapse_signature_to_unified_id[signature]
            )

        ff_synapse_to_unified: dict[int, int] = {}
        for synapse in synapse_params_FF:
            cell_name = ff_cell_id_to_name[synapse["cell_id"]]
            signature = (cell_name, synapse["name"])
            if signature not in synapse_signature_to_unified_id:
                synapse_signature_to_unified_id[signature] = unified_id
                unified_params = synapse.copy()
                unified_params["unified_synapse_id"] = unified_id
                unified_params["signature"] = signature
                unified_synapse_params.append(unified_params)
                unified_id += 1
            ff_synapse_to_unified[synapse["synapse_id"]] = (
                synapse_signature_to_unified_id[signature]
            )

        self.unified_synapse_params = unified_synapse_params
        self.n_unified_synapse_types = len(unified_synapse_params)
        self.rec_synapse_to_unified = rec_synapse_to_unified
        self.ff_synapse_to_unified = ff_synapse_to_unified
        self.synapse_signature_to_unified_id = synapse_signature_to_unified_id

        n_shared = (
            self.n_synapse_types + self.n_synapse_types_FF
        ) - self.n_unified_synapse_types
        if n_shared > 0:
            print(
                f"  Unified synapse registry: {self.n_unified_synapse_types} types "
                f"({n_shared} shared between recurrent/feedforward)"
            )

    def _create_synapse_to_cell_mappings_unified(self) -> None:
        rec_cell_name_to_id = {p["name"]: p["cell_id"] for p in self.cell_params}
        synapse_to_cell_mapping = np.zeros(self.n_unified_synapse_types, dtype=np.int64)
        for unified_params in self.unified_synapse_params:
            unified_id = unified_params["unified_synapse_id"]
            cell_name = unified_params["signature"][0]
            if cell_name in rec_cell_name_to_id:
                synapse_to_cell_mapping[unified_id] = rec_cell_name_to_id[cell_name]
            else:
                synapse_to_cell_mapping[unified_id] = -1
        self.register_buffer(
            "synapse_to_cell_id", torch.from_numpy(synapse_to_cell_mapping)
        )

    def _create_cell_to_synapse_masks_unified(self) -> None:
        recurrent_cell_types = sorted(
            set(synapse["cell_id"] for synapse in self.synapse_params)
        )
        max_recurrent_cell_type = (
            max(recurrent_cell_types) if recurrent_cell_types else -1
        )
        cell_to_synapse_mask = torch.zeros(
            max_recurrent_cell_type + 1, self.n_unified_synapse_types, dtype=torch.bool
        )
        for synapse in self.synapse_params:
            cell_type = synapse["cell_id"]
            unified_id = self.rec_synapse_to_unified[synapse["synapse_id"]]
            cell_to_synapse_mask[cell_type, unified_id] = True
        self.register_buffer("cell_to_synapse_mask", cell_to_synapse_mask)

        ff_cell_types = sorted(
            set(synapse["cell_id"] for synapse in self.synapse_params_FF)
        )
        max_ff_cell_type = max(ff_cell_types) if ff_cell_types else -1
        cell_to_synapse_mask_FF = torch.zeros(
            max_ff_cell_type + 1, self.n_unified_synapse_types, dtype=torch.bool
        )
        for synapse in self.synapse_params_FF:
            cell_type = synapse["cell_id"]
            unified_id = self.ff_synapse_to_unified[synapse["synapse_id"]]
            cell_to_synapse_mask_FF[cell_type, unified_id] = True
        self.register_buffer("cell_to_synapse_mask_FF", cell_to_synapse_mask_FF)

    def _create_cell_type_masks(
        self, synapse_params: list[dict], synapse_params_FF: list[dict]
    ) -> None:
        recurrent_cell_types = sorted(
            set(synapse["cell_id"] for synapse in synapse_params)
        )
        cell_type_masks = []
        for cell_type in recurrent_cell_types:
            cell_type_masks.append(self.cell_type_indices == cell_type)
        self.cell_type_masks = cell_type_masks

        ff_cell_types = sorted(set(synapse["cell_id"] for synapse in synapse_params_FF))
        cell_type_masks_FF = []
        for cell_type in ff_cell_types:
            cell_type_masks_FF.append(self.cell_type_indices_FF == cell_type)
        self.cell_type_masks_FF = cell_type_masks_FF

    def _create_unified_synapse_param_arrays(self) -> dict[str, torch.Tensor]:
        required_param_names = ["tau_rise", "tau_decay", "E_syn", "g_bar"]
        synapse_param_arrays: dict[str, torch.Tensor] = {}
        for param_name in required_param_names:
            param_array = np.zeros(self.n_unified_synapse_types, dtype=np.float32)
            for unified_params in self.unified_synapse_params:
                param_array[unified_params["unified_synapse_id"]] = unified_params[
                    param_name
                ]
            synapse_param_arrays[param_name] = torch.from_numpy(param_array)
        g_clip_array = np.full(
            self.n_unified_synapse_types, float("inf"), dtype=np.float32
        )
        for unified_params in self.unified_synapse_params:
            if "g_clip" in unified_params:
                g_clip_array[unified_params["unified_synapse_id"]] = unified_params[
                    "g_clip"
                ]
        synapse_param_arrays["g_clip"] = torch.from_numpy(g_clip_array)
        return synapse_param_arrays

    # ======================================================================
    # Validation
    # ======================================================================

    def _validate(
        self,
        dt: float,
        rec_projections: dict[PairKey, Projection],
        ff_projections: dict[PairKey, Projection],
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
    ) -> None:
        # dt
        assert isinstance(dt, (int, float)), "dt must be numeric"
        assert dt > 0, f"dt must be positive, got {dt}"
        assert dt <= 100.0, f"dt unusually large ({dt} ms), check units"

        # cell_params
        cell_ids = [p["cell_id"] for p in cell_params]
        n_cell_types = max(cell_ids) + 1 if cell_ids else 0
        assert sorted(cell_ids) == list(range(n_cell_types)), (
            f"cell_ids in cell_params must be 0-indexed contiguous, got {sorted(cell_ids)}"
        )

        # cell_type_indices
        assert cell_type_indices.ndim == 1, "cell_type_indices must be 1D"
        assert cell_type_indices.shape[0] > 0, "n_neurons must be positive"
        assert np.all(cell_type_indices >= 0) and np.all(
            cell_type_indices < n_cell_types
        ), "cell_type_indices out of range"

        # synapse_params
        assert len(synapse_params) > 0, "synapse_params must not be empty"
        synapse_ids = [p["synapse_id"] for p in synapse_params]
        n_syn = max(synapse_ids) + 1
        assert sorted(synapse_ids) == list(range(n_syn)), (
            "synapse_ids must be 0-indexed contiguous"
        )

        # cell_params_FF
        cell_ids_FF = [p["cell_id"] for p in cell_params_FF]
        n_cell_types_FF = max(cell_ids_FF) + 1 if cell_ids_FF else 0
        assert sorted(cell_ids_FF) == list(range(n_cell_types_FF)), (
            "ff cell_ids must be 0-indexed contiguous"
        )

        # cell_type_indices_FF
        assert cell_type_indices_FF.ndim == 1, "cell_type_indices_FF must be 1D"
        assert np.all(cell_type_indices_FF >= 0) and np.all(
            cell_type_indices_FF < n_cell_types_FF
        ), "cell_type_indices_FF out of range"

        # synapse_params_FF
        assert len(synapse_params_FF) > 0, "synapse_params_FF must not be empty"
        syn_ids_FF = [p["synapse_id"] for p in synapse_params_FF]
        n_syn_FF = max(syn_ids_FF) + 1
        assert sorted(syn_ids_FF) == list(range(n_syn_FF)), (
            "ff synapse_ids must be 0-indexed contiguous"
        )

        # Projections shape sanity
        rec_name_to_id = {p["name"]: p["cell_id"] for p in cell_params}
        ff_name_to_id = {p["name"]: p["cell_id"] for p in cell_params_FF}

        n_per_rec_ct = np.bincount(cell_type_indices, minlength=n_cell_types)
        n_per_ff_ct = np.bincount(cell_type_indices_FF, minlength=n_cell_types_FF)

        for (src, tgt), proj in rec_projections.items():
            assert src in rec_name_to_id, (
                f"rec projection source '{src}' not in cell_params"
            )
            assert tgt in rec_name_to_id, (
                f"rec projection target '{tgt}' not in cell_params"
            )
            expected = (
                int(n_per_rec_ct[rec_name_to_id[src]]),
                int(n_per_rec_ct[rec_name_to_id[tgt]]),
            )
            assert (proj.n_source, proj.n_target) == expected, (
                f"rec projection ({src!r}, {tgt!r}) shape "
                f"({proj.n_source}, {proj.n_target}) != expected {expected}"
            )

        for (src, tgt), proj in ff_projections.items():
            assert src in ff_name_to_id, (
                f"ff projection source '{src}' not in cell_params_FF"
            )
            assert tgt in rec_name_to_id, (
                f"ff projection target '{tgt}' not in cell_params"
            )
            expected = (
                int(n_per_ff_ct[ff_name_to_id[src]]),
                int(n_per_rec_ct[rec_name_to_id[tgt]]),
            )
            assert (proj.n_source, proj.n_target) == expected, (
                f"ff projection ({src!r}, {tgt!r}) shape "
                f"({proj.n_source}, {proj.n_target}) != expected {expected}"
            )

        assert isinstance(surrgrad_scale, (int, float)), (
            "surrgrad_scale must be numeric"
        )
        assert surrgrad_scale > 0, "surrgrad_scale must be positive"

    def _validate_forward(self, input_spikes: torch.Tensor) -> None:
        """Validate inputs to forward()."""
        assert input_spikes is not None, "input_spikes cannot be None"
        assert isinstance(input_spikes, torch.Tensor), "input_spikes must be a Tensor"
        assert input_spikes.device == self.device, (
            f"input_spikes on {input_spikes.device}, model on {self.device}"
        )
        assert input_spikes.ndim == 3, (
            f"input_spikes must be 3D (batch, time, n_inputs), got {input_spikes.shape}"
        )
        assert input_spikes.shape[2] == self.n_inputs, (
            f"input_spikes shape[2]={input_spikes.shape[2]}, expected {self.n_inputs}"
        )
        if input_spikes.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size {input_spikes.shape[0]} != model batch size "
                f"{self.batch_size}; call reset_state(batch_size=...) first"
            )


# =====================================================================
# Module-level helpers
# =====================================================================


def _projection_mask_np(proj: Projection) -> NDArray[np.bool_]:
    """Get a per-pair (n_src, n_tgt) bool mask from a Projection (init-time)."""
    from connectome_snns.network_simulators.projections import (
        FrozenProjection,
        ScalingFactorProjection,
        FullRankProjection,
        LowRankProjection,
    )

    if isinstance(proj, ScalingFactorProjection):
        return proj.connectome.cpu().numpy() != 0
    if isinstance(proj, FrozenProjection):
        return proj._weights.cpu().numpy() != 0
    if isinstance(proj, (FullRankProjection, LowRankProjection)):
        return proj.mask.cpu().numpy() != 0
    raise TypeError(f"Unknown Projection subclass: {type(proj).__name__}")
