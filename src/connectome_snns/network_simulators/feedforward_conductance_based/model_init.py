"""I/O utilities for feedforward LIF network parameters (projection-based API)."""

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
PairKey = Tuple[str, str]  # (input_cell_type_name, output_cell_type_name)


class FeedforwardConductanceLIFNetwork_IO(nn.Module):
    """Base class for feedforward-only conductance-based LIF network.

    The simulator treats the entire input matrix as feedforward (true FF
    inputs and chunked-feedforward-unrolled recurrent inputs are both
    rows of ``weights_FF`` and use the same dynamics path). Connectivity
    is supplied as ``projections`` — a single dict mapping
    ``(input_cell_type_name, output_cell_type_name)`` to a ``Projection``.

    Constraint: all projections sharing the same input (source) cell type
    must share the same ``caching_mode``.
    """

    def __init__(
        self,
        dt: float,
        projections: dict[PairKey, Projection],
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: Literal["deterministic", "sample"] = "deterministic",
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        return_probabilities: bool = False,
    ):
        """Initialize the chunked-feedforward conductance-based LIF network.

        Args:
            dt: Simulation timestep (ms).
            projections: ``{(input_ct_name, output_ct_name): Projection}``.
                Input names must appear in ``cell_params_FF``; output
                names in ``cell_params``.
            cell_type_indices: Per-output-neuron cell type ID, shape (n_neurons,).
            cell_type_indices_FF: Per-input-row cell type ID, shape (n_inputs,).
            cell_params: List of output cell-type configs.
            cell_params_FF: List of input cell-type configs (mitral, plus
                any cell types that appear as input rows from
                chunked-feedforward unrolling).
            synapse_params_FF: List of synapse configs (one per
                (input_cell_type, synapse_type) pair).
            surrgrad_scale, batch_size, spike_mode, track_*: as in the
                recurrent simulator.
            return_probabilities: if True, forward also returns the per-step spike
                probability p (sample mode); see the class/forward docstrings.
        """
        super().__init__()

        self.batch_size = batch_size
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
        self.spike_mode = spike_mode
        # When True (escape-noise / sample mode), forward also returns the per-step
        # spike probability p = sigmoid(surrgrad_scale*(v-theta)) so rate losses can
        # filter the *expected* spike train (no sampling variance). See _step_common.
        self.return_probabilities = return_probabilities

        self.cell_params = cell_params
        self.cell_params_FF = cell_params_FF
        self.synapse_params_FF = synapse_params_FF

        self.n_cell_types = len(cell_params)
        self.n_cell_types_FF = len(cell_params_FF)
        self.n_synapse_types_FF = len(synapse_params_FF)

        self.n_neurons = int(cell_type_indices.shape[0])
        self.n_inputs = int(cell_type_indices_FF.shape[0])

        # Cell type name <-> id maps
        self._out_name_to_id: dict[str, int] = {
            p["name"]: p["cell_id"] for p in cell_params
        }
        self._out_id_to_name: dict[int, str] = {
            p["cell_id"]: p["name"] for p in cell_params
        }
        self._in_name_to_id: dict[str, int] = {
            p["name"]: p["cell_id"] for p in cell_params_FF
        }
        self._in_id_to_name: dict[int, str] = {
            p["cell_id"]: p["name"] for p in cell_params_FF
        }

        self._validate(
            dt=dt,
            projections=projections,
            cell_type_indices=cell_type_indices,
            cell_type_indices_FF=cell_type_indices_FF,
            cell_params=cell_params,
            cell_params_FF=cell_params_FF,
            synapse_params_FF=synapse_params_FF,
            surrgrad_scale=surrgrad_scale,
        )

        # Register projections
        self.projections = nn.ModuleDict(
            {self._pair_to_key(p): proj for p, proj in projections.items()}
        )
        self._pairs: list[PairKey] = list(projections.keys())

        # Per-input-cell-type caching mode
        self._mode_per_ct: list[str | None] = self._derive_modes_per_source_ct(
            projections, self.n_cell_types_FF, self._in_id_to_name
        )
        self._has_weights_mode = "weights" in self._mode_per_ct
        self._has_scaling_mode = "scaling_factors" in self._mode_per_ct

        # Build weights_mask_FF from projection connectomes
        weights_mask_FF = self._assemble_mask_from_projections(
            projections,
            cell_type_indices_FF,
            cell_type_indices,
            self._in_id_to_name,
            self._out_id_to_name,
            (self.n_inputs, self.n_neurons),
        )
        self.register_buffer("weights_mask_FF", torch.from_numpy(weights_mask_FF))

        # Cell type indices as buffers
        self.register_buffer(
            "cell_type_indices", torch.from_numpy(np.asarray(cell_type_indices)).long()
        )
        self.register_buffer(
            "cell_type_indices_FF",
            torch.from_numpy(np.asarray(cell_type_indices_FF)).long(),
        )

        # Synapse / cell type masks
        self._create_synapse_to_cell_mappings(synapse_params_FF)
        self._create_cell_to_synapse_masks(synapse_params_FF)
        self._create_cell_type_masks(synapse_params_FF)

        # Neuron-indexed physiology
        neuron_params = self._create_neuron_param_arrays(cell_params, cell_type_indices)
        for param_name, param_array in neuron_params.items():
            self.register_buffer(param_name, param_array)

        # Synapse parameter arrays
        synapse_param_arrays_FF = self._create_synapse_param_arrays(synapse_params_FF)
        for param_name, param_tensor in synapse_param_arrays_FF.items():
            self.register_buffer(param_name, param_tensor)

        # Surrogate gradient scale
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )

        # Initialise dt-dependent and cached buffers
        self.set_timestep(dt)

    # ======================================================================
    # Helpers
    # ======================================================================

    @staticmethod
    def _pair_to_key(pair: PairKey) -> str:
        return f"{pair[0]}__{pair[1]}"

    @staticmethod
    def _derive_modes_per_source_ct(
        projections: dict[PairKey, Projection],
        n_source_cell_types: int,
        id_to_name: dict[int, str],
    ) -> list[str | None]:
        modes: list[str | None] = [None] * n_source_cell_types
        for (src_name, _), proj in projections.items():
            src_id = next((k for k, v in id_to_name.items() if v == src_name), None)
            if src_id is None:
                raise ValueError(
                    f"projection source '{src_name}' not in cell_params_FF"
                )
            mode = proj.caching_mode
            if modes[src_id] is None:
                modes[src_id] = mode
            elif modes[src_id] != mode:
                raise ValueError(
                    f"input cell type '{src_name}' has projections with mixed "
                    f"caching modes ({modes[src_id]!r} and {mode!r}). All "
                    f"projections sharing a source must use the same mode."
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
        from connectome_snns.network_simulators.conductance_based.model_init import (
            _projection_mask_np,
        )

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
            block_mask = _projection_mask_np(proj)
            if block_mask.shape != (src_idx.size, tgt_idx.size):
                raise ValueError(
                    f"Projection ({src_name!r}, {tgt_name!r}) has shape "
                    f"{block_mask.shape}, expected {(src_idx.size, tgt_idx.size)}"
                )
            mask[np.ix_(src_idx, tgt_idx)] = block_mask
        return mask

    def _resolve_modes(self) -> list[str | None]:
        """Return per-cell-type modes for the FF pathway."""
        n = len(self.cell_type_masks_FF)
        return list(self._mode_per_ct[:n])

    # ======================================================================
    # Assembly properties
    # ======================================================================

    def _assemble_weights_FF(self) -> torch.Tensor:
        out = torch.zeros(
            (self.n_inputs, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )
        for pair in self._pairs:
            proj = self.projections[self._pair_to_key(pair)]
            src_id = self._in_name_to_id[pair[0]]
            tgt_id = self._out_name_to_id[pair[1]]
            src_idx = (self.cell_type_indices_FF == src_id).nonzero(as_tuple=True)[0]
            tgt_idx = (self.cell_type_indices == tgt_id).nonzero(as_tuple=True)[0]
            if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                continue
            block = proj()
            out.index_put_(
                (src_idx[:, None], tgt_idx[None, :]), block, accumulate=False
            )
        return out

    @property
    def weights_FF(self) -> torch.Tensor:
        return self._assemble_weights_FF()

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        sf = torch.ones(
            (self.n_cell_types_FF, self.n_cell_types),
            dtype=torch.float32,
            device=self.device,
        )
        for pair, proj in zip(self._pairs, self.projections.values()):
            if isinstance(proj, ScalingFactorProjection):
                src_id = self._in_name_to_id[pair[0]]
                tgt_id = self._out_name_to_id[pair[1]]
                sf[src_id, tgt_id] = proj.kernel_factor()
        return sf

    @property
    def device(self):
        for proj in self.projections.values():
            for p in proj.parameters():
                return p.device
            for b in proj.buffers():
                return b.device
        return self.weights_mask_FF.device

    # ======================================================================
    # Caching pre-compute (per source cell type)
    # ======================================================================

    def _precompute_weight_products(self) -> None:
        """Precompute ``connection_weights_{k}`` and ``synapse_kernel_{k}``
        per source cell type, classifying caching mode by projection.
        """
        self._fixed_cell_types: list[int] = []
        self._weights_cell_types: list[int] = []
        self._scaling_cell_types: list[int] = []

        modes = self._resolve_modes()
        weights_FF_now = self._assemble_weights_FF().detach()
        sf_FF_now = self.scaling_factors_FF.detach()

        for k in range(len(self.cell_type_masks_FF)):
            mask = self.cell_type_masks_FF[k]
            syn_mask = self.cell_to_synapse_mask_FF[k]
            if not syn_mask.any():
                continue

            mode = modes[k]
            g_kernel = self.g_scale[None, :, syn_mask]  # (1, n_rise, n_syn)

            if mode is None:
                conn_w = (
                    weights_FF_now[mask, :]
                    * sf_FF_now[k, self.cell_type_indices][None, :]
                ).detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
                self.register_buffer(
                    f"connection_weights_{k}", conn_w, persistent=False
                )
                self.register_buffer(f"synapse_kernel_{k}", kernel, persistent=False)
                self._fixed_cell_types.append(k)
            elif mode == "weights":
                kernel = (
                    sf_FF_now[k, self.cell_type_indices][:, None, None] * g_kernel
                ).detach()
                self.register_buffer(f"synapse_kernel_{k}", kernel, persistent=False)
                self._weights_cell_types.append(k)
            elif mode == "scaling_factors":
                conn_w = weights_FF_now[mask, :].detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
                self.register_buffer(
                    f"connection_weights_{k}", conn_w, persistent=False
                )
                self.register_buffer(f"synapse_kernel_{k}", kernel, persistent=False)
                self._scaling_cell_types.append(k)
            elif mode == "fixed":
                # FrozenProjection: weights are constant, no scaling factor.
                # Both conn_w and kernel are fully cached at init (same as
                # the None branch), so it joins ``_fixed_cell_types`` and
                # the runtime loop just looks the buffers up.
                conn_w = weights_FF_now[mask, :].detach()
                kernel = g_kernel.expand(self.n_neurons, -1, -1).contiguous().detach()
                self.register_buffer(
                    f"connection_weights_{k}", conn_w, persistent=False
                )
                self.register_buffer(f"synapse_kernel_{k}", kernel, persistent=False)
                self._fixed_cell_types.append(k)
            else:
                raise ValueError(f"Unknown mode {mode!r}")

        self.reset_state()

    # ======================================================================
    # State / checkpointing
    # ======================================================================

    def reset_state(self, batch_size: int | None = None) -> None:
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
        g_FF = torch.zeros(
            (self.batch_size, self.n_neurons, 2, self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )
        self.register_buffer("g_FF", g_FF, persistent=False)

    def get_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {"v": self.v.cpu().numpy(), "g_FF": self.g_FF.cpu().numpy()}

    def load_checkpoint_state(self, state: dict[str, np.ndarray]) -> None:
        self.v = torch.from_numpy(state["v"]).to(self.device)
        self.g_FF = torch.from_numpy(state["g_FF"]).to(self.device)

    def _register_parameter_or_buffer(
        self, name: str, value: torch.Tensor | np.ndarray, trainable: bool = False
    ) -> None:
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
    # set_timestep + helper methods (kept structurally similar)
    # ======================================================================

    def set_timestep(self, dt: float) -> None:
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

    def _create_neuron_param_arrays(
        self,
        cell_params: list[dict],
        cell_type_indices: IntArray,
    ) -> dict[str, torch.Tensor]:
        required_param_names = ["tau_mem", "theta", "U_reset", "E_L", "g_L", "tau_ref"]
        n_cell_types = len(cell_params)
        neuron_params = {}
        for param_name in required_param_names:
            param_lookup = np.zeros(n_cell_types, dtype=np.float32)
            for cell in cell_params:
                param_lookup[cell["cell_id"]] = cell[param_name]
            neuron_params[param_name] = torch.from_numpy(
                param_lookup[cell_type_indices]
            )
        neuron_params["C_m"] = neuron_params["tau_mem"] * neuron_params["g_L"]
        return neuron_params

    def _create_synapse_param_arrays(
        self,
        synapse_params: list[dict],
    ) -> dict[str, torch.Tensor]:
        required_param_names = ["tau_rise", "tau_decay", "E_syn", "g_bar"]
        n_synapse_types = len(synapse_params)
        synapse_param_arrays: dict[str, torch.Tensor] = {}
        for param_name in required_param_names:
            param_lookup = np.zeros(n_synapse_types, dtype=np.float32)
            for synapse in synapse_params:
                param_lookup[synapse["synapse_id"]] = synapse[param_name]
            synapse_param_arrays[param_name] = torch.from_numpy(param_lookup)
        # g_clip with default inf
        g_clip_array = np.full(n_synapse_types, float("inf"), dtype=np.float32)
        for synapse in synapse_params:
            if "g_clip" in synapse:
                g_clip_array[synapse["synapse_id"]] = synapse["g_clip"]
        synapse_param_arrays["g_clip"] = torch.from_numpy(g_clip_array)
        return synapse_param_arrays

    def _create_synapse_to_cell_mappings(self, synapse_params_FF: list[dict]) -> None:
        synapse_to_cell_mapping_FF = np.zeros(self.n_synapse_types_FF, dtype=np.int64)
        for synapse in synapse_params_FF:
            synapse_to_cell_mapping_FF[synapse["synapse_id"]] = synapse["cell_id"]
        self.register_buffer(
            "synapse_to_cell_id_FF", torch.from_numpy(synapse_to_cell_mapping_FF)
        )

    def _create_cell_to_synapse_masks(self, synapse_params_FF: list[dict]) -> None:
        ff_cell_types = sorted(set(s["cell_id"] for s in synapse_params_FF))
        max_ff_cell_type = max(ff_cell_types) if ff_cell_types else -1
        cell_to_synapse_mask_FF = torch.zeros(
            max_ff_cell_type + 1, self.n_synapse_types_FF, dtype=torch.bool
        )
        for cell_type in ff_cell_types:
            cell_to_synapse_mask_FF[cell_type, :] = (
                self.synapse_to_cell_id_FF == cell_type
            )
        self.register_buffer("cell_to_synapse_mask_FF", cell_to_synapse_mask_FF)

    def _create_cell_type_masks(self, synapse_params_FF: list[dict]) -> None:
        ff_cell_types = sorted(set(s["cell_id"] for s in synapse_params_FF))
        masks = []
        for cell_type in ff_cell_types:
            masks.append(self.cell_type_indices_FF == cell_type)
        self._cell_type_masks_FF = masks

    @property
    def cell_type_masks_FF(self) -> list[torch.Tensor]:
        return self._cell_type_masks_FF

    # ======================================================================
    # Validation
    # ======================================================================

    def _validate(
        self,
        dt: float,
        projections: dict[PairKey, Projection],
        cell_type_indices: IntArray,
        cell_type_indices_FF: IntArray,
        cell_params: list[dict],
        cell_params_FF: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
    ) -> None:
        assert isinstance(dt, (int, float)), "dt must be numeric"
        assert dt > 0, f"dt must be positive, got {dt}"

        cell_ids = [p["cell_id"] for p in cell_params]
        n_cell_types = max(cell_ids) + 1 if cell_ids else 0
        assert sorted(cell_ids) == list(range(n_cell_types)), (
            "cell_ids in cell_params must be 0-indexed contiguous"
        )
        assert cell_type_indices.ndim == 1
        assert np.all(cell_type_indices >= 0) and np.all(
            cell_type_indices < n_cell_types
        )

        cell_ids_FF = [p["cell_id"] for p in cell_params_FF]
        n_cell_types_FF = max(cell_ids_FF) + 1 if cell_ids_FF else 0
        assert sorted(cell_ids_FF) == list(range(n_cell_types_FF))
        assert cell_type_indices_FF.ndim == 1
        assert np.all(cell_type_indices_FF >= 0) and np.all(
            cell_type_indices_FF < n_cell_types_FF
        )

        assert len(synapse_params_FF) > 0
        syn_ids = [p["synapse_id"] for p in synapse_params_FF]
        n_syn = max(syn_ids) + 1
        assert sorted(syn_ids) == list(range(n_syn))

        out_name_to_id = {p["name"]: p["cell_id"] for p in cell_params}
        in_name_to_id = {p["name"]: p["cell_id"] for p in cell_params_FF}

        n_per_in = np.bincount(cell_type_indices_FF, minlength=n_cell_types_FF)
        n_per_out = np.bincount(cell_type_indices, minlength=n_cell_types)

        for (src, tgt), proj in projections.items():
            assert src in in_name_to_id, (
                f"projection source '{src}' not in cell_params_FF"
            )
            assert tgt in out_name_to_id, (
                f"projection target '{tgt}' not in cell_params"
            )
            expected = (
                int(n_per_in[in_name_to_id[src]]),
                int(n_per_out[out_name_to_id[tgt]]),
            )
            assert (proj.n_source, proj.n_target) == expected, (
                f"projection ({src!r}, {tgt!r}) shape "
                f"({proj.n_source}, {proj.n_target}) != expected {expected}"
            )

        assert isinstance(surrgrad_scale, (int, float))
        assert surrgrad_scale > 0

    def _validate_forward(self, input_spikes: torch.Tensor) -> None:
        assert input_spikes is not None
        assert isinstance(input_spikes, torch.Tensor)
        assert input_spikes.device == self.device
        assert input_spikes.ndim == 3
        assert input_spikes.shape[2] == self.n_inputs
        if input_spikes.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size {input_spikes.shape[0]} != model "
                f"batch size {self.batch_size}; call reset_state(batch_size=...) first"
            )
