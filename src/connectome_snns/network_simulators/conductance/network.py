"""The machinery shared by the conductance-based LIF networks.

A network of ``N`` neurons receives synaptic input from one or more *input sides*: its
own neurons (recurrent) and external inputs (feedforward). Connectivity on each side
is a dict of ``Projection`` objects, one per (source cell type, target cell type).

Internally the input is organised into **pathways** — one per source cell type that
has projections, on each side. A pathway has a single caching mode, shared by all its
projections, which decides what is computed when:

* ``"fixed"`` — weights fixed at construction (``FrozenProjection``).
* ``"scaling_factors"`` — the connectome is fixed at construction; each forward call
  scales the pathway's conductance kernel by the current scaling factor of each
  target neuron's type (``ScalingFactorProjection``).
* ``"weights"`` — the weight matrix is assembled from the projections on every forward
  call, so gradients reach their parameters (``FullRankProjection``,
  ``LowRankProjection`` and subclasses).

The time step itself lives in ``dynamics.py`` (PyTorch) and ``kernels.py`` (Triton).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from connectome_snns.network_simulators.checkpoint_compat import drop_legacy_buffers
from connectome_snns.network_simulators.conductance import engine
from connectome_snns.network_simulators.conductance.dynamics import (
    Drive,
    Physiology,
    Run,
)
from connectome_snns.network_simulators.conductance.synapses import (
    SynapseRegistry,
    conductance_kernel,
    neuron_tables,
)
from connectome_snns.network_simulators.projections import Projection

PairKey = tuple[str, str]  # (source cell type name, target cell type name)
#: Dtypes the kernels can store conductances and recurrent weights in.
STORAGE_DTYPES = (torch.float32, torch.float16)


@dataclass
class InputSide:
    """One source of synaptic input and its connectivity.

    Attributes:
        projections: ``{(source_name, target_name): Projection}``.
        cell_params, synapse_params: The source cell types and the synapses they make.
        cell_type_indices: Cell type of each source (each input row).
        recurrent: Whether the sources are the network's own neurons.
        attribute: Name of the ``nn.ModuleDict`` holding the projections.
        mask_buffer: Name of the buffer holding the side's connectivity mask.
    """

    projections: dict[PairKey, Projection]
    cell_params: list[dict]
    synapse_params: list[dict]
    cell_type_indices: np.ndarray
    recurrent: bool
    attribute: str
    mask_buffer: str


@dataclass(frozen=True)
class Pathway:
    """One source cell type on one input side (see the module docstring)."""

    index: int
    side: int
    cell_type: int
    mode: str  # "fixed" | "scaling_factors" | "weights"


class ConductanceNetwork(nn.Module):
    """Base class of the conductance-based LIF networks.

    Subclasses describe their input sides and name their outputs; this class owns the
    physiology buffers, the synapse registry, the pathways, the simulation state and
    the forward pass. The state ``v`` ``(B, N)`` and the conductances ``(B, N, 2, S)``
    continue from one ``forward`` call to the next until ``reset_state``.
    """

    #: Name of the conductance state (also its checkpoint key).
    conductance_name = "g"

    def __init__(
        self,
        *,
        dt: float,
        cell_params: list[dict],
        cell_type_indices: np.ndarray,
        sides: list[InputSide],
        unify_synapses: bool,
        surrgrad_scale: float,
        batch_size: int,
        spike_mode: str,
        track_variables: bool,
        track_gradients: bool,
        track_batch_idx: int | None,
        storage_dtype: torch.dtype | str = torch.float32,
    ):
        super().__init__()
        self.register_load_state_dict_pre_hook(drop_legacy_buffers)
        validate(dt, cell_params, cell_type_indices, sides, surrgrad_scale)
        self.batch_size = batch_size
        self.spike_mode = spike_mode
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
        self.storage_dtype = storage_dtype
        #: Input columns that can carry a gradient — e.g. another layer's spikes —
        #: as a slice; None means any. Lets the backward pass skip pure data columns.
        self.differentiable_inputs = None
        self.cell_params = cell_params
        self.n_cell_types = len(cell_params)
        self.n_neurons = int(cell_type_indices.shape[0])
        self._sides = sides
        self._names = {c["cell_id"]: c["name"] for c in cell_params}

        self.registry = SynapseRegistry.build(
            [(s.cell_params, s.synapse_params) for s in sides], unify=unify_synapses
        )
        if unify_synapses and len(self.registry) < sum(
            len(s.synapse_params) for s in sides
        ):
            shared = sum(len(s.synapse_params) for s in sides) - len(self.registry)
            print(
                f"  Unified synapse registry: {len(self.registry)} types "
                f"({shared} shared between recurrent/feedforward)"
            )

        self.register_buffer("cell_type_indices", _long(cell_type_indices))
        for side in sides:
            setattr(
                self,
                side.attribute,
                nn.ModuleDict(
                    {f"{s}__{t}": p for (s, t), p in side.projections.items()}
                ),
            )
            self.register_buffer(side.mask_buffer, torch.from_numpy(self._mask(side)))
        for name, table in neuron_tables(cell_params, cell_type_indices).items():
            self.register_buffer(name, table)
        for name, table in self.registry.tables().items():
            self.register_buffer(name, table)
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )
        self._register_synapse_maps(unify_synapses)

        self.pathways = self._pathways()
        self.set_timestep(dt)

    @property
    def storage_dtype(self) -> torch.dtype:
        """What the CUDA kernels keep conductances and recurrent weights in.

        ``torch.float16`` (or ``"float16"``) nearly halves the memory traffic of a
        recurrent step — the teacher simulates 1.35x faster — and halves the
        conductances saved for the backward pass; arithmetic stays in float32, and
        gradients are float32. The run is then a different realisation of the same
        network: on the teacher, firing rates agree with float32 within 0.4% and
        gradients point the same way (cosine 0.9999), but individual spike times part
        after ~100 ms. Rounding happens wherever the state is stored, so splitting a
        feedforward network's simulation into chunks is no longer exactly neutral.

        Default float32, bit-identical to before the option existed. The PyTorch
        reference (CPU) always computes in float32.
        """
        return self._storage_dtype

    @storage_dtype.setter
    def storage_dtype(self, dtype: torch.dtype | str) -> None:
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"storage_dtype must be one of {STORAGE_DTYPES}, got {dtype}"
            )
        self._storage_dtype = dtype

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _mask(self, side: InputSide) -> np.ndarray:
        """``(n_sources, N)`` bool: where the side's projections have connections."""
        mask = np.zeros((len(side.cell_type_indices), self.n_neurons), dtype=bool)
        for (src, tgt), projection in side.projections.items():
            rows, cols = self._block(side, src, tgt)
            if rows.size and cols.size:
                mask[np.ix_(rows, cols)] = projection_mask(projection)
        return mask

    def _block(self, side: InputSide, src: str, tgt: str):
        """Rows (sources) and columns (targets) of the block of pair ``(src, tgt)``."""
        src_id = next(c["cell_id"] for c in side.cell_params if c["name"] == src)
        tgt_id = next(c["cell_id"] for c in self.cell_params if c["name"] == tgt)
        rows = np.flatnonzero(side.cell_type_indices == src_id)
        cols = np.flatnonzero(np.asarray(self.cell_type_indices.cpu()) == tgt_id)
        return rows, cols

    def _register_synapse_maps(self, unified: bool) -> None:
        """Which synapse types each cell type makes, as the original simulators stored
        them — kept so that checkpoints written by them still load.

        ``cell_to_synapse_mask{_FF}``: per side, (source cell type, synapse type) bool.
        ``synapse_to_cell_id``: with a unified registry, the recurrent cell type making
        each synapse type (-1 if none); otherwise ``synapse_to_cell_id_FF``, the input
        cell type making each.
        """
        for side_index, side in enumerate(self._sides):
            made = self.registry.made_by(side_index, side.synapse_params)
            table = torch.zeros((max(made) + 1, len(self.registry)), dtype=torch.bool)
            for cell_type, synapses in made.items():
                table[cell_type] = synapses
            suffix = side.mask_buffer.removeprefix("weights_mask")
            self.register_buffer(f"cell_to_synapse_mask{suffix}", table)
        if unified:
            ids = {c["name"]: c["cell_id"] for c in self._sides[0].cell_params}
            maker = [ids.get(s["signature"][0], -1) for s in self.registry.synapses]
            self.register_buffer("synapse_to_cell_id", torch.tensor(maker))
        else:
            maker = [s["cell_id"] for s in self.registry.synapses]
            self.register_buffer("synapse_to_cell_id_FF", torch.tensor(maker))

    def _pathways(self) -> list[Pathway]:
        """One pathway per source cell type with synapses and projections, per side."""
        pathways, made = [], []
        for side_index, side in enumerate(self._sides):
            names = {c["cell_id"]: c["name"] for c in side.cell_params}
            modes = {}
            for (src, _), projection in side.projections.items():
                previous = modes.setdefault(src, projection.caching_mode)
                if previous != projection.caching_mode:
                    raise ValueError(
                        f"source cell type {src!r} has projections with mixed caching "
                        f"modes ({previous!r} and {projection.caching_mode!r}); all "
                        "projections from one source must use the same mode"
                    )
            for cell_type, synapses in self.registry.made_by(
                side_index, side.synapse_params
            ).items():
                mode = modes.get(names[cell_type])
                if mode is None:
                    continue  # no projections: contributes nothing
                pathway = Pathway(len(pathways), side_index, cell_type, mode)
                rows = np.flatnonzero(side.cell_type_indices == cell_type)
                self.register_buffer(
                    f"_rows_{pathway.index}", _long(rows), persistent=False
                )
                pathways.append(pathway)
                made.append(synapses)
        self.register_buffer(
            "_made",
            torch.stack(made)
            if made
            else torch.zeros((0, len(self.registry)), dtype=torch.bool),
            persistent=False,
        )
        # Recurrent pathways first: the order the dynamics inject them in.
        return sorted(pathways, key=lambda p: not self._sides[p.side].recurrent)

    def _assemble(self, pathway: Pathway, connectome: bool = False) -> torch.Tensor:
        """The ``(n_rows, N)`` weights of a pathway, from its projections' ``forward``.

        With ``connectome=True``, scaling-factor projections contribute their unscaled
        connectome instead (the scaling factor is applied to the kernel).
        """
        side = self._sides[pathway.side]
        rows = getattr(self, f"_rows_{pathway.index}")
        W = torch.zeros((rows.numel(), self.n_neurons), device=self.device)
        projections = getattr(self, side.attribute)
        src = next(
            c["name"] for c in side.cell_params if c["cell_id"] == pathway.cell_type
        )
        for tgt_id, tgt in self._names.items():
            key = f"{src}__{tgt}"
            if key not in projections:
                continue
            cols = (self.cell_type_indices == tgt_id).nonzero(as_tuple=True)[0]
            if cols.numel():
                projection = projections[key]
                W[:, cols] = projection.connectome if connectome else projection()
        return W

    def _cache_weights(self) -> None:
        """Snapshot the weights of fixed and scaling-factor pathways."""
        for pathway in self.pathways:
            if pathway.mode != "weights":
                W = self._assemble(
                    pathway, connectome=pathway.mode == "scaling_factors"
                )
                self.register_buffer(
                    f"_weights_{pathway.index}", W.detach(), persistent=False
                )

    def set_timestep(self, dt: float) -> None:
        """Set the time step, recompute what depends on it, and reset the state."""
        assert isinstance(dt, float), "dt must be a float"
        assert dt > 0, "Timestep dt must be positive"
        self.register_buffer("dt", torch.tensor(dt, device=self.device))
        self.register_buffer("tau_syn", torch.stack((self.tau_rise, self.tau_decay)))
        self.register_buffer("alpha", torch.exp(-self.dt / self.tau_syn))
        self.register_buffer("beta", torch.exp(-self.dt / self.tau_mem))
        self.register_buffer(
            "g_scale", conductance_kernel(self.tau_rise, self.tau_decay, self.g_bar)
        )
        self._cache_weights()
        self.reset_state()

    # ------------------------------------------------------------------
    # Parameters as matrices
    # ------------------------------------------------------------------

    def _weight_matrix(self, side_index: int) -> torch.Tensor:
        """``(n_sources, N)`` weights of a side, assembled from its projections."""
        side = self._sides[side_index]
        W = torch.zeros(
            (len(side.cell_type_indices), self.n_neurons), device=self.device
        )
        for (src, tgt), projection in side.projections.items():
            rows, cols = self._block(side, src, tgt)
            if rows.size and cols.size:
                W[np.ix_(rows, cols)] = getattr(self, side.attribute)[f"{src}__{tgt}"]()
        return W

    def _scaling_matrix(self, side_index: int) -> torch.Tensor:
        """``(n_source_types, n_target_types)`` scaling factors of a side (1 if none)."""
        side = self._sides[side_index]
        ids = {c["name"]: c["cell_id"] for c in side.cell_params}
        tgt_ids = {c["name"]: c["cell_id"] for c in self.cell_params}
        sf = torch.ones((len(side.cell_params), self.n_cell_types), device=self.device)
        for key, projection in getattr(self, side.attribute).items():
            if projection.caching_mode == "scaling_factors":
                src, tgt = key.split("__", 1)
                sf[ids[src], tgt_ids[tgt]] = projection.kernel_factor()
        return sf

    @property
    def device(self) -> torch.device:
        return self.theta.device

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _drive(self) -> Drive:
        """The pathways' current weights and kernel scales, for one forward call."""
        scaling = {}
        ones = torch.ones(self.n_neurons, device=self.device)
        scales, recurrent, feedforward = [], [], []
        for pathway in self.pathways:
            if pathway.mode == "weights":
                W = self._assemble(pathway)
            else:
                W = getattr(self, f"_weights_{pathway.index}")
            if pathway.mode == "scaling_factors":
                if pathway.side not in scaling:
                    scaling[pathway.side] = self._scaling_matrix(pathway.side)
                scales.append(
                    scaling[pathway.side][pathway.cell_type, self.cell_type_indices]
                )
            else:
                scales.append(ones)
            rows = getattr(self, f"_rows_{pathway.index}")
            (recurrent if self._sides[pathway.side].recurrent else feedforward).append(
                (rows, W)
            )
        order = [p.index for p in self.pathways]
        return Drive(
            made=self._made[order],
            scale=torch.stack(scales)
            if scales
            else ones.new_zeros((0, self.n_neurons)),
            recurrent=recurrent,
            feedforward=feedforward,
            differentiable_inputs=self.differentiable_inputs,
        )

    def _physiology(self) -> Physiology:
        return Physiology(
            theta=self.theta, U_reset=self.U_reset, E_L=self.E_L, C_m=self.C_m,
            beta=self.beta, E_syn=self.E_syn, g_clip=self.g_clip, decay=self.alpha,
            jump=self.g_scale, dt=self.dt, surr_scale=self.surrgrad_scale,
            spike_mode=self.spike_mode, storage_dtype=self.storage_dtype,
        )  # fmt: skip

    # The time loop runs in the engine (Triton kernels on CUDA): there is nothing for
    # torch.compile to fuse here, so a compiled model calls this eagerly.
    @torch.compiler.disable
    def _simulate(self, input_spikes, probabilities: bool = False) -> Run:
        """Validate the input, simulate it from the current state, and keep the state."""
        if isinstance(input_spikes, np.ndarray):
            input_spikes = torch.from_numpy(input_spikes).to(self.device)
        self._validate_forward(input_spikes)
        retain = [] if self.track_gradients else None
        run = engine.simulate(
            self.v,
            self._conductances,
            input_spikes.float(),
            self._drive(),
            self._physiology(),
            probabilities=probabilities,
            track=self.track_variables,
            track_batch_idx=self.track_batch_idx,
            retain=retain,
        )
        if retain is not None:
            self._retained = retain
        # Detach: the next chunk must not backpropagate into this one's graph.
        self.v = run.v.detach()
        self._conductances = run.g.detach()
        return run

    def _validate_forward(self, input_spikes: torch.Tensor) -> None:
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

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def _conductances(self) -> torch.Tensor:
        return getattr(self, self.conductance_name)

    @_conductances.setter
    def _conductances(self, value: torch.Tensor) -> None:
        setattr(self, self.conductance_name, value)

    def reset_state(self, batch_size: int | None = None) -> None:
        """Voltages to reset, conductances to zero; optionally change the batch size."""
        if batch_size is not None:
            self.batch_size = batch_size
        v = self.U_reset.expand(self.batch_size, self.n_neurons).clone()
        # (B, N, 2, S), laid out in memory as (B, 2, S, N): see kernels.py.
        g = torch.zeros(
            (self.batch_size, 2, len(self.registry), self.n_neurons), device=self.device
        ).permute(0, 3, 1, 2)
        self.register_buffer("v", v, persistent=False)
        self.register_buffer(self.conductance_name, g, persistent=False)

    def get_checkpoint_state(self) -> dict[str, np.ndarray]:
        return {
            "v": self.v.cpu().numpy(),
            self.conductance_name: self._conductances.cpu().numpy(),
        }

    def load_checkpoint_state(self, state: dict[str, np.ndarray]) -> None:
        self.v = torch.from_numpy(state["v"]).to(self.device)
        self._conductances = torch.from_numpy(state[self.conductance_name]).to(
            self.device
        )

    def get_tracked_gradients(self) -> dict[str, torch.Tensor]:
        """Gradients of the tracked per-step ``v``, conductances and spikes.

        Call after ``backward()`` on a loss from a forward pass with
        ``track_gradients=True``. Shapes ``(T, B, ...)``; zeros where no gradient.
        """
        if not self.track_gradients:
            raise RuntimeError("track_gradients must be True during forward pass")
        if not getattr(self, "_retained", None):
            raise RuntimeError(
                "No tracked tensors found. Did you call forward() with track_gradients=True?"
            )
        names = ("grad_v", f"grad_{self.conductance_name}", "grad_s")
        return {
            name: torch.stack(
                [t.grad.detach().clone() if t.grad is not None else torch.zeros_like(t)
                 for t in tensors]
            )
            for name, tensors in zip(names, zip(*self._retained))
        }  # fmt: skip


# =====================================================================
# Helpers
# =====================================================================


def _long(array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).long()


def projection_mask(projection: Projection) -> np.ndarray:
    """Where a projection has connections, as a bool array (at construction)."""
    for name in ("connectome", "mask", "_weights"):
        tensor = getattr(projection, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy() != 0
    raise TypeError(f"Unknown Projection subclass: {type(projection).__name__}")


def validate(dt, cell_params, cell_type_indices, sides, surrgrad_scale) -> None:
    """Check the construction arguments, with messages naming what is wrong."""
    assert isinstance(dt, (int, float)), "dt must be numeric"
    assert dt > 0, f"dt must be positive, got {dt}"
    assert dt <= 100.0, f"dt unusually large ({dt} ms), check units"
    assert isinstance(surrgrad_scale, (int, float)), "surrgrad_scale must be numeric"
    assert surrgrad_scale > 0, "surrgrad_scale must be positive"

    def contiguous_ids(params, key, what):
        ids = sorted(p[key] for p in params)
        assert ids == list(range(len(ids))), (
            f"{what} must be 0-indexed contiguous, got {ids}"
        )

    contiguous_ids(cell_params, "cell_id", "cell_ids in cell_params")
    assert cell_type_indices.ndim == 1, "cell_type_indices must be 1D"
    assert cell_type_indices.shape[0] > 0, "n_neurons must be positive"
    assert np.all((cell_type_indices >= 0) & (cell_type_indices < len(cell_params))), (
        "cell_type_indices out of range"
    )
    targets = {p["name"]: p["cell_id"] for p in cell_params}
    n_per_target = np.bincount(cell_type_indices, minlength=len(cell_params))

    for side in sides:
        contiguous_ids(side.cell_params, "cell_id", f"{side.attribute} source cell_ids")
        assert len(side.synapse_params) > 0, (
            f"{side.attribute}: synapse_params must not be empty"
        )
        contiguous_ids(
            side.synapse_params, "synapse_id", f"{side.attribute} synapse_ids"
        )
        types = side.cell_type_indices
        assert types.ndim == 1, f"{side.attribute}: source cell types must be 1D"
        assert np.all((types >= 0) & (types < len(side.cell_params))), (
            f"{side.attribute}: source cell type indices out of range"
        )
        sources = {p["name"]: p["cell_id"] for p in side.cell_params}
        n_per_source = np.bincount(types, minlength=len(side.cell_params))
        for (src, tgt), projection in side.projections.items():
            assert src in sources, (
                f"{side.attribute}: source {src!r} not in its cell_params"
            )
            assert tgt in targets, (
                f"{side.attribute}: target {tgt!r} not in cell_params"
            )
            expected = (
                int(n_per_source[sources[src]]),
                int(n_per_target[targets[tgt]]),
            )
            assert (projection.n_source, projection.n_target) == expected, (
                f"{side.attribute} projection ({src!r}, {tgt!r}) shape "
                f"({projection.n_source}, {projection.n_target}) != expected {expected}"
            )
