"""The machinery shared by the current-based LIF networks.

Neurons in natural units: the voltage resets to 0 and spikes at 1. Each synapse type
carries an exponentially decaying current with a sign (excitatory +1, inhibitory -1).
One step, for every neuron of every trial:

1. **Spike** if ``v > 1`` (Heaviside with a surrogate gradient).
2. **Decay** every synaptic current, then **inject** this step's input: each source cell
   type's activation (presynaptic spikes times weights) into each synapse type it
   makes.
3. **Integrate** ``v <- (beta v + sum_j sign_j I_j) (1 - s)``.

Unlike the conductance-based networks, gradients flow through the recurrent spikes.

Connectivity is a weight matrix per input side (recurrent neurons, feedforward inputs)
with an optional ``(source type, target type)`` scaling-factor matrix. What is trained
is set by ``optimisable``: the weights (log-parametrised, within the connectivity mask;
scaling factors are then not applied), the scaling factors (log-parametrised) of one
or both sides, or nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from connectome_snns.network_simulators.checkpoint_compat import drop_legacy_buffers
from connectome_snns.training_utils.surrogate_gradients import SurrGradSpike

OptimisableParams = Literal[
    "weights",
    "scaling_factors",
    "scaling_factors_recurrent",
    "scaling_factors_feedforward",
    None,
]


@dataclass
class CurrentSide:
    """One source of synaptic input.

    Attributes:
        suffix: Appended to the names of the side's parameters and buffers ("" for
            the recurrent side, "_FF" for the feedforward one).
        state: Name of the side's synaptic current state, ``(B, N, n_synapse_types)``.
        weights: ``(n_sources, N)`` non-negative weights.
        cell_type_indices: Cell type of each source.
        synapse_params: Synapse types made by the source cell types ('synapse_id',
            'cell_id', 'name', 'tau_syn', 'type').
        scaling_factors: ``(n_source_types, n_cell_types)``, or None for ones.
        mask: Connectivity mask (required when learning weights; default weights != 0).
        learn_weights, learn_scaling: What is trained on this side.
        recurrent: Whether the sources are the network's own neurons.
    """

    suffix: str
    state: str
    weights: np.ndarray
    cell_type_indices: np.ndarray
    synapse_params: list[dict]
    scaling_factors: np.ndarray | None
    mask: np.ndarray | None
    learn_weights: bool
    learn_scaling: bool
    recurrent: bool


@dataclass(frozen=True)
class _Pathway:
    side: int
    cell_type: int
    index: int


class CurrentNetwork(nn.Module):
    """Base class of the current-based LIF networks.

    The state (``v`` and each side's synaptic currents) continues across ``forward``
    calls; call ``reset_state()`` before an independent simulation.
    """

    def __init__(
        self,
        *,
        dt: float,
        cell_params: list[dict],
        cell_type_indices: np.ndarray,
        sides: list[CurrentSide],
        surrgrad_scale: float,
        batch_size: int,
        track_variables: bool,
        track_gradients: bool,
        track_batch_idx: int | None,
        use_tqdm: bool,
    ):
        super().__init__()
        self.register_load_state_dict_pre_hook(drop_legacy_buffers)
        validate(dt, cell_params, sides, surrgrad_scale)
        self.batch_size = batch_size
        self.track_variables = track_variables
        self.track_gradients = track_gradients
        self.track_batch_idx = track_batch_idx
        self.use_tqdm = use_tqdm
        self.cell_params = cell_params
        self.n_cell_types = len(cell_params)
        self.n_neurons = int(cell_type_indices.shape[0])
        self._sides = sides

        self.register_buffer(
            "_neuron_types", _long(cell_type_indices), persistent=False
        )
        self.register_buffer("cell_type_indices", _long(cell_type_indices).int())
        tau_mem = np.zeros(len(cell_params), dtype=np.float32)
        for cell in cell_params:
            tau_mem[cell["cell_id"]] = cell["tau_mem"]
        self.register_buffer("tau_mem", torch.from_numpy(tau_mem[cell_type_indices]))
        for side in sides:
            self._register_side(side)
        self.register_buffer(
            "surrgrad_scale", torch.tensor(surrgrad_scale, dtype=torch.float32)
        )

        self._pathways = [
            _Pathway(s, cell_type, i)
            for i, (s, cell_type) in enumerate(
                (s, c)
                for s, side in enumerate(sides)
                for c in sorted({syn["cell_id"] for syn in side.synapse_params})
            )
        ]
        self._cache_weights()
        self.set_timestep(dt)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _register_side(self, side: CurrentSide) -> None:
        x = side.suffix
        mask = side.mask if side.mask is not None else side.weights != 0
        self.register_buffer(f"weights_mask{x}", torch.from_numpy(np.asarray(mask)))
        if side.learn_weights:
            log_w = np.log(side.weights[np.asarray(mask)] + 1e-8)
            self.register_parameter(f"log_weights{x}_flat", nn.Parameter(_float(log_w)))
        else:
            self.register_buffer(f"_weights{x}_buffer", _float(side.weights))
        sf = side.scaling_factors
        if sf is None:
            n_source_types = (
                len(self.cell_params)
                if side.recurrent
                else int(side.cell_type_indices.max()) + 1
            )
            sf = np.ones((n_source_types, self.n_cell_types), dtype=np.float32)
        if side.learn_scaling:
            self.register_parameter(
                f"log_scaling_factors{x}", nn.Parameter(_float(np.log(sf + 1e-8)))
            )
        else:
            self.register_buffer(f"_scaling_factors{x}_buffer", _float(sf))

        n_types = len(side.synapse_params)
        tau, sign, made_by = (
            np.zeros(n_types, np.float32),
            np.zeros(n_types, np.float32),
            np.zeros(n_types, np.int64),
        )
        for syn in side.synapse_params:
            tau[syn["synapse_id"]] = syn["tau_syn"]
            sign[syn["synapse_id"]] = 1.0 if syn["type"] == "excitatory" else -1.0
            made_by[syn["synapse_id"]] = syn["cell_id"]
        self.register_buffer(f"tau_syn{x}", torch.from_numpy(tau))
        self.register_buffer(f"sign{x}", torch.from_numpy(sign))
        self.register_buffer(f"synapse_to_cell_id{x}", torch.from_numpy(made_by))
        # Which synapse types each source cell type makes (as the original stored it).
        types = sorted({syn["cell_id"] for syn in side.synapse_params})
        table = torch.zeros((types[-1] + 1, n_types), dtype=torch.bool)
        for cell_type in types:
            table[cell_type] = torch.from_numpy(made_by == cell_type)
        self.register_buffer(f"cell_to_synapse_mask{x}", table)
        if not side.recurrent:
            self.register_buffer(
                "cell_type_indices_FF", _long(side.cell_type_indices).int()
            )

    def _weights(self, side_index: int) -> torch.Tensor:
        """``(n_sources, N)`` weights of a side (from the log-weights if learnt)."""
        x = self._sides[side_index].suffix
        if self._sides[side_index].learn_weights:
            mask = getattr(self, f"weights_mask{x}")
            log_w = getattr(self, f"log_weights{x}_flat")
            return torch.zeros_like(mask, dtype=torch.float32).masked_scatter_(
                mask, torch.exp(log_w)
            )
        return getattr(self, f"_weights{x}_buffer")

    def _scaling(self, side_index: int) -> torch.Tensor:
        """``(n_source_types, n_cell_types)`` scaling factors of a side."""
        x = self._sides[side_index].suffix
        if self._sides[side_index].learn_scaling:
            return torch.exp(getattr(self, f"log_scaling_factors{x}"))
        return getattr(self, f"_scaling_factors{x}_buffer")

    def _rows(self, pathway: _Pathway) -> torch.Tensor:
        return getattr(self, f"_rows_{pathway.index}")

    def _cache_weights(self) -> None:
        """Per pathway: its rows, which synapse types it feeds, and fixed weights.

        Fixed weights are cached with their (fixed) scaling factors baked in; learnt
        scaling factors are applied to the activation on every step instead.
        """
        for p in self._pathways:
            side = self._sides[p.side]
            x = side.suffix
            rows = np.flatnonzero(side.cell_type_indices == p.cell_type)
            self.register_buffer(f"_rows_{p.index}", _long(rows), persistent=False)
            made = getattr(self, f"synapse_to_cell_id{x}") == p.cell_type
            self.register_buffer(f"_made_{p.index}", made.float(), persistent=False)
            if side.learn_weights:
                continue
            W = self._weights(p.side)[self._rows(p)]
            if not side.learn_scaling:
                W = W * self._scaling(p.side)[p.cell_type, self._neuron_types][None, :]
            self.register_buffer(f"_weights_{p.index}", W.detach(), persistent=False)

    def set_timestep(self, dt: float) -> None:
        """Set the time step, recompute the decay factors, and reset the state."""
        assert isinstance(dt, float), "dt must be a float"
        assert dt > 0, "Timestep dt must be positive"
        self.register_buffer("dt", torch.tensor(dt, device=self.device))
        for side in self._sides:
            x = side.suffix
            self.register_buffer(
                f"alpha_syn{x}", torch.exp(-self.dt / getattr(self, f"tau_syn{x}"))
            )
        self.register_buffer("beta_mem", torch.exp(-self.dt / self.tau_mem))
        self.reset_state()

    @property
    def device(self) -> torch.device:
        return self.tau_mem.device

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _pathway_weights(self, p: _Pathway) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The pathway's current ``(n_rows, N)`` weights and per-target scale (or None)."""
        side = self._sides[p.side]
        if side.learn_weights:
            return self._weights(p.side)[self._rows(p)], None
        scale = (
            self._scaling(p.side)[p.cell_type, self._neuron_types]
            if side.learn_scaling
            else None
        )
        return getattr(self, f"_weights_{p.index}"), scale

    def _simulate(self, x: torch.Tensor) -> tuple[torch.Tensor, dict | None]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        self._validate_forward(x)
        B, T = x.shape[:2]
        x = x.float()
        currents = [getattr(self, side.state) for side in self._sides]
        alphas = [getattr(self, f"alpha_syn{side.suffix}") for side in self._sides]
        signs = [getattr(self, f"sign{side.suffix}") for side in self._sides]

        drive = []  # per pathway: (side, rows or None, W, scale, made, precomputed)
        for p in self._pathways:
            W, scale = self._pathway_weights(p)
            rows = self._rows(p)
            # Feedforward input is known for the whole chunk: one matmul per pathway.
            pre = None if self._sides[p.side].recurrent else x[:, :, rows] @ W
            drive.append(
                (p.side, rows, W, scale, getattr(self, f"_made_{p.index}"), pre)
            )

        keep = (
            slice(None)
            if self.track_batch_idx is None
            else slice(self.track_batch_idx, self.track_batch_idx + 1)
        )
        spikes, tracked, retained = [], {"v": [], "I": [[] for _ in self._sides]}, []
        v = self.v
        steps = (
            tqdm(range(T), desc="Simulating current-based network", unit="step")
            if self.use_tqdm
            else range(T)
        )
        for t in steps:
            s = SurrGradSpike.apply(v - 1.0, self.surrgrad_scale)
            currents = [alpha * I for alpha, I in zip(alphas, currents)]
            for side, rows, W, scale, made, pre in drive:
                h = s[:, rows] @ W if pre is None else pre[:, t]
                if scale is not None:
                    h = h * scale
                currents[side] = currents[side] + h[:, :, None] * made
            I_total = sum((I * sign).sum(dim=2) for I, sign in zip(currents, signs))
            v = (self.beta_mem * v + I_total) * (1.0 - s.detach())
            spikes.append(s)
            if self.track_gradients:
                for tensor in (v, s, *currents):
                    if tensor.requires_grad:
                        tensor.retain_grad()
                retained.append((v, s, *currents))
            if self.track_variables:
                tracked["v"].append(v[keep].detach())
                for store, I in zip(tracked["I"], currents):
                    store.append(I[keep].detach())

        self.v = v.detach()
        for side, I in zip(self._sides, currents):
            setattr(self, side.state, I.detach())
        if self.track_gradients:
            self._retained = retained
        if not self.track_variables:
            return torch.stack(spikes, dim=1), None
        return torch.stack(spikes, dim=1), {
            "voltages": torch.stack(tracked["v"], dim=1),
            "currents": [torch.stack(I, dim=1) for I in tracked["I"]],
        }

    def _validate_forward(self, x: torch.Tensor) -> None:
        assert isinstance(x, torch.Tensor), "input_spikes must be a Tensor"
        assert x.ndim == 3, (
            f"input_spikes must be 3D (batch, time, n_inputs), got {x.shape}"
        )
        assert x.shape[2] == self.n_inputs, (
            f"input_spikes shape[2]={x.shape[2]}, expected {self.n_inputs}"
        )
        if x.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size {x.shape[0]} != model batch size {self.batch_size}; "
                "call reset_state(batch_size=...) first"
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset_state(self, batch_size: int | None = None) -> None:
        """Voltages and currents to zero; optionally change the batch size."""
        if batch_size is not None:
            self.batch_size = batch_size
        self.register_buffer(
            "v",
            torch.zeros((self.batch_size, self.n_neurons), device=self.device),
            persistent=False,
        )
        for side in self._sides:
            n_types = len(side.synapse_params)
            I = torch.zeros(
                (self.batch_size, self.n_neurons, n_types), device=self.device
            )
            self.register_buffer(side.state, I, persistent=False)

    def get_tracked_gradients(self) -> dict[str, torch.Tensor]:
        """Gradient magnitudes of the tracked per-step ``v``, spikes and currents.

        Call after ``backward()`` on a loss from a forward pass with
        ``track_gradients=True``. Shapes ``(T, B, ...)``; zeros where no gradient.
        """
        if not getattr(self, "_retained", None):
            raise RuntimeError(
                "No gradients tracked. Ensure track_gradients=True during forward() call."
            )
        names = ["grad_v", "grad_s", *(f"grad_{side.state}" for side in self._sides)]
        return {
            name: torch.stack(
                [
                    t.grad.abs() if t.grad is not None else torch.zeros_like(t)
                    for t in tensors
                ]
            )
            for name, tensors in zip(names, zip(*self._retained))
        }


def _long(array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).long()


def _float(array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def validate(dt, cell_params, sides, surrgrad_scale) -> None:
    """Check the construction arguments, with messages naming what is wrong."""
    assert isinstance(dt, (int, float)), "dt must be numeric (int or float)"
    assert dt > 0, f"Simulation timestep dt must be positive, got {dt}"
    assert dt <= 100.0, (
        f"Simulation timestep dt seems unusually large ({dt} ms), please check units"
    )
    assert isinstance(surrgrad_scale, (int, float)) and surrgrad_scale > 0, (
        "surrgrad_scale must be a positive number"
    )
    assert isinstance(cell_params, list) and cell_params, (
        "cell_params must be a non-empty list"
    )
    for cell in cell_params:
        for name in ("cell_id", "name", "tau_mem"):
            assert name in cell, (
                f"Missing required parameter '{name}' in cell_params entry: {cell}"
            )
    ids = sorted(c["cell_id"] for c in cell_params)
    assert ids == list(range(len(ids))), (
        f"cell_ids must be 0-indexed and contiguous, got {ids}"
    )
    for side in sides:
        what = f"synapse_params{side.suffix}"
        assert isinstance(side.synapse_params, list) and side.synapse_params, (
            f"{what} must be a non-empty list"
        )
        for syn in side.synapse_params:
            for name in ("synapse_id", "name", "cell_id", "tau_syn", "type"):
                assert name in syn, (
                    f"Missing required parameter '{name}' in {what} entry: {syn}"
                )
            assert syn["type"] in ("excitatory", "inhibitory"), (
                f"Synapse type must be 'excitatory' or 'inhibitory', got '{syn['type']}' in {what}"
            )
        ids = sorted(s["synapse_id"] for s in side.synapse_params)
        assert ids == list(range(len(ids))), (
            f"synapse_ids in {what} must be 0-indexed and contiguous, got {ids}"
        )
        w = side.weights
        assert w.ndim == 2, f"weights{side.suffix} must be a 2D matrix"
        assert np.all(w >= 0), (
            f"weights{side.suffix} must be non-negative (sign comes from the synapse type)"
        )
        assert w.shape[0] == len(side.cell_type_indices), (
            f"weights{side.suffix} has {w.shape[0]} rows but {len(side.cell_type_indices)} sources"
        )
        if side.recurrent:
            assert w.shape[0] == w.shape[1], "Recurrent weights must be square matrix"
        if side.learn_weights:
            assert side.mask is not None, (
                f"{'connectome_mask' if side.recurrent else 'feedforward_mask'} is required "
                "when optimisable='weights'"
            )
