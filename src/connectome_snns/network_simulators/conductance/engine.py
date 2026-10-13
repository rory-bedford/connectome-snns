"""Runs the dynamics: the Triton kernels on CUDA, the PyTorch reference otherwise.

``simulate`` is the single entry point the networks call. On a CUDA device it drives
the kernels of ``kernels.py``:

* A network with no recurrent pathways is integrated a chunk at a time by
  ``scan_forward``, after one matmul per pathway computes the whole chunk's synaptic
  input. Without gradients, long chunks are fed in blocks to bound memory.
* A recurrent network takes two launches per step: ``compact_spikes`` lists who fired,
  ``recurrent_step`` gathers just those weight rows and integrates.
* With gradients, the forward pass saves the voltage, conductances and recurrent
  activations of every step, and ``scan_backward`` is the reverse-time adjoint.

Set ``CONNECTOME_SNNS_BACKEND=torch`` to force the reference implementation.
"""

from __future__ import annotations

import functools
import itertools
import os
import warnings
from dataclasses import dataclass

import torch

from connectome_snns.network_simulators.conductance import dynamics
from connectome_snns.network_simulators.conductance.dynamics import (
    Drive,
    Physiology,
    Run,
)

warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")

try:
    import triton

    from connectome_snns.network_simulators.conductance import kernels
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

BLOCK = 128  # neurons per program (scan kernels)
STEP_BLOCK = 256  # neurons per program (recurrent step)
GATHER_ROWS = 16  # weight rows of fired neurons summed per load
COMPACT_BLOCK = 1024  # neurons per iteration of the spike compaction
FF_TIME_BLOCK = 256  # steps of feedforward input materialised at once (no gradients)


def uses_kernels(x: torch.Tensor, retain) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and retain is None
        and os.environ.get("CONNECTOME_SNNS_BACKEND", "triton") != "torch"
    )


def simulate(
    v, g, x, drive: Drive, p: Physiology, *, probabilities=False, track=False,
    track_batch_idx=None, retain=None,
) -> Run:  # fmt: skip
    """Simulate ``x.shape[1]`` steps; see ``dynamics.simulate`` for the arguments."""
    if not uses_kernels(x, retain):
        return dynamics.simulate(
            v, g, x, drive, p,
            probabilities=probabilities, track=track,
            track_batch_idx=track_batch_idx, retain=retain,
        )  # fmt: skip
    return _simulate_kernels(v, g, x, drive, p, probabilities, track, track_batch_idx)


# =====================================================================
# Kernel launches
# =====================================================================


@dataclass
class _Launch:
    """Shapes, tables and output buffers shared by the launches of one call."""

    B: int
    T: int
    N: int
    S: int
    SP: int
    K_rec: int
    K_ff: int
    made: torch.Tensor  # (K, SP) float
    jump: torch.Tensor  # (2, SP)
    decay: torch.Tensor  # (2, SP)
    E_syn: torch.Tensor  # (SP,)
    g_clip: torch.Tensor  # (SP,)
    p: Physiology
    sample: bool
    probabilities: bool
    noise: torch.Tensor  # (B, T, N), or a placeholder
    track_b: int  # tracked trial, -1 for all, or 0 when not tracking
    tracked: dict[str, torch.Tensor] | None
    dummy: torch.Tensor  # stands in for unused pointer arguments

    @property
    def K(self) -> int:
        return self.K_rec + self.K_ff


def _pad(t: torch.Tensor, SP: int) -> torch.Tensor:
    out = t.new_zeros((*t.shape[:-1], SP))
    out[..., : t.shape[-1]] = t
    return out.contiguous()


def _prepare(v, x, drive, p, probabilities, track, track_batch_idx) -> _Launch:
    B, T = x.shape[:2]
    N = v.shape[1]
    S = p.E_syn.shape[0]
    SP = triton.next_power_of_2(S + 1)  # > S: tracking stores the leak in column S
    dummy = torch.empty(1, device=x.device)
    sample = p.spike_mode == "sample"
    tracked = None
    if track:
        Bt = B if track_batch_idx is None else 1
        tracked = {
            "v": torch.empty((Bt, T, N), device=x.device),
            "I": torch.empty((Bt, T, N, S + 1), device=x.device),
            "g": torch.empty((Bt, T, N, 2, S), device=x.device),
        }
    return _Launch(
        B=B, T=T, N=N, S=S, SP=SP,
        K_rec=len(drive.recurrent), K_ff=len(drive.feedforward),
        made=_pad(drive.made.float(), SP),
        jump=_pad(p.jump, SP), decay=_pad(p.decay, SP),
        E_syn=_pad(p.E_syn, SP), g_clip=_pad(p.g_clip, SP),
        p=p, sample=sample, probabilities=probabilities,
        noise=torch.rand((B, T, N), device=x.device) if sample else dummy,
        track_b=(-1 if track_batch_idx is None else track_batch_idx) if track else 0,
        tracked=tracked, dummy=dummy,
    )  # fmt: skip


def _neurons(p: Physiology):
    return p.theta, p.U_reset, p.E_L, p.C_m, p.beta


def _track_buffers(L: _Launch):
    if L.tracked is None:
        return L.dummy, L.dummy, L.dummy
    return L.tracked["v"], L.tracked["I"], L.tracked["g"]


def _scan(L, A, scale, v, g, spikes, prob, saved, t0=0):
    """``scan_forward`` over the ``A.shape[2]`` steps from ``t0``; updates v, g in place."""
    v_saved, G_saved = saved if saved else (L.dummy, L.dummy)
    TA = A.shape[2]
    kernels.scan_forward[(triton.cdiv(L.B * L.N, BLOCK),)](
        A, scale, L.made, L.jump, L.decay, L.E_syn, L.g_clip,
        *_neurons(L.p),
        v, g, spikes, L.dummy if prob is None else prob, L.noise,
        v_saved, G_saved, *_track_buffers(L),
        L.p.dt, L.p.surr_scale, L.track_b, t0, TA, L.B, L.T, L.N,
        K=L.K, S=L.S, SP=L.SP, BLOCK=BLOCK,
        SAVE=bool(saved), TRACK=L.tracked is not None,
        PROB=prob is not None, SAMPLE=L.sample,
    )  # fmt: skip


def _recurrent(L, A_ff_blocks, W, rows, sizes, scale, v, g, spikes, saved):
    """Per step: ``compact_spikes`` then ``recurrent_step``; updates v, g in place.

    ``A_ff_blocks`` yields ``(t0, A_ff)`` blocks of feedforward activations in order;
    ``rows`` holds the recurrent pathways' source neurons back to back, ``sizes`` how
    many each has.
    """
    row_start = _row_start(sizes, v.device)
    w_start = row_start[:-1]
    v_saved, G_saved, A_saved = saved if saved else (L.dummy, L.dummy, L.dummy)
    W = W.to(L.p.storage_dtype)  # a copy if reduced: gradients stay float32
    capacity = max(sizes) if L.K_rec else 1
    lists = torch.empty(
        (max(L.K_rec, 1), L.B, capacity), dtype=torch.int32, device=v.device
    )
    counts = torch.zeros((max(L.K_rec, 1), L.B), dtype=torch.int32, device=v.device)
    dt, surr = L.p.dt, L.p.surr_scale
    for t0, A_ff in A_ff_blocks:
        T_ff = A_ff.shape[2]
        for i in range(T_ff):
            t = t0 + i
            if L.K_rec:
                kernels.compact_spikes[(L.B, L.K_rec)](
                    v, L.p.theta, L.noise, rows, row_start, lists, counts,
                    surr, t, L.B, L.T, L.N, capacity,
                    BLOCK=COMPACT_BLOCK, SAMPLE=L.sample,
                )  # fmt: skip
            kernels.recurrent_step[(L.B, triton.cdiv(L.N, STEP_BLOCK))](
                A_ff, W, w_start, lists, counts, scale, L.made, L.jump, L.decay,
                L.E_syn, L.g_clip, *_neurons(L.p),
                v, g, spikes, L.noise,
                v_saved, G_saved, A_saved, *_track_buffers(L),
                dt, surr, L.track_b, t, i, L.T, T_ff, L.B, L.N, capacity,
                K_REC=L.K_rec, K_FF=L.K_ff, S=L.S, SP=L.SP,
                BLOCK=STEP_BLOCK, ROWS=GATHER_ROWS,
                SAVE=bool(saved), TRACK=L.tracked is not None, SAMPLE=L.sample,
            )  # fmt: skip


@functools.lru_cache(maxsize=64)
def _row_start(sizes: tuple[int, ...], device) -> torch.Tensor:
    """Where each recurrent pathway's rows begin, plus the end: ``(K + 1,)`` int32.

    Cached, so a forward call copies nothing from the host (a small copy would wait
    behind the trainer's prefetch of the next batch).
    """
    return torch.tensor(
        [0, *itertools.accumulate(sizes)], device=device, dtype=torch.int32
    )


def _backward(L, saved, A, scale, grad_spikes, grad_prob, want_A, want_scale):
    """``scan_backward``: returns ``(grad_A, grad_scale)`` (None where not wanted)."""
    spikes, v_saved, G_saved = saved
    grad_A = torch.empty_like(A) if want_A else L.dummy
    partial = torch.empty((L.K, L.B, L.N), device=A.device) if want_scale else L.dummy
    kernels.scan_backward[(triton.cdiv(L.B * L.N, BLOCK),)](
        grad_spikes.contiguous(),
        L.dummy if grad_prob is None else grad_prob.contiguous(),
        spikes, v_saved, G_saved, A, scale, L.made, L.jump, L.decay,
        L.E_syn, L.g_clip, L.p.theta, L.p.C_m, L.p.beta, grad_A, partial,
        L.p.dt, L.p.surr_scale, L.B, L.T, L.N,
        K=L.K, KP=triton.next_power_of_2(L.K), S=L.S, SP=L.SP, BLOCK=BLOCK,
        WANT_A=want_A, WANT_SCALE=want_scale,
        PROB=grad_prob is not None, SAMPLE=L.sample,
    )  # fmt: skip
    return (grad_A if want_A else None), (partial.sum(dim=1) if want_scale else None)


# =====================================================================
# Autograd
# =====================================================================


class _FeedforwardScan(torch.autograd.Function):
    """Spikes (and probabilities) of a chunk, differentiable in ``A`` and ``scale``."""

    @staticmethod
    def forward(ctx, A, scale, v, g, L: _Launch):
        spikes = torch.empty((L.B, L.T, L.N), device=A.device)
        prob = torch.empty_like(spikes) if L.probabilities else None
        saved = (
            torch.empty_like(spikes),
            torch.empty((L.B, L.T, L.N, L.S), device=A.device, dtype=L.p.storage_dtype),
        )
        _scan(L, A, scale, v, g, spikes, prob, saved)
        ctx.L = L
        ctx.save_for_backward(A, scale, spikes, *saved)
        ctx.mark_non_differentiable(v, g)
        return spikes, prob, v, g

    @staticmethod
    def backward(ctx, grad_spikes, grad_prob, _v, _g):
        A, scale, spikes, v_saved, G_saved = ctx.saved_tensors
        if grad_spikes is None:
            grad_spikes = torch.zeros_like(spikes)
        grad_A, grad_scale = _backward(
            ctx.L, (spikes, v_saved, G_saved), A, scale, grad_spikes, grad_prob,
            want_A=ctx.needs_input_grad[0], want_scale=ctx.needs_input_grad[1],
        )  # fmt: skip
        return grad_A, grad_scale, None, None, None


class _RecurrentRun(torch.autograd.Function):
    """Spikes of a recurrent chunk, differentiable in ``A_ff``, ``scale`` and ``W``."""

    @staticmethod
    def forward(ctx, A_ff, scale, W, v, g, L: _Launch, rows, sizes):
        spikes = torch.empty((L.B, L.T, L.N), device=A_ff.device)
        saved = (
            torch.empty_like(spikes),
            torch.empty(
                (L.B, L.T, L.N, L.S), device=A_ff.device, dtype=L.p.storage_dtype
            ),
            torch.empty((L.K_rec, L.B, L.T, L.N), device=A_ff.device),
        )
        _recurrent(L, [(0, A_ff)], W, rows, sizes, scale, v, g, spikes, saved)
        ctx.L, ctx.rows, ctx.sizes = L, rows, sizes
        ctx.save_for_backward(A_ff, scale, spikes, *saved)
        ctx.mark_non_differentiable(v, g)
        return spikes, v, g

    @staticmethod
    def backward(ctx, grad_spikes, _v, _g):
        A_ff, scale, spikes, v_saved, G_saved, A_rec = ctx.saved_tensors
        L = ctx.L
        want_ff, want_scale, want_W = ctx.needs_input_grad[:3]
        A = torch.cat([A_rec, A_ff]) if L.K_ff else A_rec
        grad_A, grad_scale = _backward(
            L, (spikes, v_saved, G_saved), A, scale, grad_spikes, None,
            want_A=want_ff or want_W, want_scale=want_scale,
        )  # fmt: skip
        grad_W = None
        if want_W:
            # dL/dW_k = sum over (trial, step) of presynaptic spikes x dL/dA_k.
            pieces, starts = [], [0, *itertools.accumulate(ctx.sizes)]
            for k in range(L.K_rec):
                pre = spikes[:, :, ctx.rows[starts[k] : starts[k + 1]]]
                pieces.append(
                    pre.reshape(-1, pre.shape[-1]).T @ grad_A[k].reshape(-1, L.N)
                )
            grad_W = torch.cat(pieces)
        grad_ff = grad_A[L.K_rec :] if want_ff else None
        return grad_ff, grad_scale, grad_W, None, None, None, None, None


class _SparseActivations(torch.autograd.Function):
    """Feedforward activations of sparse input, one cuSPARSE product per pathway.

    Differentiable in the input and the weights; both gradients are dense matmuls
    (they are needed only for learnt weights, or when the input is another layer's
    spikes).
    """

    @staticmethod
    def forward(ctx, x2d, rows, differentiable, *weights):
        ctx.rows, ctx.differentiable = rows, differentiable
        ctx.save_for_backward(x2d, *weights)
        x2d = x2d.detach()
        A = x2d.new_empty((len(rows), x2d.shape[0], weights[0].shape[1]))
        for k, (r, W) in enumerate(zip(rows, weights)):
            torch.mm(x2d[:, r].to_sparse_csr(), W, out=A[k])
        return A

    @staticmethod
    def backward(ctx, grad_A):
        x2d, *weights = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros_like(x2d)
            span = ctx.differentiable or slice(0, x2d.shape[1])
            for k, (r, W) in enumerate(zip(ctx.rows, weights)):
                keep = (r >= span.start) & (r < span.stop)  # only columns that need it
                grad_x[:, r[keep]] = grad_A[k] @ W[keep].T
        grad_W = [
            x2d[:, r].T @ grad_A[k] if ctx.needs_input_grad[3 + k] else None
            for k, r in enumerate(ctx.rows)
        ]
        return grad_x, None, None, *grad_W


SPARSE_INPUT = (
    0.1  # use the sparse path when at most this fraction of inputs is non-zero
)


def _activations(drive: Drive, x: torch.Tensor) -> torch.Tensor:
    """``(K_ff, B, T, N)`` feedforward activations, by sparse products if input is sparse.

    Density is judged from the first step, which is cheap and, for spike trains, typical.
    """
    first = x[:, 0]
    if (
        not drive.feedforward
        or torch.count_nonzero(first) > SPARSE_INPUT * first.numel()
    ):
        return drive.feedforward_activations(x)
    B, T, n_in = x.shape
    rows = [r for r, _ in drive.feedforward]
    weights = [W.float() for _, W in drive.feedforward]
    A = _SparseActivations.apply(
        x.reshape(B * T, n_in), rows, drive.differentiable_inputs, *weights
    )
    return A.view(len(rows), B, T, -1)


# =====================================================================
# Entry point for the kernels
# =====================================================================


def _simulate_kernels(v, g, x, drive, p, probabilities, track, track_batch_idx) -> Run:
    L = _prepare(v, x, drive, p, probabilities, track, track_batch_idx)
    v = v.detach().float().contiguous().clone()
    # Conductances are stored (B, 2, S, N), so each component is contiguous over neurons,
    # in ``p.storage_dtype`` (as are the saved conductances and the recurrent weights).
    g = torch.empty((L.B, 2, L.S, L.N), dtype=p.storage_dtype, device=x.device).copy_(
        g.detach().permute(0, 2, 3, 1)
    )
    scale = drive.scale.float().contiguous()
    weights = [W for _, W in drive.recurrent] + [W for _, W in drive.feedforward]
    grad = torch.is_grad_enabled() and (
        scale.requires_grad or x.requires_grad or any(W.requires_grad for W in weights)
    )

    if L.K_rec == 0:
        if grad:
            A = _activations(drive, x).contiguous()
            spikes, prob, v, g = _FeedforwardScan.apply(A, scale, v, g, L)
        else:
            spikes = torch.empty((L.B, L.T, L.N), device=x.device)
            prob = torch.empty_like(spikes) if probabilities else None
            with torch.no_grad():
                for t0 in range(0, L.T, FF_TIME_BLOCK):
                    A = _activations(drive, x[:, t0 : t0 + FF_TIME_BLOCK])
                    _scan(L, A.contiguous(), scale, v, g, spikes, prob, None, t0=t0)
        return Run(spikes, v, g.permute(0, 3, 1, 2).float(), prob, L.tracked)

    rows = torch.cat([r for r, _ in drive.recurrent]).int().contiguous()
    sizes = tuple(r.numel() for r, _ in drive.recurrent)
    W = torch.cat([W for _, W in drive.recurrent]).float().contiguous()

    if grad:
        A_ff = _activations(drive, x).contiguous()
        spikes, v, g = _RecurrentRun.apply(A_ff, scale, W, v, g, L, rows, sizes)
    else:
        spikes = torch.empty((L.B, L.T, L.N), device=x.device)

        def blocks():
            for t0 in range(0, L.T, FF_TIME_BLOCK):
                yield (
                    t0,
                    _activations(drive, x[:, t0 : t0 + FF_TIME_BLOCK]).contiguous(),
                )

        with torch.no_grad():
            _recurrent(L, blocks(), W, rows, sizes, scale, v, g, spikes, None)
    return Run(spikes, v, g.permute(0, 3, 1, 2).float(), None, L.tracked)
