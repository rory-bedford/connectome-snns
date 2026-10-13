"""The spike-train losses compute what their docstrings define.

The van Rossum filter is written out here as the literal sum over past inputs, with
the documented chunk carry: the last filtered value of a chunk enters the next one
decayed by ``exp(-t dt / tau)`` for ``t = 0, 1, ...``, with no gradient through it. The library computes the same
thing with batched triangular matrix products.
"""

import numpy as np
import pytest
import torch

from connectome_snns.training_utils.losses import PoissonKLLoss, VanRossumLoss

TAU_RISE, TAU_DECAY, DT, CHUNK = 10.0, 100.0, 1.0, 40
# CPU: the same sums as the definition. CUDA: a recursion, whose rounding differs.
RTOL = {"cpu": 1e-5, "cuda": 1e-4}


def exponential_filter(x, tau, carried):
    """``y[t] = sum_{k <= t} exp(-k dt / tau) x[t - k] + carried * exp(-t dt / tau)``."""
    time = x.shape[1]
    kernel = torch.exp(-torch.arange(time, dtype=x.dtype, device=x.device) * DT / tau)
    y = torch.stack(
        [
            (kernel[: t + 1].flip(0)[None, :, None] * x[:, : t + 1]).sum(1)
            for t in range(time)
        ],
        dim=1,
    )
    return y if carried is None else y + carried * kernel[None, :, None]


def reference_smooth(chunks):
    """Filter a sequence of chunks, carrying state, as the docstring defines."""
    rise = smooth = None
    out = []
    for x in chunks:
        r = exponential_filter(x, TAU_RISE, rise)
        y = exponential_filter(r, TAU_DECAY, smooth)
        rise, smooth = (
            r[:, -1:].detach(),
            y[:, -1:].detach(),
        )  # no gradient across chunks
        out.append(y)
    return out


@pytest.fixture
def trains():
    rng = np.random.default_rng(0)

    def train():
        return torch.from_numpy(
            (rng.random((3, 3 * CHUNK, 7)) < 0.05).astype(np.float32)
        )

    return train(), train()


@pytest.mark.parametrize("loss_cls", [VanRossumLoss, PoissonKLLoss])
def test_loss_matches_definition(trains, loss_cls, device):
    output, target = (x.to(device) for x in trains)
    output.requires_grad_(True)
    loss_fn = loss_cls(TAU_RISE, TAU_DECAY, DT, CHUNK, device=device)
    chunks = [slice(c, c + CHUNK) for c in range(0, output.shape[1], CHUNK)]

    o_ref = reference_smooth([output[:, c] for c in chunks])
    t_ref = reference_smooth([target[:, c] for c in chunks])
    for c, o, t in zip(chunks, o_ref, t_ref):
        loss = loss_fn(output[:, c], target[:, c])
        if loss_cls is VanRossumLoss:
            expected = ((o - t) ** 2).mean()
        else:
            eps, t = loss_fn.eps, t.detach()
            expected = (t * torch.log(t / (o + eps) + eps) - t + o).mean() * DT
        torch.testing.assert_close(loss, expected, rtol=RTOL[device], atol=1e-7)
        (grad,) = torch.autograd.grad(loss, output, retain_graph=True)
        (expected_grad,) = torch.autograd.grad(expected, output, retain_graph=True)
        torch.testing.assert_close(
            grad, expected_grad, rtol=1e-4, atol=1e-4 * float(expected_grad.abs().max())
        )


def test_reset_state_forgets(trains):
    output, target = trains
    loss_fn = VanRossumLoss(TAU_RISE, TAU_DECAY, DT, CHUNK)
    first = loss_fn(output[:, :CHUNK], target[:, :CHUNK])
    loss_fn(output[:, CHUNK : 2 * CHUNK], target[:, CHUNK : 2 * CHUNK])
    loss_fn.reset_state()
    torch.testing.assert_close(loss_fn(output[:, :CHUNK], target[:, :CHUNK]), first)


def test_zero_weight_losses_are_not_computed():
    from connectome_snns.snn_runners.chunks import weighted_losses

    calls = []

    class Counted(torch.nn.Module):
        def forward(self, output_spikes):
            calls.append(1)
            return output_spikes.float().mean()

    chunk = {"spikes": torch.ones(2, 3, 4)}
    losses = weighted_losses(
        {"used": Counted(), "off": Counted(), "unweighted": Counted()},
        {"used": 2.0, "off": 0.0},
        chunk,
        model=None,
    )
    assert len(calls) == 1
    assert sorted(losses) == ["total", "used"]
    assert losses["total"].item() == 2.0
