"""The GPU engine's building blocks agree with their plain-PyTorch definitions."""

import pytest
import torch

from connectome_snns.network_simulators.conductance import engine
from connectome_snns.network_simulators.conductance.dynamics import Drive

pytestmark = pytest.mark.gpu


def test_sparse_activations_match_dense():
    """The gather kernel computes ``x[..., rows] @ W`` per pathway, gradients included."""
    torch.manual_seed(0)
    B, T, n_in, N = 3, 20, 300, 70
    x = (torch.rand(B, T, n_in, device="cuda") < 0.02).float().requires_grad_()
    columns = torch.randperm(n_in, device="cuda")
    pathways = [
        columns[:120].sort().values,
        columns[120:250].sort().values,
    ]  # 50 unused
    weights = [
        torch.rand(len(r), N, device="cuda", requires_grad=True) for r in pathways
    ]
    drive = Drive(
        made=torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        scale=torch.ones(2, N, device="cuda"),
        feedforward=list(zip(pathways, weights)),
    )
    probe = torch.randn(2, B, T, N, device="cuda")

    results = []
    for A in (engine._activations(drive, x), drive.feedforward_activations(x)):
        grads = torch.autograd.grad((A * probe).sum(), [x, *weights])
        results.append((A, *grads))
    for sparse, dense in zip(*results):
        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)
