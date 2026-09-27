"""Gaussian smoothing for spike trains."""

import torch


def smooth_spike_trains(spikes: torch.Tensor, dt: float, tau: float) -> torch.Tensor:
    """Convolve binary spikes with a normalised Gaussian kernel to estimate rates.

    The kernel has sigma = *tau* (in ms) and is truncated at ±3 sigma.
    The output is normalised so the sum of the kernel equals 1, preserving
    mean spike probability.

    Args:
        spikes: Binary spike tensor of shape ``(batch, time, neurons)``.
        dt: Simulation timestep in ms.
        tau: Gaussian kernel sigma in ms.

    Returns:
        Smoothed rate tensor (same shape), values in [0, 1].
    """
    batch, time, neurons = spikes.shape

    half_width = min(int(3.0 * tau / dt), time // 2)
    if half_width < 1:
        half_width = 1

    t_kernel = (
        torch.arange(
            -half_width, half_width + 1, device=spikes.device, dtype=torch.float32
        )
        * dt
    )
    kernel = torch.exp(-0.5 * (t_kernel / tau) ** 2)
    kernel = kernel / kernel.sum()

    # Conv1d: (batch, channels, time) with groups=neurons
    spikes_float = spikes.float().permute(0, 2, 1)  # (batch, neurons, time)
    spikes_padded = torch.nn.functional.pad(
        spikes_float, (half_width, half_width), mode="reflect"
    )
    kernel_1d = kernel.view(1, 1, -1).expand(neurons, 1, -1)  # (neurons, 1, K)
    rates = torch.nn.functional.conv1d(
        spikes_padded,
        kernel_1d,
        groups=neurons,
    )  # (batch, neurons, time)
    rates = rates.permute(0, 2, 1)  # (batch, time, neurons)
    rates = rates.clamp(0.0, 1.0)

    return rates
