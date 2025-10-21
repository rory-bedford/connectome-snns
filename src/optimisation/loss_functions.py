"""
Loss functions for spiking neural network training.

Currently includes Van Rossum distance for spike train comparison.

Example:
    loss_fn = VanRossumLoss(tau=20.0, delta_t=1.0)
    loss = loss_fn(output_spikes, target_spikes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanRossumLoss(nn.Module):
    def __init__(self, tau: float, delta_t: float):
        """
        Van Rossum distance for spike trains.

        Args:
            tau (float): Time constant for exponential kernel (ms).
            delta_t (float): Simulation time step (ms).
        """
        super(VanRossumLoss, self).__init__()
        self.tau = tau
        self.delta_t = delta_t

        # Pre-compute and register kernel as buffer
        kernel_size = int(5 * tau / delta_t)  # 5 time constants
        t = torch.arange(kernel_size, dtype=torch.float32) * delta_t
        kernel = torch.exp(-t / tau)
        kernel = kernel / kernel.sum()  # Normalize

        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Van Rossum distance between spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains (batch, time, n_neurons).
            target_spikes (torch.Tensor): Target spike trains (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean Van Rossum loss across all dimensions.
        """
        # Reshape for conv1d: (batch * n_neurons, 1, time) (makes convolution easier)
        batch, time, n_neurons = output_spikes.shape

        output_flat = output_spikes.permute(0, 2, 1).reshape(-1, 1, time)
        target_flat = target_spikes.permute(0, 2, 1).reshape(-1, 1, time)

        # Convolve with exponential kernel
        output_smooth = F.conv1d(
            output_flat, self.kernel, padding=self.kernel.shape[-1] // 2
        )

        target_smooth = F.conv1d(
            target_flat, self.kernel, padding=self.kernel.shape[-1] // 2
        )

        # Compute squared difference
        diff = (output_smooth - target_smooth) ** 2

        # Compute mean over all dimensions
        return diff.mean()
