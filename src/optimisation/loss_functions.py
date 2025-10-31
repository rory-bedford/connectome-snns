"""
Loss functions for spiking neural network training.

Currently includes Van Rossum distance for spike train comparison, CV loss, and firing rate loss.

Example:
    loss_fn = VanRossumLoss(tau=20.0, dt=1.0)
    loss = loss_fn(output_spikes, target_spikes)

    cv_loss_fn = CVLoss(target_cv=torch.ones(n_neurons))
    loss = cv_loss_fn(output_spikes)

    fr_loss_fn = FiringRateLoss(target_rate=torch.ones(n_neurons) * 10.0, dt=1.0)
    loss = fr_loss_fn(output_spikes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanRossumLoss(nn.Module):
    def __init__(self, tau: float, dt: float):
        """
        Van Rossum distance for spike trains.

        Args:
            tau (float): Time constant for exponential kernel (ms).
            dt (float): Simulation time step (ms).
        """
        super(VanRossumLoss, self).__init__()
        self.tau = tau
        self.dt = dt

        # Pre-compute and register kernel as buffer
        kernel_size = int(5 * tau / dt)  # 5 time constants
        t = torch.arange(kernel_size, dtype=torch.float32) * dt
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


class CVLoss(nn.Module):
    def __init__(self, target_cv: torch.Tensor, penalty_value: float = 10.0):
        """
        Coefficient of Variation (CV) loss for spike trains.

        Computes CV for each neuron and compares to target values using L2 loss.
        CV = std(ISI) / mean(ISI) where ISI = inter-spike intervals.

        Args:
            target_cv (torch.Tensor): Target CV values for each neuron (n_neurons,).
            penalty_value (float): Penalty value for neurons with insufficient spikes. Default: 10.0.
        """
        super(CVLoss, self).__init__()
        self.register_buffer("target_cv", target_cv)
        self.penalty_value = penalty_value

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute CV loss for spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean L2 loss between actual and target CVs.
        """
        batch, time, n_neurons = output_spikes.shape

        # Pre-allocate tensor for CVs
        cvs_tensor = torch.zeros(batch, n_neurons, device=output_spikes.device)

        # Compute CV for each neuron across batch
        for b in range(batch):
            for n in range(n_neurons):
                spike_times = torch.where(output_spikes[b, :, n] > 0)[0].float()

                if len(spike_times) < 3:
                    # Need at least 3 spikes (2 ISIs) to compute meaningful CV, use high penalty value
                    cvs_tensor[b, n] = self.penalty_value
                else:
                    # Compute inter-spike intervals
                    isi = torch.diff(spike_times)

                    # Compute CV
                    mean_isi = isi.mean()
                    std_isi = isi.std()

                    if mean_isi > 0:
                        cvs_tensor[b, n] = std_isi / mean_isi
                    else:
                        cvs_tensor[b, n] = self.penalty_value

        # Compute L2 loss against target
        target_expanded = self.target_cv.unsqueeze(0).expand(batch, -1)
        loss = F.mse_loss(cvs_tensor, target_expanded)

        return loss


class FiringRateLoss(nn.Module):
    def __init__(self, target_rate: torch.Tensor, dt: float):
        """
        Firing rate loss for spike trains.

        Computes firing rate (Hz) for each neuron and compares to target values using L2 loss.

        Args:
            target_rate (torch.Tensor): Target firing rates for each neuron in Hz (n_neurons,).
            dt (float): Simulation time step (ms).
        """
        super(FiringRateLoss, self).__init__()
        self.register_buffer("target_rate", target_rate)
        self.dt = dt

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute firing rate loss for spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean L2 loss between actual and target firing rates.
        """
        batch, time, n_neurons = output_spikes.shape

        # Compute firing rate: sum spikes over time, convert to Hz
        # Total time in seconds = time * dt / 1000
        total_time_s = time * self.dt / 1000.0

        # Sum spikes over time dimension
        spike_counts = output_spikes.sum(dim=1)  # (batch, n_neurons)

        # Convert to firing rate in Hz
        firing_rates = spike_counts / total_time_s  # (batch, n_neurons)

        # Compute L2 loss against target
        target_expanded = self.target_rate.unsqueeze(0).expand(batch, -1)
        loss = F.mse_loss(firing_rates, target_expanded)

        return loss
