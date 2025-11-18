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
    required_inputs = ["output_spikes"]

    def __init__(self, target_cv: torch.Tensor, penalty_value: float = 10.0):
        """
        Coefficient of Variation (CV) loss for spike trains.

        Concatenates spike trains across batches for each neuron and computes CV
        on the trial-averaged spike train. CV = std(ISI) / mean(ISI) where ISI = inter-spike intervals.
        This treats all batches as trials of the same neuron.

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

        # Pre-allocate tensor for CVs (one per neuron, not per batch)
        cvs_tensor = torch.zeros(n_neurons, device=output_spikes.device)

        # Compute CV for each neuron using concatenated spike trains across batches
        for n in range(n_neurons):
            # Concatenate spike trains across batches for this neuron
            # Shape: (batch * time,)
            neuron_spikes = output_spikes[:, :, n].reshape(-1)

            # Find spike times in the concatenated spike train
            spike_times = torch.where(neuron_spikes > 0)[0].float()

            if len(spike_times) < 3:
                # Need at least 3 spikes (2 ISIs) to compute meaningful CV, use high penalty value
                cvs_tensor[n] = self.penalty_value
            else:
                # Compute inter-spike intervals
                isi = torch.diff(spike_times)

                # Compute CV
                mean_isi = isi.mean()
                std_isi = isi.std()

                if mean_isi > 0:
                    cvs_tensor[n] = std_isi / mean_isi
                else:
                    cvs_tensor[n] = self.penalty_value

        # Compute L2 loss against target
        loss = F.mse_loss(cvs_tensor, self.target_cv)

        return loss


class FiringRateLoss(nn.Module):
    required_inputs = ["output_spikes"]

    def __init__(self, target_rate: torch.Tensor, dt: float):
        """
        Firing rate loss for spike trains.

        Computes firing rate (Hz) for each neuron across all batches (treating batches as trials)
        and compares to target values using normalized MSE loss. Each neuron's squared error is
        divided by its target rate to give equal relative importance regardless of target firing rate.

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
            torch.Tensor: Mean normalized MSE loss between actual and target firing rates.
        """
        batch, time, n_neurons = output_spikes.shape

        # Compute firing rate across all batches: sum spikes over time and batch, convert to Hz
        # Total time in seconds = batch * time * dt / 1000
        total_time_s = batch * time * self.dt / 1000.0

        # Sum spikes over time and batch dimensions
        spike_counts = output_spikes.sum(dim=(0, 1))  # (n_neurons,)

        # Convert to firing rate in Hz
        firing_rates = spike_counts / total_time_s  # (n_neurons,)

        # Compute squared error for each neuron
        squared_errors = (firing_rates - self.target_rate) ** 2  # (n_neurons,)

        # Normalize by target rate to weight low firing rate neurons appropriately
        normalized_errors = squared_errors / (self.target_rate + 1e-6)

        # Take mean across neurons
        loss = normalized_errors.mean()

        return loss


class SilentNeuronPenalty(nn.Module):
    required_inputs = ["output_spikes"]

    def __init__(self, alpha: float = 1.0, dt: float = 1.0):
        """
        Exponential penalty for silent neurons.

        Applies an exponential penalty to neurons with very low or zero firing rates
        to encourage all neurons to be active. The penalty is: exp(-alpha * firing_rate)

        Args:
            alpha (float): Exponential decay parameter. Higher values create stronger
                          penalties for silent neurons. Default: 1.0.
            dt (float): Simulation time step (ms). Default: 1.0.
        """
        super(SilentNeuronPenalty, self).__init__()
        self.alpha = alpha
        self.dt = dt

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute silent neuron penalty.

        Args:
            output_spikes (torch.Tensor): Output spike trains (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean exponential penalty across neurons.
        """
        batch, time, n_neurons = output_spikes.shape

        # Compute firing rate across all batches: sum spikes over time and batch, convert to Hz
        # Total time in seconds = batch * time * dt / 1000
        total_time_s = batch * time * self.dt / 1000.0

        # Sum spikes over time and batch dimensions
        spike_counts = output_spikes.sum(dim=(0, 1))  # (n_neurons,)

        # Convert to firing rate in Hz
        firing_rates = spike_counts / total_time_s  # (n_neurons,)

        # Compute exponential penalty: exp(-alpha * firing_rate)
        # Silent neurons (firing_rate ~ 0) get penalty ~ 1
        # Active neurons (firing_rate > 0) get penalty < 1
        penalty = torch.exp(-self.alpha * firing_rates)

        # Take mean across neurons
        loss = penalty.mean()

        return loss


class SubthresholdVarianceLoss(nn.Module):
    required_inputs = ["voltages"]

    def __init__(self, v_threshold: torch.Tensor, target_ratio: torch.Tensor):
        """
        Loss to drive membrane potential fluctuations to be subthreshold by a target ratio.

        Encourages voltage fluctuations (std) to be a specific fraction of the distance
        to threshold. This helps ensure neurons operate in a subthreshold regime with
        appropriate variability for spiking.

        Args:
            v_threshold (torch.Tensor): Spike threshold voltage per neuron (n_neurons,) in mV.
            target_ratio (torch.Tensor): Target ratio of std(V) to (V_th - mean(V)) per neuron (n_neurons,).
        """
        super(SubthresholdVarianceLoss, self).__init__()
        self.register_buffer("v_threshold", v_threshold)
        self.register_buffer("target_ratio", target_ratio)

    def forward(self, voltages: torch.Tensor) -> torch.Tensor:
        """
        Compute subthreshold variance loss.

        Args:
            voltages (torch.Tensor): Membrane potentials (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean squared error between actual and target variance ratios.
        """
        # Compute mean and std across time dimension only (per batch, per neuron)
        mean_V = voltages.mean(dim=1)  # (batch, n_neurons)
        std_V = voltages.std(dim=1)  # (batch, n_neurons)

        # Distance from mean voltage to threshold
        distance_to_threshold = self.v_threshold - mean_V

        # Compute ratio of std to distance to threshold
        ratio = std_V / (distance_to_threshold + 1e-8)

        # Loss: actual ratio should match target ratio
        # Average over batch and neurons at the end
        loss = ((ratio - self.target_ratio) ** 2).mean()

        return loss
