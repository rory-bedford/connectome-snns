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
    required_inputs = ["output_spikes"]
    requires_target = True

    def __init__(
        self, tau: float, dt: float, kernel_size: int = None, device: str = "cpu"
    ):
        """
        Van Rossum distance for spike trains with state continuation across chunks.

        Works with any leading batch-like dimensions - treats first N-2 dims as independent
        trials, last 2 dims as (time, neurons).

        Maintains internal state to handle temporal continuity across chunks. The smoothed
        values from the end of one chunk decay exponentially and contribute to the next chunk.

        Args:
            tau (float): Time constant for exponential kernel (ms).
            dt (float): Simulation time step (ms).
            kernel_size (int, optional): Number of timesteps in the kernel.
                If None, defaults to 5 * tau / dt (5 time constants).
            device (str, optional): Device to place kernel on. Defaults to "cpu".

        Note:
            If kernel_size is larger than chunk_size, the convolution will still work correctly
            due to padding, and the state continuation ensures proper temporal continuity.
        """
        super(VanRossumLoss, self).__init__()
        self.tau = tau
        self.dt = dt

        # Pre-compute and register kernel as buffer
        if kernel_size is None:
            kernel_size = int(5 * tau / dt)  # 5 time constants

        self.kernel_size = kernel_size
        t = torch.arange(kernel_size, dtype=torch.float32, device=device) * dt
        kernel = torch.exp(-t / tau)
        kernel = kernel / kernel.sum()  # Normalize

        self.register_buffer("kernel", kernel.view(1, 1, -1))

        # Internal state: smoothed values from previous chunk
        # Will be initialized on first forward pass
        self.prev_output_smooth = None
        self.prev_target_smooth = None

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Van Rossum distance between spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (..., time, n_neurons) where ... represents any batch/pattern dims.
                Can be bool or float.
            target_spikes (torch.Tensor): Target spike trains (same shape as output_spikes).
                Can be bool or float.

        Returns:
            torch.Tensor: Mean Van Rossum loss across all dimensions.
        """
        # Convert to float if needed (for convolution operations)
        output_spikes = output_spikes.float()
        target_spikes = target_spikes.float()

        # Get dimensions
        *leading_dims, time, n_neurons = output_spikes.shape
        n_trials = (
            int(torch.prod(torch.tensor(leading_dims)).item()) if leading_dims else 1
        )

        # Flatten leading dimensions: (n_trials, time, n_neurons)
        output_flat = output_spikes.reshape(n_trials, time, n_neurons)
        target_flat = target_spikes.reshape(n_trials, time, n_neurons)

        # Reshape for conv1d: (n_trials * n_neurons, 1, time)
        output_conv = output_flat.permute(0, 2, 1).reshape(
            n_trials * n_neurons, 1, time
        )
        target_conv = target_flat.permute(0, 2, 1).reshape(
            n_trials * n_neurons, 1, time
        )

        # Ensure kernel matches input dtype (for mixed precision training)
        kernel = self.kernel.to(dtype=output_conv.dtype)

        # Convolve with exponential kernel (smoothing from current chunk only)
        output_smooth = F.conv1d(output_conv, kernel, padding=kernel.shape[-1] // 2)

        target_smooth = F.conv1d(target_conv, kernel, padding=kernel.shape[-1] // 2)

        # Get actual output time dimension after convolution (may differ due to padding)
        conv_time = output_smooth.shape[-1]

        # Add decayed previous state if it exists
        if self.prev_output_smooth is not None:
            # Create decay vector for actual convolution output length
            t = (
                torch.arange(
                    conv_time, device=output_spikes.device, dtype=output_smooth.dtype
                )
                * self.dt
            )
            decay = torch.exp(-t / self.tau)  # (conv_time,)

            # Ensure previous state matches current dtype (for mixed precision)
            prev_output = self.prev_output_smooth.to(dtype=output_smooth.dtype)
            prev_target = self.prev_target_smooth.to(dtype=target_smooth.dtype)

            # Add decayed previous state across all timesteps
            # Broadcasting: (n_trials*n_neurons, 1, 1) * (conv_time,) -> (n_trials*n_neurons, 1, conv_time)
            output_smooth = output_smooth + prev_output * decay.view(1, 1, -1)
            target_smooth = target_smooth + prev_target * decay.view(1, 1, -1)

        # Compute squared difference
        diff = (output_smooth - target_smooth) ** 2

        # Save final smoothed values for next chunk (detached to prevent gradient flow)
        self.prev_output_smooth = output_smooth[:, :, -1:].detach()
        self.prev_target_smooth = target_smooth[:, :, -1:].detach()

        # Compute mean over all dimensions
        return diff.mean()

    def reset_state(self):
        """
        Reset internal state to zero.

        Call this at the start of a new sequence or epoch to clear memory
        from previous chunks.
        """
        self.prev_output_smooth = None
        self.prev_target_smooth = None


class CVLoss(nn.Module):
    required_inputs = ["output_spikes"]
    requires_target = False

    def __init__(self, target_cv: torch.Tensor):
        """
        Coefficient of Variation (CV) loss for spike trains.

        Concatenates spike trains across batches for each neuron and computes CV
        on the trial-averaged spike train. CV = std(ISI) / mean(ISI) where ISI = inter-spike intervals.
        This treats all batches as trials of the same neuron.

        Neurons with insufficient spikes (<3) are ignored in the loss computation.

        Args:
            target_cv (torch.Tensor): Target CV values for each neuron (n_neurons,).
        """
        super(CVLoss, self).__init__()
        self.register_buffer("target_cv", target_cv)

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute CV loss for spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean L2 loss between actual and target CVs (ignoring neurons with <3 spikes).
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
                # Need at least 3 spikes (2 ISIs) to compute meaningful CV, set to NaN
                cvs_tensor[n] = float("nan")
            else:
                # Compute inter-spike intervals
                isi = torch.diff(spike_times)

                # Compute CV
                mean_isi = isi.mean()
                std_isi = isi.std()

                if mean_isi > 0:
                    cvs_tensor[n] = std_isi / mean_isi
                else:
                    cvs_tensor[n] = float("nan")

        # Compute squared errors
        squared_errors = (cvs_tensor - self.target_cv) ** 2

        # Use nanmean to ignore NaN values (silent neurons)
        loss = torch.nanmean(squared_errors)

        return loss


class FiringRateLoss(nn.Module):
    required_inputs = ["output_spikes"]
    requires_target = False

    def __init__(self, target_rate: torch.Tensor, dt: float):
        """
        Firing rate loss for spike trains.

        Computes firing rate (Hz) for each trial/pattern independently and compares to
        target values using normalized MSE loss. Each neuron's squared error is divided
        by its target rate to give equal relative importance.

        Works with any leading batch-like dimensions - treats first N-2 dims as independent
        trials, last 2 dims as (time, neurons).

        Args:
            target_rate (torch.Tensor): Target firing rates for each neuron in Hz.
                Shape must be broadcastable to spike shape with time dim removed.
                E.g., (n_neurons,) for any input, or (n_patterns, n_neurons) for 4D input.
            dt (float): Simulation time step (ms).
        """
        super(FiringRateLoss, self).__init__()
        self.register_buffer("target_rate", target_rate)
        self.dt = dt

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute firing rate loss for spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (..., time, n_neurons) where ... represents any batch/pattern dims.
                Each leading dimension is treated as an independent trial.

        Returns:
            torch.Tensor: Mean normalized MSE loss between actual and target firing rates.
        """
        # Sum over time dimension (always second-to-last)
        spike_counts = output_spikes.sum(dim=-2)  # (..., n_neurons)

        # Compute time duration in seconds
        time_steps = output_spikes.shape[-2]
        total_time_s = time_steps * self.dt / 1000.0

        # Convert to firing rate in Hz
        firing_rates = spike_counts / total_time_s  # (..., n_neurons)

        # Compute squared error (broadcasting handles shape differences)
        squared_errors = (firing_rates - self.target_rate) ** 2

        # Normalize by target rate
        normalized_errors = squared_errors / (self.target_rate + 1e-6)

        # Take mean across all dimensions
        loss = normalized_errors.mean()

        return loss


class SilentNeuronPenalty(nn.Module):
    required_inputs = ["output_spikes"]
    requires_target = False

    def __init__(self, alpha: float = 1.0, dt: float = 1.0):
        """
        Exponential penalty for silent neurons.

        Applies an exponential penalty to neurons with very low or zero firing rates
        to encourage all neurons to be active. The penalty is: exp(-alpha * firing_rate).

        Works with any leading batch-like dimensions - treats first N-2 dims as independent
        trials, last 2 dims as (time, neurons).

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
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (..., time, n_neurons) where ... represents any batch/pattern dims.
                Each leading dimension is treated as an independent trial.

        Returns:
            torch.Tensor: Mean exponential penalty across all trials and neurons.
        """
        # Sum over time dimension (always second-to-last)
        spike_counts = output_spikes.sum(dim=-2)  # (..., n_neurons)

        # Compute time duration in seconds
        time_steps = output_spikes.shape[-2]
        total_time_s = time_steps * self.dt / 1000.0

        # Convert to firing rate in Hz
        firing_rates = spike_counts / total_time_s  # (..., n_neurons)

        # Compute exponential penalty: exp(-alpha * firing_rate)
        # Silent neurons (firing_rate ~ 0) get penalty ~ 1
        # Active neurons (firing_rate > 0) get penalty < 1
        penalty = torch.exp(-self.alpha * firing_rates)

        # Take mean across all dimensions
        loss = penalty.mean()

        return loss


class SubthresholdVarianceLoss(nn.Module):
    required_inputs = ["voltages"]
    requires_target = False

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
