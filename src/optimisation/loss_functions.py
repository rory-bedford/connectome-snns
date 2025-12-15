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

    def __init__(self, tau: float, dt: float, window_size: int, device: str = "cpu"):
        """
        Van Rossum distance for spike trains with state continuation across chunks.

        Expects input shape: (batch, time, n_neurons).

        Maintains internal state to handle temporal continuity across chunks. The smoothed
        values from the end of one chunk decay exponentially and contribute to the next chunk.

        Uses causal convolution (only looks backward in time) with kernel size equal to
        the window size.

        Args:
            tau (float): Time constant for exponential kernel (ms).
            dt (float): Simulation time step (ms).
            window_size (int): Number of timesteps in each chunk. Used as kernel size.
            device (str, optional): Device to place kernel on. Defaults to "cpu".
        """
        super(VanRossumLoss, self).__init__()
        self.tau = tau
        self.dt = dt
        self.kernel_size = window_size

        # Pre-compute exponential kernel
        t = torch.arange(window_size, dtype=torch.float32, device=device) * dt
        kernel = torch.exp(-t / tau)

        # Flip kernel for causal convolution: recent spikes get highest weight
        kernel = torch.flip(kernel, dims=[0])

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
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).
                Can be bool or float.
            target_spikes (torch.Tensor): Target spike trains (same shape as output_spikes).
                Can be bool or float.

        Returns:
            torch.Tensor: Mean Van Rossum loss across all dimensions.
        """
        # Convert to float if needed (for convolution operations)
        output_spikes = output_spikes.float()
        target_spikes = target_spikes.float()

        # Handle both 3D (batch, time, neurons) and 4D (batch, patterns, time, neurons)
        if output_spikes.ndim == 4:
            # Flatten batch and patterns: (batch, patterns, time, neurons) -> (batch*patterns, time, neurons)
            batch, patterns, time, n_neurons = output_spikes.shape
            output_spikes = output_spikes.reshape(batch * patterns, time, n_neurons)
            target_spikes = target_spikes.reshape(batch * patterns, time, n_neurons)
            batch = batch * patterns
        else:
            batch, time, n_neurons = output_spikes.shape

        # Reshape for conv1d: (batch * n_neurons, 1, time)
        output_conv = output_spikes.permute(0, 2, 1).reshape(batch * n_neurons, 1, time)
        target_conv = target_spikes.permute(0, 2, 1).reshape(batch * n_neurons, 1, time)

        # Ensure kernel matches input dtype (for mixed precision training)
        kernel = self.kernel.to(dtype=output_conv.dtype)

        # Causal convolution: pad only on the left (look backward in time)
        # Padding of (kernel_size - 1) ensures output has same length as input
        padding = self.kernel_size - 1
        output_smooth = F.conv1d(output_conv, kernel, padding=padding)
        target_smooth = F.conv1d(target_conv, kernel, padding=padding)

        # Remove extra timesteps from right side (causal padding adds to left only)
        output_smooth = output_smooth[:, :, :time]
        target_smooth = target_smooth[:, :, :time]

        # Add decayed previous state if it exists
        if self.prev_output_smooth is not None:
            # Create decay vector for current chunk
            t = (
                torch.arange(
                    time, device=output_spikes.device, dtype=output_smooth.dtype
                )
                * self.dt
            )
            decay = torch.exp(-t / self.tau)  # (time,)

            # Ensure previous state matches current dtype (for mixed precision)
            prev_output = self.prev_output_smooth.to(dtype=output_smooth.dtype)
            prev_target = self.prev_target_smooth.to(dtype=target_smooth.dtype)

            # Add decayed previous state across all timesteps
            # Broadcasting: (batch*n_neurons, 1, 1) * (time,) -> (batch*n_neurons, 1, time)
            output_smooth = output_smooth + prev_output * decay.view(1, 1, -1)
            target_smooth = target_smooth + prev_target * decay.view(1, 1, -1)

        # Compute squared difference
        diff = (output_smooth - target_smooth) ** 2

        # Save final smoothed values for next chunk (detached to prevent gradient flow)
        self.prev_output_smooth = output_smooth[:, :, -1:].detach()
        self.prev_target_smooth = target_smooth[:, :, -1:].detach()

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
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).

        Returns:
            torch.Tensor: Mean L2 loss between actual and target CVs (ignoring neurons with <3 spikes).
        """
        # Handle both 3D and 4D inputs by flattening leading dimensions
        if output_spikes.ndim == 4:
            # Flatten batch and patterns: (batch, patterns, time, neurons) -> (batch*patterns, time, neurons)
            batch, patterns, time, n_neurons = output_spikes.shape
            output_spikes = output_spikes.reshape(batch * patterns, time, n_neurons)
        else:
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

        # Use masked mean to ignore NaN values (silent neurons)
        # This is gradient-safe unlike nanmean
        valid_mask = ~torch.isnan(squared_errors)
        if valid_mask.sum() > 0:
            loss = squared_errors[valid_mask].mean()
        else:
            # All neurons are silent - return zero loss with no gradient
            loss = torch.tensor(
                0.0, device=output_spikes.device, dtype=squared_errors.dtype
            )

        return loss


class FiringRateLoss(nn.Module):
    required_inputs = ["output_spikes"]
    requires_target = False

    def __init__(self, target_rate: torch.Tensor, dt: float, epsilon: float = 1.0):
        """
        Firing rate loss for spike trains.

        Pools spike data over batch dimension, computes firing rates, then compares to
        target values using normalized MSE loss. Each neuron's squared error is divided
        by its target rate (plus epsilon) to give equal relative importance.

        Supports two modes:
        - No patterns: (batch, time, neurons) input with (neurons,) target
          → pools over batch, computes rates per neuron
        - With patterns: (batch, patterns, time, neurons) input with (patterns, neurons) target
          → pools over batch for each pattern separately, computes rates per pattern per neuron

        Args:
            target_rate (torch.Tensor): Target firing rates for each neuron in Hz.
                Shape: (n_neurons,) for no-pattern mode, or (n_patterns, n_neurons) for pattern mode.
            dt (float): Simulation time step (ms).
            epsilon (float): Regularization term added to target_rate in denominator to prevent
                numerical instability. Default 1.0 Hz provides reasonable normalization even for
                near-zero target rates.
        """
        super(FiringRateLoss, self).__init__()
        self.register_buffer("target_rate", target_rate)
        self.dt = dt
        self.epsilon = epsilon

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute firing rate loss for spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).

        Returns:
            torch.Tensor: Mean normalized MSE loss between actual and target firing rates.
        """
        # Sum over time dimension (always at position -2)
        spike_counts = output_spikes.sum(
            dim=-2
        )  # (batch, neurons) or (batch, patterns, neurons)

        # Average over batch dimension (always at position 0)
        mean_spike_counts = spike_counts.mean(
            dim=0
        )  # (neurons,) or (patterns, neurons)

        # Compute time duration in seconds
        time_steps = output_spikes.shape[-2]
        total_time_s = time_steps * self.dt / 1000.0

        # Convert to firing rate in Hz
        firing_rates = (
            mean_spike_counts / total_time_s
        )  # (neurons,) or (patterns, neurons)

        # Validate target shape compatibility
        if firing_rates.shape != self.target_rate.shape:
            raise ValueError(
                f"Firing rate shape {firing_rates.shape} does not match target shape {self.target_rate.shape}. "
                f"Expected output_spikes with shape (batch, time, neurons) for target (neurons,), "
                f"or (batch, patterns, time, neurons) for target (patterns, neurons)."
            )

        # Compute squared error
        squared_errors = (firing_rates - self.target_rate) ** 2

        # Normalize by target rate with epsilon regularization
        normalized_errors = squared_errors / (self.target_rate + self.epsilon)

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

        Expects input shape: (batch, time, n_neurons).

        Args:
            alpha (float or torch.Tensor): Exponential decay parameter(s). Higher values create stronger
                          penalties for silent neurons. Can be scalar or tensor of shape (n_neurons,). Default: 1.0.
            dt (float): Simulation time step (ms). Default: 1.0.
        """
        super(SilentNeuronPenalty, self).__init__()
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = alpha
        self.dt = dt

    def forward(self, output_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute silent neuron penalty.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons).

        Returns:
            torch.Tensor: Mean exponential penalty across all batches and neurons.
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


class RecurrentFeedforwardBalanceLoss(nn.Module):
    """
    Loss to encourage recurrent weights to be larger than feedforward weights.

    Computes the ratio of mean recurrent weights to mean feedforward weights
    and penalizes deviations from a target ratio. Only applies to excitatory
    recurrent connections.

    Args:
        target_ratio (float): Target ratio of recurrent/feedforward mean weights.
            For example, 2.0 means recurrent weights should be 2x larger on average.
        excitatory_cell_type (int): Cell type index for excitatory neurons (default: 0).
    """

    required_inputs = [
        "recurrent_weights",
        "feedforward_weights",
        "cell_type_indices",
        "connectome_mask",
        "feedforward_mask",
    ]
    requires_target = False

    def __init__(self, target_ratio: float, excitatory_cell_type: int):
        """
        Initialize the recurrent-feedforward balance loss.

        Args:
            target_ratio (float): Target ratio of mean(recurrent_weights) / mean(feedforward_weights).
            excitatory_cell_type (int): Cell type index for excitatory neurons (default: 0).
        """
        super(RecurrentFeedforwardBalanceLoss, self).__init__()
        self.target_ratio = target_ratio
        self.excitatory_cell_type = excitatory_cell_type

    def forward(
        self,
        recurrent_weights: torch.Tensor,
        feedforward_weights: torch.Tensor,
        cell_type_indices: torch.Tensor,
        connectome_mask: torch.Tensor,
        feedforward_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute recurrent-feedforward balance loss for excitatory recurrent connections.

        Args:
            recurrent_weights (torch.Tensor): Recurrent weight matrix (n_neurons, n_neurons).
            feedforward_weights (torch.Tensor): Feedforward weight matrix (n_inputs, n_neurons).
            cell_type_indices (torch.Tensor): Cell type indices for recurrent neurons (n_neurons,).
            connectome_mask (torch.Tensor): Binary mask for valid recurrent connections (n_neurons, n_neurons).
            feedforward_mask (torch.Tensor): Binary mask for valid feedforward connections (n_inputs, n_neurons).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Get mask for excitatory neurons
        exc_mask = cell_type_indices == self.excitatory_cell_type

        # Extract excitatory-to-all recurrent weights (excitatory sources) and apply connectome mask
        exc_rec_weights = recurrent_weights[exc_mask, :] * connectome_mask[exc_mask, :]

        # Compute mean of non-zero excitatory recurrent weights
        rec_mean = exc_rec_weights[exc_rec_weights > 0].mean()

        # Apply feedforward mask and compute mean of non-zero feedforward weights
        masked_ff_weights = feedforward_weights * feedforward_mask
        ff_mean = masked_ff_weights[masked_ff_weights > 0].mean()

        # Compute actual ratio
        actual_ratio = rec_mean / (ff_mean + 1e-8)

        # Loss is squared difference from target ratio
        loss = (actual_ratio - self.target_ratio) ** 2

        return loss


class ScalingFactorBalanceLoss(nn.Module):
    """
    Loss to encourage recurrent scaling factors to be larger than feedforward scaling factors.

    Computes the ratio of recurrent excitatory scaling factors to feedforward scaling factors
    and penalizes deviations from a target ratio. This is the scaling factor equivalent of
    RecurrentFeedforwardBalanceLoss.

    Args:
        target_ratio (float): Target ratio of recurrent/feedforward scaling factors.
            For example, 2.0 means recurrent scaling factors should be 2x larger on average.
        excitatory_cell_type (int): Cell type index for excitatory neurons (default: 0).
    """

    required_inputs = ["scaling_factors", "scaling_factors_FF"]
    requires_target = False

    def __init__(self, target_ratio: float, excitatory_cell_type: int = 0):
        """
        Initialize the scaling factor balance loss.

        Args:
            target_ratio (float): Target ratio of mean(recurrent_scaling_factors) / mean(feedforward_scaling_factors).
            excitatory_cell_type (int): Cell type index for excitatory neurons (default: 0).
        """
        super(ScalingFactorBalanceLoss, self).__init__()
        self.target_ratio = target_ratio
        self.excitatory_cell_type = excitatory_cell_type

    def forward(
        self,
        scaling_factors: torch.Tensor,
        scaling_factors_FF: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scaling factor balance loss.

        Args:
            scaling_factors (torch.Tensor): Recurrent scaling factors (n_source_types, n_target_types).
            scaling_factors_FF (torch.Tensor): Feedforward scaling factors (n_source_types, n_target_types).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Get the recurrent scaling factors for excitatory sources to all targets
        # scaling_factors shape: (n_source_types, n_target_types)
        # We want the mean of scaling_factors[excitatory_source, :] (excitatory to all targets)
        rec_mean = scaling_factors[self.excitatory_cell_type, :].mean()

        # Get the feedforward scaling factors - average across all source types
        ff_mean = scaling_factors_FF.mean()

        # Compute actual ratio
        actual_ratio = rec_mean / (ff_mean + 1e-8)

        # Loss is squared difference from target ratio
        loss = (actual_ratio - self.target_ratio) ** 2

        return loss
