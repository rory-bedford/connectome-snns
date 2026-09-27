"""
Loss functions for spiking neural network training.

Currently includes Van Rossum distance for spike train comparison and firing rate loss.

Example:
    loss_fn = VanRossumLoss(tau_rise=10.0, tau_decay=100.0, dt=1.0, window_size=50)
    loss = loss_fn(output_spikes, target_spikes)

    fr_loss_fn = FiringRateLoss(target_rate=torch.ones(n_neurons) * 10.0, dt=1.0)
    loss = fr_loss_fn(output_spikes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectome_snns.observation_models import CalciumObservationModel


class VanRossumLoss(nn.Module):
    required_inputs = ["output_spikes", "target_spikes"]
    requires_target = True

    def __init__(
        self,
        tau_rise: float,
        tau_decay: float,
        dt: float,
        window_size: int,
        device: str = "cpu",
        debug: bool = False,
    ):
        """
        Van Rossum distance for spike trains with double exponential filter.

        Expects input shape: (batch, time, n_neurons).

        Uses a double exponential filter: first filters with rise time constant,
        then filters the result with decay time constant. This creates a more
        realistic synaptic kernel shape with both rise and decay phases.

        Maintains internal state to handle temporal continuity across chunks.
        Tracks both the intermediate (rise-filtered) and final (doubly-filtered)
        signals.

        Uses causal convolution (only looks backward in time) with kernel size equal to
        the window size.

        Args:
            tau_rise (float): Rise time constant for first exponential filter (ms).
            tau_decay (float): Decay time constant for second exponential filter (ms).
            dt (float): Simulation time step (ms).
            window_size (int): Number of timesteps in each chunk. Used as kernel size.
            device (str, optional): Device to place kernel on. Defaults to "cpu".
            debug (bool, optional): If True, return (loss, output_smooth, target_smooth).
                If False, return only loss. Defaults to False.
        """
        super(VanRossumLoss, self).__init__()
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.dt = dt
        self.kernel_size = window_size
        self.debug = debug

        # Pre-compute exponential kernels for rise and decay
        t = torch.arange(window_size, dtype=torch.float32, device=device) * dt
        kernel_rise = torch.exp(-t / tau_rise)
        kernel_decay = torch.exp(-t / tau_decay)

        # Flip kernels for causal convolution: recent spikes get highest weight
        kernel_rise = torch.flip(kernel_rise, dims=[0])
        kernel_decay = torch.flip(kernel_decay, dims=[0])

        self.register_buffer("kernel_rise", kernel_rise.view(1, 1, -1))
        self.register_buffer("kernel_decay", kernel_decay.view(1, 1, -1))

        # Internal state: smoothed values from previous chunk
        # Track both intermediate (rise-filtered) and final (doubly-filtered) states
        # Will be initialized on first forward pass
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Van Rossum distance between spike trains.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).
                Can be bool or float.
            target_spikes (torch.Tensor): Target spike trains (same shape as output_spikes).
                Can be bool or float.

        Returns:
            torch.Tensor: Mean Van Rossum loss across all dimensions (if debug=False).
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (loss, output_smooth, target_smooth) if debug=True.
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

        # Causal convolution: pad only on the left (look backward in time)
        # Padding of (kernel_size - 1) ensures output has same length as input
        padding = self.kernel_size - 1

        # First filter: rise time constant
        output_rise = F.conv1d(output_conv, self.kernel_rise, padding=padding)
        target_rise = F.conv1d(target_conv, self.kernel_rise, padding=padding)

        # Remove extra timesteps from right side (causal padding adds to left only)
        output_rise = output_rise[:, :, :time]
        target_rise = target_rise[:, :, :time]

        # Precompute time vector for chunk carryover decay (used by both rise and smooth states)
        if self.prev_output_rise is not None or self.prev_output_smooth is not None:
            t = (
                torch.arange(time, device=output_spikes.device, dtype=output_rise.dtype)
                * self.dt
            )

        # Add decayed previous rise state if it exists
        if self.prev_output_rise is not None:
            decay_rise = torch.exp(-t / self.tau_rise)

            output_rise = output_rise + self.prev_output_rise * decay_rise.view(
                1, 1, -1
            )
            target_rise = target_rise + self.prev_target_rise * decay_rise.view(
                1, 1, -1
            )

        # Second filter: decay time constant (applied to rise-filtered signal)
        output_smooth = F.conv1d(output_rise, self.kernel_decay, padding=padding)
        target_smooth = F.conv1d(target_rise, self.kernel_decay, padding=padding)

        # Remove extra timesteps from right side
        output_smooth = output_smooth[:, :, :time]
        target_smooth = target_smooth[:, :, :time]

        # Add decayed previous smooth state if it exists
        if self.prev_output_smooth is not None:
            decay_decay = torch.exp(-t / self.tau_decay)

            output_smooth = output_smooth + self.prev_output_smooth * decay_decay.view(
                1, 1, -1
            )
            target_smooth = target_smooth + self.prev_target_smooth * decay_decay.view(
                1, 1, -1
            )

        # Compute squared difference
        diff = (output_smooth - target_smooth) ** 2

        # Save states for next chunk (detached to prevent gradient flow)
        self.prev_output_rise = output_rise[:, :, -1:].detach()
        self.prev_target_rise = target_rise[:, :, -1:].detach()
        self.prev_output_smooth = output_smooth[:, :, -1:].detach()
        self.prev_target_smooth = target_smooth[:, :, -1:].detach()

        # Return based on debug flag
        if self.debug:
            return diff.mean(), output_smooth, target_smooth
        else:
            return diff.mean()

    def reset_state(self):
        """
        Reset internal state to zero.

        Call this at the start of a new sequence or epoch to clear memory
        from previous chunks.
        """
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None


class VanRossumRateLoss(VanRossumLoss):
    """Van Rossum distance against a smooth firing-rate target (native target_rate).

    Identical double-exponential filter to :class:`VanRossumLoss` — the stochastic
    output spikes are double-filtered exactly as before — but the target is the
    measured **smooth firing rate** (expected spikes per bin, ``rate * dt``) put
    through the *same* filter, rather than a target spike train. Because the filter
    is linear, ``E[filter(spikes)] = filter(E[spikes]) = filter(rate * dt)``, so
    the MSE between the filtered stochastic output and the filtered rate is
    unbiased in expectation — a lower-variance objective than matching a sampled
    target spike train.

    The target is supplied natively as ``SpikeData.target_rate`` (routed by the
    trainer via ``required_inputs``), not through the spike-target slot.

    With ``use_probability=True`` (escape-noise / sample-mode models only) the loss
    filters the network's per-step spike *probability* ``p`` instead of the sampled
    spikes ``s``. Since ``E[s] = p``, this removes the sampling-variance term of the
    objective: filtering ``s`` makes the expected loss ``bias² + r·E_k``, whose
    minimum is silence whenever the target rate ``r < E_k/K²``; filtering ``p`` makes
    it pure ``bias²`` so the minimum sits at the target rate. The trainer supplies
    ``p`` via the ``output_probabilities`` snapshot key.
    """

    requires_target = False

    def __init__(self, *args, use_probability: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_probability = use_probability
        # Instance-level so the two modes never collide across loss instances.
        self.required_inputs = (
            ["output_probabilities", "target_rate"]
            if use_probability
            else ["output_spikes", "target_rate"]
        )

    def forward(
        self,
        target_rate: torch.Tensor,
        output_spikes: torch.Tensor | None = None,
        output_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Double-filter the output signal and the smooth rate, MSE between them.

        Args:
            target_rate: Smooth target rate as expected spikes per bin
                (``rate * dt``), shape ``(batch, time, n_cells)``.
            output_spikes: Sampled network spikes (default signal).
            output_probabilities: Per-step spike probability ``p``; used instead of
                ``output_spikes`` when ``use_probability=True``.
        """
        signal = (
            output_probabilities if output_probabilities is not None else output_spikes
        )
        return super().forward(signal, target_rate)


class FiringRateLoss(nn.Module):
    required_inputs = ["output_spikes"]

    def __init__(
        self,
        dt: float,
        target_rate: torch.Tensor | None = None,
        epsilon: float = 1.0,
    ):
        """
        Firing rate loss for spike trains.

        Computes per-chunk firing rates for the student, then compares to target
        values using normalized MSE loss. Each neuron's squared error is divided
        by its target rate (plus epsilon) to give equal relative importance.
        The loss is averaged over both trials and neurons.

        Two modes:
        - Fixed target: provide target_rate at init. Uses precomputed rates that
          stay constant across chunks. Shape can be (n_neurons,) for a single
          global target, or (batch, n_neurons) for per-trial targets.
        - Per-chunk target: omit target_rate. Computes target rates from
          target_spikes each forward call (requires_target=True).

        Use ``from_spike_data`` to precompute per-trial, per-neuron target rates
        from a full zarr dataset (averaged across the entire trial).

        Args:
            dt (float): Simulation time step (ms).
            target_rate (torch.Tensor | None): Target firing rates in Hz.
                Shape: (n_neurons,), (batch, n_neurons), or (batch, patterns, n_neurons).
                If None, target rates are computed from target_spikes each forward call.
            epsilon (float): Regularization term added to target_rate in denominator to prevent
                numerical instability. Default 1.0 Hz provides reasonable normalization even for
                near-zero target rates.
        """
        super(FiringRateLoss, self).__init__()
        self.dt = dt
        self.epsilon = epsilon

        if target_rate is not None:
            self.register_buffer("target_rate", target_rate)
            self.requires_target = False
        else:
            self.target_rate = None
            self.requires_target = True

    @classmethod
    def from_spike_data(
        cls,
        spike_data,
        dt: float,
        epsilon: float = 1.0,
    ) -> "FiringRateLoss":
        """Create a FiringRateLoss with per-trial, per-neuron target rates.

        Computes target firing rates averaged over the entire trial duration,
        giving a stable per-trial, per-neuron target that doesn't fluctuate
        chunk-to-chunk.

        Args:
            spike_data: Target spike data, either a numpy/zarr array of shape
                (batch, total_time, n_neurons) or a path to a zarr group
                containing an 'output_spikes' dataset.
            dt (float): Simulation time step (ms).
            epsilon (float): Regularization epsilon. Default 1.0.

        Returns:
            FiringRateLoss with target_rate shape (batch, n_neurons).
        """
        import numpy as np
        from pathlib import Path

        if isinstance(spike_data, (str, Path)):
            import zarr

            root = zarr.open_group(str(spike_data), mode="r")
            spike_data = root["output_spikes"]

        # spike_data shape: (batch, total_time, n_neurons)
        # Sum over time axis, compute rate in Hz
        spike_counts = np.array(spike_data).sum(axis=1)  # (batch, n_neurons)
        total_time_s = spike_data.shape[1] * dt / 1000.0
        target_rate = torch.from_numpy((spike_counts / total_time_s).astype(np.float32))

        return cls(dt=dt, target_rate=target_rate, epsilon=epsilon)

    def _compute_rates(self, spikes: torch.Tensor) -> torch.Tensor:
        """Compute firing rates in Hz from spike tensor.

        Returns per-trial rates: (batch, neurons) or (batch, patterns, neurons).
        """
        spike_counts = spikes.float().sum(dim=-2)
        time_steps = spikes.shape[-2]
        total_time_s = time_steps * self.dt / 1000.0
        return spike_counts / total_time_s

    def forward(
        self,
        output_spikes: torch.Tensor,
        target_spikes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute firing rate loss for spike trains.

        Computes per-trial, per-neuron normalized squared error between student
        and target firing rates, then averages over all trials and neurons.

        Args:
            output_spikes (torch.Tensor): Output spike trains.
                Shape: (batch, time, n_neurons) or (batch, patterns, time, n_neurons).
            target_spikes (torch.Tensor | None): Target spike trains (same shape as
                output_spikes). Required when target_rate was not provided at init.

        Returns:
            torch.Tensor: Mean normalized MSE loss between actual and target firing rates.
        """
        firing_rates = self._compute_rates(output_spikes)

        if self.target_rate is not None:
            target_rates = self.target_rate.to(firing_rates.device)
        else:
            target_rates = self._compute_rates(target_spikes)

        # Compute per-trial per-neuron squared error normalized by target rate
        squared_errors = (firing_rates - target_rates) ** 2
        normalized_errors = squared_errors / (target_rates + self.epsilon)

        return normalized_errors.mean()


class SilentNeuronLoss(nn.Module):
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
        super(SilentNeuronLoss, self).__init__()
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


class AsymmetricSilencePenalty(nn.Module):
    """Penalizes under-firing output neurons relative to the target rate.

    For each (batch, neuron) pair where the target fires at least once,
    computes penalty = exp(-alpha * output_count / target_count). The ratio
    is relative to the target, so the penalty adapts to each neuron's
    expected activity level. Chunks where the target is silent are ignored.

    This creates an asymmetric pressure: under-firing is penalized harshly,
    while over-firing is handled by other losses (van Rossum, FiringRateLoss).
    """

    required_inputs = ["output_spikes"]
    requires_target = True

    def __init__(self, alpha: float = 5.0):
        """
        Args:
            alpha (float): Controls how fast the penalty decays as output
                approaches the target rate. penalty = exp(-alpha * ratio)
                where ratio = output_count / target_count.
                At ratio=0 (silent): penalty=1.
                At ratio=1 (matching target): penalty=exp(-alpha).
                With alpha=5: ratio=0.5 -> penalty=0.08, ratio=1.0 -> penalty=0.007.
                Default: 5.0.
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            output_spikes: Shape (batch, time, neurons) or (batch, patterns, time, neurons).
            target_spikes: Same shape as output_spikes.

        Returns:
            Scalar penalty averaged over (batch, neuron) pairs where target is active.
        """
        # Sum over time dimension -> spike counts per (batch, ..., neuron)
        output_counts = output_spikes.float().sum(dim=-2)
        target_counts = target_spikes.float().sum(dim=-2)

        # Only penalise where target actually fires in this chunk
        target_active = target_counts > 0

        if not target_active.any():
            return torch.tensor(
                0.0, device=output_spikes.device, dtype=output_spikes.dtype
            )

        # Ratio of output to target spike counts (0 = silent, 1 = matching)
        ratio = output_counts[target_active] / target_counts[target_active]

        penalty = torch.exp(-self.alpha * ratio)
        return penalty.mean()


class BernoulliNLLLoss(nn.Module):
    """Pointwise Bernoulli negative log-likelihood loss.

    Computes binary cross-entropy between the model's per-timestep firing
    probabilities (from the escape-noise sigmoid) and the target spike train::

        L = -mean(target * log(p) + (1 - target) * log(1 - p))

    This operates on **raw probabilities at each timestep** — no temporal
    filtering is applied.  Compare with :class:`PoissonKLLoss`, which first
    smooths spike trains with a double-exponential kernel before computing a
    KL divergence.

    Requires the model to output ``output_probabilities`` (the sigmoid
    values *before* Bernoulli sampling).
    """

    requires_target = True
    required_inputs = ["output_probabilities", "target_spikes"]

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self, output_probabilities: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor:
        prob = torch.clamp(output_probabilities, self.eps, 1.0 - self.eps)
        target_spikes = target_spikes.float()
        nll = -(
            target_spikes * torch.log(prob) + (1 - target_spikes) * torch.log(1 - prob)
        )
        return nll.mean()


class PoissonKLLoss(nn.Module):
    """Poisson KL divergence loss with double-exponential temporal filtering.

    Drop-in replacement for :class:`VanRossumLoss`.  Both losses smooth spike
    trains with the same double-exponential kernel (rise + decay time
    constants), but this loss computes Poisson KL divergence between the
    filtered teacher and student rates instead of squared error::

        KL = λ_T * log(λ_T / λ_S) - λ_T + λ_S

    The KL formulation penalises under-firing proportionally to the teacher
    rate, eliminating the systematic bias toward silence that squared-error
    losses exhibit.  Maintains internal state across chunks for seamless
    streaming during chunked training.

    Compare with :class:`BernoulliNLLLoss`, which operates on raw per-timestep
    probabilities without temporal filtering.
    """

    required_inputs = ["output_spikes", "target_spikes"]
    requires_target = True

    def __init__(
        self,
        tau_rise: float,
        tau_decay: float,
        dt: float,
        window_size: int,
        device: str = "cpu",
        debug: bool = False,
        eps: float = 1e-7,
    ):
        super(PoissonKLLoss, self).__init__()
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.dt = dt
        self.kernel_size = window_size
        self.debug = debug
        self.eps = eps

        # Pre-compute exponential kernels for rise and decay
        t = torch.arange(window_size, dtype=torch.float32, device=device) * dt
        kernel_rise = torch.exp(-t / tau_rise)
        kernel_decay = torch.exp(-t / tau_decay)

        # Flip kernels for causal convolution: recent spikes get highest weight
        kernel_rise = torch.flip(kernel_rise, dims=[0])
        kernel_decay = torch.flip(kernel_decay, dims=[0])

        self.register_buffer("kernel_rise", kernel_rise.view(1, 1, -1))
        self.register_buffer("kernel_decay", kernel_decay.view(1, 1, -1))

        # Internal state for chunk continuity
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None

    def forward(
        self, output_spikes: torch.Tensor, target_spikes: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert to float if needed
        output_spikes = output_spikes.float()
        target_spikes = target_spikes.float()

        # Handle both 3D (batch, time, neurons) and 4D (batch, patterns, time, neurons)
        if output_spikes.ndim == 4:
            batch, patterns, time, n_neurons = output_spikes.shape
            output_spikes = output_spikes.reshape(batch * patterns, time, n_neurons)
            target_spikes = target_spikes.reshape(batch * patterns, time, n_neurons)
            batch = batch * patterns
        else:
            batch, time, n_neurons = output_spikes.shape

        # Reshape for conv1d: (batch * n_neurons, 1, time)
        output_conv = output_spikes.permute(0, 2, 1).reshape(batch * n_neurons, 1, time)
        target_conv = target_spikes.permute(0, 2, 1).reshape(batch * n_neurons, 1, time)

        padding = self.kernel_size - 1

        # First filter: rise time constant
        output_rise = F.conv1d(output_conv, self.kernel_rise, padding=padding)[
            :, :, :time
        ]
        target_rise = F.conv1d(target_conv, self.kernel_rise, padding=padding)[
            :, :, :time
        ]

        # Precompute time vector for chunk carryover decay (used by both rise and smooth states)
        if self.prev_output_rise is not None or self.prev_output_smooth is not None:
            t = (
                torch.arange(time, device=output_spikes.device, dtype=output_rise.dtype)
                * self.dt
            )

        # Add decayed previous rise state
        if self.prev_output_rise is not None:
            decay_rise = torch.exp(-t / self.tau_rise)

            output_rise = output_rise + self.prev_output_rise * decay_rise.view(
                1, 1, -1
            )
            target_rise = target_rise + self.prev_target_rise * decay_rise.view(
                1, 1, -1
            )

        # Second filter: decay time constant
        output_smooth = F.conv1d(output_rise, self.kernel_decay, padding=padding)[
            :, :, :time
        ]
        target_smooth = F.conv1d(target_rise, self.kernel_decay, padding=padding)[
            :, :, :time
        ]

        # Add decayed previous smooth state
        if self.prev_output_smooth is not None:
            decay_decay = torch.exp(-t / self.tau_decay)

            output_smooth = output_smooth + self.prev_output_smooth * decay_decay.view(
                1, 1, -1
            )
            target_smooth = target_smooth + self.prev_target_smooth * decay_decay.view(
                1, 1, -1
            )

        # Poisson KL divergence: λ_T * log(λ_T / (λ_S + ε)) - λ_T + λ_S
        eps = self.eps
        lambda_T = target_smooth.detach()
        lambda_S = output_smooth
        kl = (
            lambda_T * torch.log(lambda_T / (lambda_S + eps) + eps)
            - lambda_T
            + lambda_S
        )

        # Save states for next chunk (detached)
        self.prev_output_rise = output_rise[:, :, -1:].detach()
        self.prev_target_rise = target_rise[:, :, -1:].detach()
        self.prev_output_smooth = output_smooth[:, :, -1:].detach()
        self.prev_target_smooth = target_smooth[:, :, -1:].detach()

        loss = kl.mean() * self.dt

        if self.debug:
            return loss, output_smooth, target_smooth
        else:
            return loss

    def reset_state(self):
        self.prev_output_rise = None
        self.prev_target_rise = None
        self.prev_output_smooth = None
        self.prev_target_smooth = None


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


class MomentMatchLoss(nn.Module):
    """Penalise differences in mean and variance between two tensors.

    Generic and stateless: takes any two tensors, returns a single scalar
    ``lambda_mean * (mean(x) - mean(y))**2 + lambda_var * (var(x) - var(y))**2``.
    Moments are pooled over all elements. Callers handle any slicing
    (e.g. by cell type) before calling.
    """

    def __init__(self, lambda_mean: float = 1.0, lambda_var: float = 1.0):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean_err = (x.mean() - y.mean()).pow(2)
        var_err = (x.var(unbiased=False) - y.var(unbiased=False)).pow(2)
        return self.lambda_mean * mean_err + self.lambda_var * var_err


class CalciumMSELoss(nn.Module):
    """MSE between a rendered dF/F prediction and a measured dF/F target.

    Runs the network's spikes through a :class:`CalciumObservationModel` to get a
    predicted dF/F trace at the simulation timestep, frame-averages it down to the
    measured imaging frame rate, and compares it to the measured dF/F with mean
    squared error — i.e. the network is fit in *observable* space, with no
    spike-deconvolution of the target.

    The target is a **first-class calcium target**, supplied by the dataloader as
    ``SpikeData.target_calcium`` and routed by the trainer via ``required_inputs``
    (no overloading of the spike-target slot).

    Resolution: ``output_spikes`` is ``(batch, time, n_cells)`` at the simulation
    dt; ``target_calcium`` is ``(batch, n_frames, n_cells)`` on the (coarser)
    imaging grid, where ``n_frames`` divides ``time``. The rendered prediction is
    averaged within each frame (matching how an imaging frame integrates the
    signal) before the MSE. The simulation steps per frame are inferred from the
    two lengths, so the loss adapts to any dt / frame-rate combination. NaN
    entries in the target (invalid frames) are masked out of the MSE.

    The calcium model is stateful (it carries latent calcium across chunks), so
    this loss is too: :meth:`reset_state` forwards to the model and must be
    called at the start of each sequence. The trainer does this automatically for
    any loss exposing ``reset_state``.

    Args:
        calcium_model: A configured :class:`CalciumObservationModel` carrying the
            per-cell gain / baseline / kinetics. Owned as a submodule so it moves
            with ``.to(device)`` and its state resets with this loss.
    """

    required_inputs = ["output_spikes", "target_calcium"]
    requires_target = False

    def __init__(self, calcium_model: CalciumObservationModel):
        super().__init__()
        self.calcium_model = calcium_model
        # The exact frame-averaged prediction from the last forward, cached
        # (detached) so the trainer/plotter can show the real computed dF/F — with
        # the correct carried calcium state — instead of cold-re-rendering it.
        self.last_prediction = None

    def forward(
        self, output_spikes: torch.Tensor, target_calcium: torch.Tensor
    ) -> torch.Tensor:
        """Render spikes to dF/F, frame-average to the target grid, masked MSE.

        Args:
            output_spikes: Network spikes, shape ``(batch, time, n_cells)``.
            target_calcium: Measured dF/F, shape ``(batch, n_frames, n_cells)``
                with ``n_frames`` dividing ``time``. NaN marks invalid entries.
        """
        pred_dff = self.calcium_model(output_spikes)  # (batch, time, n_cells)
        batch, n_time, n_cells = pred_dff.shape
        n_frames = target_calcium.shape[1]
        if n_time % n_frames != 0:
            raise ValueError(
                f"prediction length {n_time} is not divisible by the number of "
                f"target frames {n_frames}; cannot frame-average to the imaging grid."
            )
        samples_per_frame = n_time // n_frames
        if samples_per_frame > 1:  # frame-average sim dt -> imaging grid
            pred_dff = pred_dff.reshape(
                batch, n_frames, samples_per_frame, n_cells
            ).mean(dim=2)

        # Cache the exact prediction (on the imaging grid) for plotting.
        self.last_prediction = pred_dff.detach()

        target = target_calcium.float()
        mask = torch.isfinite(target).float()  # invalid frames -> 0 weight
        diff = (pred_dff - torch.nan_to_num(target)) * mask
        return (diff**2).sum() / mask.sum().clamp(min=1.0)

    def reset_state(self):
        """Reset the underlying calcium model's carried latent state."""
        self.calcium_model.reset_state()
