"""
Time-varying rate processes for inhomogeneous Poisson spike generation.

This module provides dataset classes that generate temporal firing rate dynamics,
which can be used as input to inhomogeneous Poisson spike generators. Each process
generates firing rates that evolve over time according to different stochastic or
deterministic dynamics.

Rate processes maintain continuous state across iterations, allowing for seamless
chunked generation of long time series.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Tuple


class OrnsteinUhlenbeckRateProcess(Dataset):
    """
    Continuous Ornstein-Uhlenbeck process operating in pattern space.

    Generates smooth, mean-reverting rate trajectories by running independent OU
    processes for each pattern, then combining them as a weighted sum. The process
    maintains continuous state across iterations, yielding chunks of specified size
    from an ongoing trajectory.

    Each pattern undergoes its own OU dynamics (mean-reverting to zero):
        da_i = -(a_i / tau) * dt + sigma_i * sqrt(dt) * dW_i

    where a_i is the activation of pattern i. The final firing rate is computed
    by applying softmax normalization to the activations, then taking a weighted
    sum of the patterns:
        weights_i(t) = softmax(a_i(t) / temperature)
        r(t) = sum_i [weights_i(t) * pattern_i]

    The softmax temperature controls the sharpness of the pattern mixture: lower
    values make the distribution spikier (one pattern dominates), higher values
    make it softer (patterns blend more uniformly).

    Args:
        patterns: Array of firing rate patterns, shape (n_patterns, n_neurons) in Hz.
            Each row is a spatial pattern that will be modulated over time.
        chunk_size: Number of timesteps per chunk/sample.
        dt: Timestep in milliseconds.
        tau: Timescale of mean reversion in milliseconds (shared across all patterns).
        temperature: Softmax temperature for normalizing pattern activations. Higher values
            make the distribution softer (more uniform), lower values sharpen it. Default: 1.0.
        sigma: Noise amplitude for each pattern. Shape (n_patterns,) or scalar. Default: 0.5.
        a_init: Initial activation for each pattern. Shape (n_patterns,) or scalar.
            If None, uses value 1.0 for all patterns. Default: None.
        return_rates: If True, __getitem__ returns tuple (rates, weights)
            for diagnostic/visualization purposes. If False, returns only rates. Default: False.

    Attributes:
        n_neurons: Number of neurons in each pattern.
        n_patterns: Number of patterns.
        chunk_size: Number of timesteps per chunk.
        dt: Timestep in milliseconds.
        activations: Current state of pattern activations, shape (n_patterns,).

    Example:
        >>> from src.network_inputs.odourants import generate_odour_firing_rates
        >>> # Generate 20 odour patterns for 5000 neurons
        >>> patterns = generate_odour_firing_rates(...)  # Shape: (20, 5000)
        >>>
        >>> # Continuous OU process in pattern space with softmax normalization
        >>> rate_process = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=100,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     temperature=1.0,
        ...     sigma=0.5
        ... )
        >>>
        >>> # Each call returns the next chunk from continuous trajectory
        >>> chunk1 = rate_process[0]  # Shape: (100, 5000)
        >>> chunk2 = rate_process[1]  # Shape: (100, 5000) - continues from chunk1
        >>>
        >>> # For diagnostic/visualization purposes, enable return_rates
        >>> rate_process_diag = OrnsteinUhlenbeckRateProcess(
        ...     patterns=patterns,
        ...     chunk_size=100,
        ...     dt=0.1,
        ...     tau=50.0,
        ...     temperature=1.0,
        ...     return_rates=True
        ... )
        >>> rates, weights = rate_process_diag[0]
        >>> # rates: (100, 5000), weights: (100, 20)
    """

    def __init__(
        self,
        patterns: Union[np.ndarray, torch.Tensor],
        chunk_size: int,
        dt: float,
        tau: float,
        temperature: float,
        sigma: Union[float, np.ndarray, torch.Tensor],
        a_init: Union[float, np.ndarray, torch.Tensor, None],
        return_rates: bool,
    ):
        # Convert patterns to tensor (CPU)
        if isinstance(patterns, np.ndarray):
            patterns = torch.from_numpy(patterns).float()
        self.patterns = patterns.cpu().float()  # Shape: (n_patterns, n_neurons)

        self.n_patterns, self.n_neurons = self.patterns.shape
        self.chunk_size = chunk_size
        self.dt = dt
        self.tau = tau
        self.temperature = temperature

        # Convert sigma to tensor with shape (n_patterns,)
        self.sigma = self._to_pattern_tensor(sigma, self.n_patterns)

        # Store return_rates flag for diagnostic outputs
        self.return_rates = return_rates

        # Initialize activations (state of the continuous process)
        if a_init is None:
            self.activations = torch.ones((self.n_patterns,), dtype=torch.float32)
        else:
            self.activations = self._to_pattern_tensor(a_init, self.n_patterns)

        # Precompute OU parameters
        self.drift = -1.0 / tau
        self.diffusion = self.sigma * np.sqrt(dt)  # Shape: (n_patterns,)

    def _to_pattern_tensor(
        self, value: Union[float, np.ndarray, torch.Tensor], n_patterns: int
    ) -> torch.Tensor:
        """Convert parameter to tensor of shape (n_patterns,)."""
        if isinstance(value, (int, float)):
            return torch.full((n_patterns,), float(value), dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()
        elif isinstance(value, torch.Tensor):
            value = value.cpu().float()

        if value.numel() == 1:
            return value.expand(n_patterns)
        elif value.shape[0] != n_patterns:
            raise ValueError(
                f"Parameter shape {value.shape} incompatible with "
                f"n_patterns={n_patterns}"
            )

        return value

    def __len__(self) -> int:
        """Return arbitrary large number for infinite generation."""
        return int(1e9)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate next chunk of OU rate trajectory in pattern space.

        The process continues from its current state, maintaining temporal continuity
        across chunks.

        Args:
            idx: Index (ignored, process continues from current state).

        Returns:
            If return_rates=False:
                Rate trajectory chunk of shape (chunk_size, n_neurons) in Hz.
            If return_rates=True:
                Tuple of (rates, weights) where:
                - rates: shape (chunk_size, n_neurons) in Hz
                - weights: shape (chunk_size, n_patterns) - normalized mixing weights
        """
        # Store activation trajectory for this chunk
        activation_chunk = torch.zeros(
            (self.chunk_size, self.n_patterns), dtype=torch.float32
        )

        # Generate OU trajectories for each pattern
        for t in range(self.chunk_size):
            # Store current state
            activation_chunk[t] = self.activations

            # Update state with OU dynamics (mean-reverting to 0)
            dW = torch.randn(self.n_patterns)
            da = self.drift * self.activations * self.dt + self.diffusion * dW
            self.activations = self.activations + da

            # Clip to ensure non-negative activations
            self.activations = torch.clamp(self.activations, min=0.0)

        # Combine patterns using softmax normalization of activations
        # activation_chunk: (chunk_size, n_patterns)
        # patterns: (n_patterns, n_neurons)
        # Output: (chunk_size, n_neurons)

        # Apply softmax with temperature to normalize activations
        softmax_input = activation_chunk / self.temperature
        normalized_activations = torch.softmax(
            softmax_input, dim=1
        )  # (chunk_size, n_patterns)

        # Weighted sum of patterns
        rates = normalized_activations @ self.patterns  # (chunk_size, n_neurons)

        if self.return_rates:
            return rates, normalized_activations
        else:
            return rates

    def reset(self, a_init: Union[float, np.ndarray, torch.Tensor, None] = None):
        """
        Reset the process state to initial conditions.

        Args:
            a_init: New initial activation. If None, resets to ones.
        """
        if a_init is None:
            self.activations = torch.ones((self.n_patterns,), dtype=torch.float32)
        else:
            self.activations = self._to_pattern_tensor(a_init, self.n_patterns)
