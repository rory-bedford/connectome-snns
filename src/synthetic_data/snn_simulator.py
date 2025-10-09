"""Defines a straightforward simulator of a network of LIF neurons in discrete time"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class LIFNetwork(nn.Module):
    def __init__(
        self,
        neuron_types: IntArray,
        weights: FloatArray,
        tau_m: float = 10e-3,
        tau_I: float = 5e-3,
        dt: float = 1e-3,
    ):
        """
        Initialize the LIF network.

        Args:
            neuron_types (IntArray): Array of shape (n_neurons,) with +1 (excitatory) or -1 (inhibitory).
            weights (FloatArray): Weight matrix of shape (n_neurons, n_neurons).
            tau_m (float): Membrane time constant in seconds. Defaults to 10e-3.
            tau_I (float): Synaptic current time constant in seconds. Defaults to 5e-3.
            v_th (float): Spike threshold. Defaults to 1.0.
            v_reset (float): Reset potential after spike. Defaults to 0.0.
            dt (float): Simulation time step in seconds. Defaults to 1e-3.
        """
        super(LIFNetwork, self).__init__()

        # Convert numpy arrays to tensors
        neuron_types = torch.from_numpy(neuron_types).long()
        weights = torch.from_numpy(weights).float()

        # Validate inputs
        assert neuron_types.ndim == 1, "neuron_types must be a 1D array"
        assert weights.ndim == 2, "weights must be a 2D array"
        assert weights.shape[0] == weights.shape[1], "weights must be square"
        assert neuron_types.shape[0] == weights.shape[0], (
            "neuron_types and weights dimensions must match"
        )
        assert torch.all((neuron_types == 1) | (neuron_types == -1)), (
            "neuron_types must contain only +1 or -1"
        )
        assert tau_m > 0, "tau_m must be positive"
        assert tau_I > 0, "tau_I must be positive"
        assert dt > 0, "dt must be positive"
        assert dt < tau_m, "dt should be smaller than tau_m for numerical stability"
        assert dt < tau_I, "dt should be smaller than tau_I for numerical stability"

        # Register fixed parameters as buffers (not trainable, saved with model state)
        self.register_buffer("neuron_types", neuron_types)
        self.register_buffer("weights", weights)
        self.register_buffer("tau_m", torch.tensor(tau_m))
        self.register_buffer("tau_I", torch.tensor(tau_I))
        self.register_buffer("dt", torch.tensor(dt))
        self.register_buffer("tau_m", torch.tensor(tau_m))
        self.register_buffer("tau_I", torch.tensor(tau_I))
        self.register_buffer("dt", torch.tensor(dt))

        # Precompute and register decay factors
        self.register_buffer("alpha", torch.exp(-self.dt / self.tau_m))
        self.register_buffer("beta", torch.exp(-self.dt / self.tau_I))

    def initialise_parameters(
        self,
        E_weight: float = 1.0,
        I_weight: float = 1.0,
        E_v_th: float = 1.0,
        I_v_th: float = 1.0,
    ):
        """
        Initialize optimisable parameters.

        Args:
            E_weight (float): Scaling factor for excitatory weights. Defaults to 1.0.
            I_weight (float): Scaling factor for inhibitory weights. Defaults to 1.0.
            E_v_th (float): Spike threshold for excitatory neurons. Defaults to 1.0.
            I_v_th (float): Spike threshold for inhibitory neurons. Defaults to 1.0.
        """
        assert E_weight > 0, "E_weight must be positive"
        assert I_weight > 0, "I_weight must be positive"
        assert E_v_th > 0, "E_v_th must be positive"
        assert I_v_th > 0, "I_v_th must be positive"

        self.E_weight = nn.Parameter(torch.tensor(E_weight, dtype=torch.float32))
        self.I_weight = nn.Parameter(torch.tensor(I_weight, dtype=torch.float32))
        self.E_v_th = nn.Parameter(torch.tensor(E_v_th, dtype=torch.float32))
        self.I_v_th = nn.Parameter(torch.tensor(I_v_th, dtype=torch.float32))

    def forward(
        self,
        inputs: FloatArray,
        n_steps: int,
        initial_v: FloatArray | None = None,
        initial_I: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Simulate the network for a given number of time steps.

        Args:
            inputs (FloatArray): External input current of shape (n_neurons,).
            n_steps (int): Number of time steps to simulate.
            initial_v (FloatArray | None): Initial membrane potentials. Defaults to zeros.
            initial_I (FloatArray | None): Initial synaptic currents. Defaults to zeros.
        """
