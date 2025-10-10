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
        recurrent_weights: FloatArray,
        feedforward_weights: FloatArray | None = None,
        tau_m: float = 10e-3,
        tau_I: float = 5e-3,
        dt: float = 1e-3,
    ):
        """
        Initialize the non-optimisable parameters of the LIF network.

        Args:
            neuron_types (IntArray): Array of shape (n_neurons,) with +1 (excitatory) or -1 (inhibitory).
            recurrent_weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            feedforward_weights (FloatArray | None): Feedforward weight matrix of shape (n_neurons, n_inputs) or None.
            tau_m (float): Membrane time constant in seconds. Defaults to 10e-3.
            tau_I (float): Synaptic current time constant in seconds. Defaults to 5e-3.
            dt (float): Simulation time step in seconds. Defaults to 1e-3.
        """
        super(LIFNetwork, self).__init__()

        # Convert numpy arrays to tensors
        neuron_types = torch.from_numpy(neuron_types).long()
        recurrent_weights = torch.from_numpy(recurrent_weights).float()
        if feedforward_weights is not None:
            feedforward_weights = torch.from_numpy(feedforward_weights).float()

        # Validate inputs
        assert neuron_types.ndim == 1, "neuron_types must be a 1D array"
        assert recurrent_weights.ndim == 2, "recurrent_weights must be a 2D array"
        assert recurrent_weights.shape[0] == recurrent_weights.shape[1], (
            "recurrent_weights must be square"
        )
        assert neuron_types.shape[0] == recurrent_weights.shape[0], (
            "neuron_types and recurrent_weights dimensions must match"
        )
        if feedforward_weights is not None:
            assert feedforward_weights.ndim == 2, (
                "feedforward_weights must be a 2D array"
            )
            assert feedforward_weights.shape[0] == neuron_types.shape[0], (
                "feedforward_weights first dimension must match neuron count"
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
        self.register_buffer("number_neurons", torch.tensor(neuron_types.shape[0]))
        self.register_buffer("recurrent_weights", recurrent_weights)
        if feedforward_weights is not None:
            self.register_buffer("feedforward_weights", feedforward_weights)
        else:
            self.feedforward_weights = None  # Store as None for clarity

        self.register_buffer("tau_m", torch.tensor(tau_m))
        self.register_buffer("tau_I", torch.tensor(tau_I))
        self.register_buffer("dt", torch.tensor(dt))

        # Precompute and register decay factors
        self.register_buffer("alpha", torch.exp(-self.dt / self.tau_m))
        self.register_buffer("beta", torch.exp(-self.dt / self.tau_I))

    @property
    def device(self):
        """Get the device the model is on"""
        return self.recurrent_weights.device

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

        self.E_weight = nn.Parameter(
            torch.tensor(E_weight, dtype=torch.float32, device=self.device)
        )
        self.I_weight = nn.Parameter(
            torch.tensor(I_weight, dtype=torch.float32, device=self.device)
        )
        self.E_v_th = nn.Parameter(
            torch.tensor(E_v_th, dtype=torch.float32, device=self.device)
        )
        self.I_v_th = nn.Parameter(
            torch.tensor(I_v_th, dtype=torch.float32, device=self.device)
        )

    @property
    def scaled_recurrent_weights(self) -> torch.Tensor:
        """Get the recurrent weights scaled by E_weight and I_weight"""
        assert hasattr(self, "E_weight") and hasattr(self, "I_weight"), (
            "Parameters must be initialized first"
        )
        scaling_factors = torch.where(
            self.neuron_types == 1, self.E_weight, self.I_weight
        )
        return self.recurrent_weights * scaling_factors.unsqueeze(0)

    def forward(
        self,
        n_steps: int,
        initial_v: FloatArray | None = None,
        initial_I: FloatArray | None = None,
        inputs: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Changed return type hint
        """
        Simulate the network for a given number of time steps.

        Args:
            n_steps (int): Number of time steps to simulate.
            initial_v (FloatArray | None): Initial membrane potentials of shape (batches, n_neurons). Defaults to zeros.
            initial_I (FloatArray | None): Initial synaptic currents of shape (batches, n_neurons). Defaults to zeros.
            inputs (FloatArray | None): External input spikes of shape (batches, n_steps, n_inputs).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Spike trains, voltages, and currents, all on same device as model
        """
        assert n_steps > 0, "n_steps must be positive"

        # Create default initial conditions if not provided
        if initial_v is None:
            initial_v = np.zeros((1, self.number_neurons.item()), dtype=np.float32)
        if initial_I is None:
            initial_I = np.zeros((1, self.number_neurons.item()), dtype=np.float32)

        # Validate dimensions
        assert initial_v.ndim == 2, "initial_v must be a 2D array"
        assert initial_v.shape[1] == self.number_neurons, (
            "initial_v second dimension must match number of neurons"
        )
        assert initial_I.ndim == 2, "initial_I must be a 2D array"
        assert initial_I.shape[1] == self.number_neurons, (
            "initial_I second dimension must match number of neurons"
        )
        assert initial_v.shape[0] == initial_I.shape[0], (
            "initial_v and initial_I batch sizes must match"
        )

        if inputs is not None:
            assert self.feedforward_weights is not None, (
                "feedforward_weights must be provided if inputs are given"
            )
            assert inputs.ndim == 3, "inputs must be a 3D array"
            assert inputs.shape[0] == initial_v.shape[0], (
                "inputs batch size must match initial conditions"
            )
            assert inputs.shape[1] == n_steps, (
                "inputs second dimension must match n_steps"
            )
            assert inputs.shape[2] == self.feedforward_weights.shape[1], (
                "inputs third dimension must match feedforward_weights input dimension"
            )

        # Convert inputs to tensors and move to device
        initial_v = torch.from_numpy(initial_v).float().to(self.device)
        initial_I = torch.from_numpy(initial_I).float().to(self.device)
        if inputs is not None:
            inputs = torch.from_numpy(inputs).float().to(self.device)

        # Initialize variables
        v = initial_v  # Shape: (batch_size, n_neurons)
        current = initial_I  # Shape: (batch_size, n_neurons)
        s = torch.zeros_like(v)  # Spike tensor, shape: (batch_size, n_neurons)

        # For storage
        spikes = torch.zeros(
            (initial_v.shape[0], n_steps, self.number_neurons.item()),
            device=self.device,
        )
        voltages = torch.zeros(
            (initial_v.shape[0], n_steps, self.number_neurons.item()),
            device=self.device,
        )
        currents = torch.zeros(
            (initial_v.shape[0], n_steps, self.number_neurons.item()),
            device=self.device,
        )

        # First compute current inputs across all time simultaneously
        if inputs is not None:
            feedforward_inputs = torch.einsum(
                "bti,ni->btn", inputs, self.feedforward_weights
            )
        else:
            feedforward_inputs = torch.zeros(
                (initial_v.shape[0], n_steps, self.number_neurons.item()),
                device=self.device,
            )

        # Simulation loop
        for t in range(n_steps):
            # Update synaptic current
            current = (
                self.alpha * current
                + feedforward_inputs[:, t, :]
                + torch.einsum("bi,ij->bj", s, self.scaled_recurrent_weights)
            )

            # Update membrane potential
            v = self.beta * v + current - s

            # Determine spike threshold based on neuron type
            v_th = torch.where(self.neuron_types == 1, self.E_v_th, self.I_v_th)

            # Generate spikes
            spikes_t = (v >= v_th).float()
            spikes[:, t, :] = spikes_t

            # Reset membrane potential of spiking neurons
            v = v * (1 - spikes_t)

            # Store voltages and currents
            voltages[:, t, :] = v
            currents[:, t, :] = current

        return v, current  # Placeholder return
