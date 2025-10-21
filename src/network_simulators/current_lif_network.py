"""Defines a straightforward simulator of recurrent current-based LIF network"""
# ruff: noqa

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.current_lif_io import CurrentLIFNetwork_IO

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class CurrentLIFNetwork(CurrentLIFNetwork_IO):
    """Current-based LIF network simulator with connectome-constrained weights."""

    def forward(
        self,
        n_steps: int,
        delta_t: float,
        inputs: FloatArray | None = None,
        initial_v: FloatArray | None = None,
        initial_I: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Args:
            n_steps (int): Number of time steps to simulate.
            delta_t (float): Time step duration in milliseconds.
            inputs (FloatArray | None): External input spikes of shape (batches, n_steps, n_inputs).
            initial_v (FloatArray | None): Initial membrane potentials of shape (batches, n_neurons). Defaults to resting potentials.
            initial_I (FloatArray | None): Initial excitatory synaptic currents of shape (batches, n_neurons). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - all_s: Spike trains of shape (batches, n_steps, n_neurons)
                - all_v: Membrane potentials of shape (batches, n_steps, n_neurons)
                - all_I: Synaptic currents of shape (batches, n_steps, n_neurons)

        Notes:
            Model equations (discrete-time):

            Membrane potential update:
                V_E[t+1] = U_rest_E + (V_E[t] - U_rest_E) * beta_E + I_total[t] * R_E * (1 - beta_E)
                V_I[t+1] = U_rest_I + (V_I[t] - U_rest_I) * beta_I + I_total[t] * R_I * (1 - beta_I)

            Synaptic current update:
                I_exc[t+1] = I_exc[t] * alpha_E + Σ w_ij * s_j[t]  (for excitatory presynaptic spikes)
                I_inh[t+1] = I_inh[t] * alpha_I + Σ w_ij * s_j[t]  (for inhibitory presynaptic spikes)
                I_total[t] = I_exc[t] + I_inh[t]

            Spike generation and reset:
                s[t] = 1 if V[t] >= theta, else 0
                If s[t] = 1: V[t] = U_reset

            Decay factors:
                alpha_E = exp(-dt / tau_syn_E)
                alpha_I = exp(-dt / tau_syn_I)
                beta_E = exp(-dt / tau_mem_E)
                beta_I = exp(-dt / tau_mem_I)
        """

        # Convert delta_t from ms to seconds and compute decay factors
        dt = torch.tensor(delta_t * 1e-3, dtype=torch.float32, device=self.device)
        alpha = torch.exp(-dt / self.tau_syn)
        print(alpha)
        beta = torch.exp(-dt / self.tau_mem)
        exit()

        # Default initial membrane potentials to resting potential if not provided
        if initial_v is None:
            initial_v = self.U_rest.copy()
        else:
            initial_v = torch.as_tensor(
                initial_v, dtype=torch.float32, device=self.device
            )

        # Default initial currents to zero if not provided
        if initial_I_exc is None:
            initial_I_exc = torch.zeros(
                (1, self.n_neurons), dtype=torch.float32, device=self.device
            )
        else:
            initial_I_exc = torch.as_tensor(
                initial_I_exc, dtype=torch.float32, device=self.device
            )

        if initial_I_inh is None:
            initial_I_inh = torch.zeros(
                (1, self.n_neurons), dtype=torch.float32, device=self.device
            )
        else:
            initial_I_inh = torch.as_tensor(
                initial_I_inh, dtype=torch.float32, device=self.device
            )

        if inputs is not None:
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        batch_size = initial_v.shape[0]

        # Validate dimensions of inputs
        assert n_steps > 0
        assert initial_v.ndim == 2
        assert initial_v.shape[1] == self.n_neurons
        assert initial_I_exc.ndim == 2
        assert initial_I_exc.shape[1] == self.n_neurons
        assert initial_I_exc.shape[0] == batch_size
        assert initial_I_inh.ndim == 2
        assert initial_I_inh.shape[1] == self.n_neurons
        assert initial_I_inh.shape[0] == batch_size

        if inputs is not None:
            assert self.feedforward_weights is not None
            assert inputs.ndim == 3
            assert inputs.shape[0] == batch_size
            assert inputs.shape[1] == n_steps
            assert inputs.shape[2] == self.n_inputs

        # Initialize membrane potentials by neuron type
        v_exc = initial_v[:, self.exc_indices]  # Shape: (batch, n_exc)
        v_inh = initial_v[:, self.inh_indices]  # Shape: (batch, n_inh)

        # Initialize synaptic currents by synapse type
        I_exc = initial_I_exc  # Excitatory synaptic currents, shape: (batch, n_neurons)
        I_inh = initial_I_inh  # Inhibitory synaptic currents, shape: (batch, n_neurons)

        # Initialize spike trains
        s_exc = torch.zeros(
            (batch_size, len(self.exc_indices)), device=self.device
        )  # Shape: (batch, n_exc)
        s_inh = torch.zeros(
            (batch_size, len(self.inh_indices)), device=self.device
        )  # Shape: (batch, n_inh)

        # Preallocate output tensors
        all_v = torch.zeros((batch_size, n_steps, self.n_neurons), device=self.device)
        all_I_exc = torch.zeros(
            (batch_size, n_steps, self.n_neurons), device=self.device
        )
        all_I_inh = torch.zeros(
            (batch_size, n_steps, self.n_neurons), device=self.device
        )
        all_s = torch.zeros((batch_size, n_steps, self.n_neurons), device=self.device)

        # Run simulation
        for t in tqdm(range(n_steps), desc="Simulating network", unit="step"):
            # Compute total current at each neuron
            I_total = I_exc + I_inh
            I_to_exc = I_total[:, self.exc_indices]
            I_to_inh = I_total[:, self.inh_indices]

            # Update membrane potentials (without reset)
            v_exc = (
                self.U_rest_E  # Resting potential
                + (v_exc - self.U_rest_E) * beta_E  # Leak
                + I_to_exc * self.R_E * (1 - beta_E)  # Input current
            )

            v_inh = (
                self.U_rest_I  # Resting potential
                + (v_inh - self.U_rest_I) * beta_I  # Leak
                + I_to_inh * self.R_I * (1 - beta_I)  # Input current
            )

            # Generate spikes based on threshold - uses surrogate gradient
            s_exc = self.spike_fn(v_exc - self.theta_E)
            s_inh = self.spike_fn(v_inh - self.theta_I)

            # Reset neurons that spiked
            v_exc = v_exc * (1 - s_exc) + self.U_reset_E * s_exc
            v_inh = v_inh * (1 - s_inh) + self.U_reset_I * s_inh

            # Update synaptic currents by synapse type (presynaptic neuron type)
            # Excitatory synaptic currents decay with alpha_E
            I_exc = (
                I_exc * alpha_E  # Decay with excitatory synapse time constant
                + s_exc @ self.scaled_recurrent_weights[self.exc_indices, :]  # E→all
            )

            # Inhibitory synaptic currents decay with alpha_I
            I_inh = (
                I_inh * alpha_I  # Decay with inhibitory synapse time constant
                + s_inh @ self.scaled_recurrent_weights[self.inh_indices, :]  # I→all
            )

            # Add feedforward input if present
            if inputs is not None:
                # Assume feedforward inputs are excitatory, so they contribute to I_exc
                I_exc = I_exc + inputs[:, t, :] @ self.scaled_feedforward_weights

            # Store results
            all_v[:, t, self.exc_indices] = v_exc
            all_v[:, t, self.inh_indices] = v_inh
            all_I_exc[:, t, :] = I_exc
            all_I_inh[:, t, :] = I_inh
            all_s[:, t, self.exc_indices] = s_exc
            all_s[:, t, self.inh_indices] = s_inh

        return all_s, all_v, all_I_exc, all_I_inh
