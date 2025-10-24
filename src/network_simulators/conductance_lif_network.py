"""Defines a straightforward simulator of recurrent current-based LIF network"""
# ruff: noqa

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.conductance_lif_io import ConductanceLIFNetwork_IO

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class ConductanceLIFNetwork(ConductanceLIFNetwork_IO):
    """Conductance-based LIF network simulator with connectome-constrained weights."""

    def forward(
        self,
        n_steps: int,
        dt: float,
        inputs: FloatArray | None = None,
        initial_v: FloatArray | None = None,
        initial_g: FloatArray | None = None,
        initial_g_FF: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Args:
            n_steps (int): Number of time steps to simulate.
            dt (float): Time step duration in milliseconds.
            inputs (FloatArray | None): External input spikes of shape (batch_size, n_steps, n_inputs).
            initial_v (FloatArray | None): Initial membrane potentials of shape (batch_size, n_neurons). Defaults to resting potentials.
            initial_g (FloatArray | None): Initial synaptic conductances of shape (batch_size, n_steps, n_neurons, n_cell_types). Defaults to zeros.
            initial_g_FF (FloatArray | None): Initial feedforward synaptic conductances of shape (batch_size, n_steps, n_neurons, n_cell_types_FF). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - all_s: Spike trains of shape (batch_size, n_steps, n_neurons)
                - all_v: Membrane potentials of shape (batch_size, n_steps, n_neurons)
                - all_g: Synaptic conductances of shape (batch_size, n_steps, n_neurons, n_cell_types)
        """

        # Determine batch size
        batch_size = inputs.shape[0] if inputs is not None else 1

        # Validate inputs if provided
        if inputs is not None:
            assert inputs.ndim == 3, (
                "inputs must have 3 dimensions (batch_size, n_steps, n_inputs)."
            )
            assert inputs.shape[0] == batch_size, (
                "inputs batch size must match batch_size."
            )
            assert inputs.shape[1] == n_steps, "inputs must have n_steps time steps."
            assert inputs.shape[2] == self.n_inputs, (
                "inputs must match the number of feedforward inputs."
            )

        # Validate initial conductances if provided
        if initial_g is not None:
            assert initial_g.ndim == 4, (
                "initial_g must have 4 dimensions (batch_size, n_steps, n_neurons, n_cell_types)."
            )
            assert initial_g.shape[0] == batch_size, (
                "initial_g batch size must match batch_size."
            )
            assert initial_g.shape[1] == n_steps, (
                "initial_g must have n_steps time steps."
            )
            assert initial_g.shape[2] == self.n_neurons, (
                "initial_g must match n_neurons."
            )
            assert initial_g.shape[3] == len(self.cell_types_FF), (
                "initial_g must match the number of cell types."
            )

        # Validate initial feedforward conductances if provided
        if initial_g_FF is not None:
            assert initial_g_FF.ndim == 4, (
                "initial_g_FF must have 4 dimensions (batch_size, n_steps, n_neurons, n_cell_types_FF)."
            )
            assert initial_g_FF.shape[0] == batch_size, (
                "initial_g_FF batch size must match batch_size."
            )
            assert initial_g_FF.shape[1] == n_steps, (
                "initial_g_FF must have n_steps time steps."
            )
            assert initial_g_FF.shape[2] == self.n_neurons, (
                "initial_g_FF must match n_neurons."
            )
            assert initial_g_FF.shape[3] == len(self.cell_types_FF), (
                "initial_g_FF must match the number of feedforward cell types."
            )

        # Validate initial membrane potentials if provided
        if initial_v is not None:
            assert initial_v.ndim == 2, (
                "initial_v must have 2 dimensions (batch_size, n_neurons)."
            )
            assert initial_v.shape[1] == self.n_neurons, (
                "initial_v must match n_neurons."
            )
            assert initial_v.shape[0] == batch_size, (
                "initial_v batch size must match batch_size."
            )

        # Convert inputs to torch if provided
        if inputs is not None:
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        # Handle initial membrane potentials
        if initial_v is not None:
            v = torch.as_tensor(
                initial_v, dtype=torch.float32, device=self.device
            ).clone()
        else:
            v = self.U_rest.clone().detach()
            v = v.repeat(batch_size, 1)

        # Handle initial synaptic conductances
        if initial_g is not None:
            g = torch.as_tensor(initial_g, dtype=torch.float32, device=self.device)
        else:
            g = torch.zeros(
                (batch_size, self.n_neurons, len(self.cell_types)),
                dtype=torch.float32,
                device=self.device,
            )
        if initial_g_FF is not None:
            g_FF = torch.as_tensor(
                initial_g_FF, dtype=torch.float32, device=self.device
            )
        else:
            g_FF = torch.zeros(
                (batch_size, self.n_neurons, len(self.cell_types_FF)),
                dtype=torch.float32,
                device=self.device,
            )

        # Create output tensors for spikes and membrane potentials
        all_v = torch.zeros(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )
        all_g = torch.zeros(
            (batch_size, n_steps, self.n_neurons, len(self.cell_types)),
            dtype=torch.float32,
            device=self.device,
        )
        all_g_FF = torch.zeros(
            (batch_size, n_steps, self.n_neurons, len(self.cell_types_FF)),
            dtype=torch.float32,
            device=self.device,
        )
        all_s = torch.zeros(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize decay constants
        alpha = torch.exp(-dt / self.tau_syn).T  # Shape (n_neurons, n_cell_types)
        alpha_FF = torch.exp(
            -dt / self.tau_syn_FF
        ).T  # Shape (n_neurons, n_cell_types_FF)
        beta = torch.exp(-dt / self.tau_mem)  # Shape (n_neurons,)

        # Run simulation
        for t in tqdm(range(n_steps), desc="Simulating network", unit="step"):
            # Compute total conductance at each neuron
            g_total = g.sum(dim=-1)  # Shape (batch_size, n_neurons)
            if inputs is not None:
                g_total += g_FF.sum(dim=-1)

            # Update membrane potentials (without reset)
            v = (
                self.U_rest  # Resting potential
                + (v - self.U_rest) * beta  # Leak
                + g_total * self.R * (1 - beta)  # Input conductance
            )

            # Generate spikes based on threshold - uses surrogate gradient
            s = self.spike_fn(v - self.theta)

            # Reset membrane potentials where spikes occurred
            v = v * (1 - s) + self.U_reset * s

            # Update synaptic conductances using self.cell_typed_weights directly
            g = (
                g * alpha  # Decay with synapse time constant
                + torch.einsum(
                    "bi,cij->bjc", s, self.cell_typed_weights
                )  # Sum over recurrent spikes with weights
            )
            if inputs is not None:
                g_FF = (
                    g_FF * alpha_FF  # Decay with feedforward synapse time constant
                    + torch.einsum(
                        "bi,cij->bjc", inputs[:, t, :], self.cell_typed_weights_FF
                    )  # Sum over feedforward spikes with weights
                )

            # Store results
            all_v[:, t, :] = v
            all_g[:, t, :, :] = g
            all_g_FF[:, t, :, :] = g_FF
            all_s[:, t, :] = s

        return all_s, all_v, all_g, all_g_FF
