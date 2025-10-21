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
        dt: float,
        inputs: FloatArray | None = None,
        initial_v: FloatArray | None = None,
        initial_I: FloatArray | None = None,
        initial_I_FF: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Args:
            n_steps (int): Number of time steps to simulate.
            dt (float): Time step duration in milliseconds.
            inputs (FloatArray | None): External input spikes of shape (batch_size, n_steps, n_inputs).
            initial_v (FloatArray | None): Initial membrane potentials of shape (batch_size, n_neurons). Defaults to resting potentials.
            initial_I (FloatArray | None): Initial synaptic currents of shape (batch_size, n_steps, n_neurons, n_cell_types). Defaults to zeros.
            initial_I_FF (FloatArray | None): Initial feedforward synaptic currents of shape (batch_size, n_steps, n_neurons, n_cell_types_FF). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - all_s: Spike trains of shape (batch_size, n_steps, n_neurons)
                - all_v: Membrane potentials of shape (batch_size, n_steps, n_neurons)
                - all_I: Synaptic currents of shape (batch_size, n_steps, n_neurons, n_cell_types)
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

        # Validate initial currents if provided
        if initial_I is not None:
            assert initial_I.ndim == 4, (
                "initial_I must have 4 dimensions (batch_size, n_steps, n_neurons, n_cell_types)."
            )
            assert initial_I.shape[0] == batch_size, (
                "initial_I batch size must match batch_size."
            )
            assert initial_I.shape[1] == n_steps, (
                "initial_I must have n_steps time steps."
            )
            assert initial_I.shape[2] == self.n_neurons, (
                "initial_I must match n_neurons."
            )
            assert initial_I.shape[3] == len(self.cell_types_FF), (
                "initial_I must match the number of cell types."
            )

        # Validate initial feedforward currents if provided
        if initial_I_FF is not None:
            assert initial_I_FF.ndim == 4, (
                "initial_I_FF must have 4 dimensions (batch_size, n_steps, n_neurons, n_cell_types_FF)."
            )
            assert initial_I_FF.shape[0] == batch_size, (
                "initial_I_FF batch size must match batch_size."
            )
            assert initial_I_FF.shape[1] == n_steps, (
                "initial_I_FF must have n_steps time steps."
            )
            assert initial_I_FF.shape[2] == self.n_neurons, (
                "initial_I_FF must match n_neurons."
            )
            assert initial_I_FF.shape[3] == len(self.cell_types_FF), (
                "initial_I_FF must match the number of feedforward cell types."
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

        # Handle initial synaptic currents
        if initial_I is not None:
            I = torch.as_tensor(initial_I, dtype=torch.float32, device=self.device)
        else:
            I = torch.zeros(
                (batch_size, self.n_neurons, len(self.cell_types)),
                dtype=torch.float32,
                device=self.device,
            )
        if initial_I_FF is not None:
            I_FF = torch.as_tensor(
                initial_I_FF, dtype=torch.float32, device=self.device
            )
        else:
            I_FF = torch.zeros(
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
        all_I = torch.zeros(
            (batch_size, n_steps, self.n_neurons, len(self.cell_types)),
            dtype=torch.float32,
            device=self.device,
        )
        all_I_FF = torch.zeros(
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
        alpha = torch.exp(-dt / self.tau_syn)  # Shape (n_neurons, n_cell_types)
        alpha_FF = torch.exp(
            -dt / self.tau_syn_FF
        )  # Shape (n_neurons, n_cell_types_FF)
        beta = torch.exp(-dt / self.tau_mem)  # Shape (n_neurons,)

        print(f"I.shape: {I.shape} ")
        print(f"I_FF.shape: {I_FF.shape} ")
        print(f"alpha.shape: {alpha.shape} ")
        print(f"alpha_FF.shape: {alpha_FF.shape} ")
        print(f"beta.shape: {beta.shape} ")

        # Run simulation
        for t in tqdm(range(n_steps), desc="Simulating network", unit="step"):
            # Compute total current at each neuron
            I_total = I.sum(dim=-1)  # Shape (batch_size, n_neurons)
            if inputs is not None:
                I_total += I_FF.sum(dim=-1)

            # Update membrane potentials (without reset)
            v = (
                self.U_rest  # Resting potential
                + (v - self.U_rest) * beta  # Leak
                + I_total * self.R * (1 - beta)  # Input current
            )

            # Generate spikes based on threshold - uses surrogate gradient
            s = self.spike_fn(v - self.theta)

            # Reset membrane potentials where spikes occurred
            v = v * (1 - s) + self.U_reset * s

            # Update synaptic currents using self.cell_typed_weights directly
            I = (
                I * alpha  # Decay with synapse time constant
                + torch.einsum(
                    "bi,cij->bcj", s, self.cell_typed_weights
                )  # Sum over recurrent spikes with weights
            )
            if inputs is not None:
                I_FF = (
                    I_FF * alpha_FF  # Decay with feedforward synapse time constant
                    + torch.einsum(
                        "bi,cij->bcj", inputs, self.cell_typed_weights_FF
                    )  # Sum over feedforward spikes with weights
                )

            # Store results
            all_v[:, t, :] = v
            all_I[:, t, :, :] = I
            all_s[:, t, :] = s

        return all_s, all_v, all_I
