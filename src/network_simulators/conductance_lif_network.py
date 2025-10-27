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
            initial_g (FloatArray | None): Initial synaptic conductances of shape (batch_size, n_neurons, 2, n_cell_types). Defaults to zeros.
            initial_g_FF (FloatArray | None): Initial feedforward synaptic conductances of shape (batch_size, n_neurons, 2, n_cell_types_FF). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - all_s: Spike trains of shape (batch_size, n_steps, n_neurons)
                - all_v: Membrane potentials of shape (batch_size, n_steps, n_neurons)
                - all_g: Synaptic conductances of shape (batch_size, n_steps, n_neurons, n_cell_types)
        """

        # ===============
        # Validate inputs
        # ===============

        self._validate(n_steps, dt, inputs, initial_v, initial_g, initial_g)

        # ==========================
        # Initialize state variables
        # ==========================

        batch_size = inputs.shape[0] if inputs is not None else 1

        # Input spike trains (n_steps, n_inputs)
        if inputs is not None:
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        # Membrane potentials (batch_size, n_neurons)
        if initial_v is not None:
            v = torch.as_tensor(
                initial_v, dtype=torch.float32, device=self.device
            ).clone()
        else:
            v = self.U_rest.clone().detach()
            v = v.repeat(batch_size, 1)

        # Synaptic conductances (batch_size, n_neurons, 2, n_synapse_types)
        if initial_g is not None:
            g = torch.as_tensor(initial_g, dtype=torch.float32, device=self.device)
        else:
            g = torch.zeros(
                (batch_size, self.n_neurons, 2, self.n_synapse_types),
                dtype=torch.float32,
                device=self.device,
            )

        # Feedforward synaptic conductances (batch_size, n_neurons, 2, n_synapse_types_FF)
        if initial_g_FF is not None:
            g_FF = torch.as_tensor(
                initial_g_FF, dtype=torch.float32, device=self.device
            )
        else:
            g_FF = torch.zeros(
                (batch_size, self.n_neurons, 2, self.n_synapse_types_FF),
                dtype=torch.float32,
                device=self.device,
            )

        # Membrane potential storage (batch_size, n_steps, n_neurons)
        all_v = torch.zeros(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )

        # Synaptic input current storage (batch_size, n_steps, n_neurons, n_synapse_types)
        all_I = torch.zeros(
            (batch_size, n_steps, self.n_neurons, self.n_synapse_types),
            dtype=torch.float32,
            device=self.device,
        )

        # Feedforward synaptic input current storage (batch_size, n_steps, n_neurons, n_synapse_types_FF)
        all_I_FF = torch.zeros(
            (batch_size, n_steps, self.n_neurons, self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )

        # Spike train storage (batch_size, n_steps, n_neurons)
        all_s = torch.zeros(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )

        # Precompute decay factors
        tau_syn = torch.stack(
            self.tau_syn_rise, self.tau_syn_decay, dim=0
        )  # Shape (2, n_synapse_types)
        alpha = torch.exp(-dt / tau_syn)  # Shape (2, n_synapse_types)

        if inputs is not None:
            tau_syn_FF = torch.stack(
                self.tau_syn_rise_FF, self.tau_syn_decay_FF, dim=0
            )  # Shape (2, n_synapse_types_FF)
            alpha_FF = torch.exp(-dt / tau_syn_FF).T  # Shape (2, n_synapse_types_FF)

        beta = torch.exp(-dt / self.tau_mem)  # Shape (n_neurons,)

        # Stack g_bar with its negative to simplify the difference of exponential conductance update
        g_scale = torch.stack(
            [-self.g_bar, self.g_bar], dim=0
        )  # Shape (2, n_synapse_types)
        norm_alpha = 1 / (
            self.tau_syn_decay - self.tau_syn_rise
        )  # Shape (n_synapse_types,)
        g_scale /= norm_alpha

        if inputs is not None:
            g_scale_FF = torch.stack(
                [-self.g_bar_FF, self.g_bar_FF], dim=0
            )  # Shape (2, n_synapse_types_FF)
            norm_alpha_FF = 1 / (
                self.tau_syn_decay_FF - self.tau_syn_rise_FF
            )  # Shape (n_synapse_types_FF,)
            g_scale_FF /= norm_alpha_FF

        # ==============
        # Run simulation
        # ==============

        for t in tqdm(range(n_steps), desc="Simulating network", unit="step"):
            # === Simulation Variables ===
            # s:                           (batch_size, n_neurons)                        -- spikes at current time step
            # I:                           (batch_size, n_neurons, n_synapse_types)       -- input current at current time step
            # I_FF:                        (batch_size, n_neurons, n_synapse_types_FF)    -- feedforward input current at current time step
            # v:                           (batch_size, n_neurons)                        -- membrane potentials at current time step
            # g:                           (batch_size, n_neurons, 2, n_synapse_types)    -- synaptic conductances at current time step
            # g_FF:                        (batch_size, n_neurons, 2, n_synapse_types_FF) -- feedforward synaptic conductances at current time step
            # self.U_rest:                 (n_neurons,)                                   -- resting potentials
            # self.E_L:                    (n_neurons,)                                   -- leak reversal potentials
            # self.g_L:                    (n_neurons,)                                   -- leak conductances
            # self.C_m:                    (n_neurons,)                                   -- membrane capacitances
            # alpha:                       (2, n_synapse_types)                           -- synaptic conductance decay factors
            # alpha_FF:                    (2, n_synapse_types_FF)                        -- feedforward synaptic conductance decay factors
            # beta:                        (n_neurons,)                                   -- membrane potential decay factors
            # self.theta:                  (n_neurons,)                                   -- spike thresholds
            # g_scale:                     (2, n_synapse_types)                           -- maximal synaptic conductances, stacked for easy summation
            # g_scale_FF:                  (2, n_synapse_types_FF)                        -- maximal feedforward synaptic conductances, stacked for easy summation
            # self.E_syn:                  (n_synapse_types,)                             -- synaptic reversal potentials
            # self.E_syn_FF:               (n_synapse_types_FF,)                          -- feedforward synaptic reversal potentials
            # self.cell_typed_weights:     (n_neurons, n_neurons, n_synapse_types)        -- recurrent synaptic weights
            # self.cell_typed_weights_FF:  (n_inputs, n_neurons, n_synapse_types_FF)      -- feedforward synaptic weights
            # ============================

            # Compute spikes
            s = self.spike_fn(v - self.theta)

            # Compute currents
            I = (
                g.sum(dim=2) * (v - self.E_syn)
            )  # Conductance times driving force (conductance is difference of rise and decay here)
            I_FF = g_FF.sum(dim=2) * (v - self.E_syn_FF) if inputs is not None else 0

            if inputs is not None:
                I += g_FF.sum(dim=2) * (v - self.E_syn_FF)

            # Update membrane potentials (without reset)
            v = (
                self.U_rest  # Resting potential
                + (v - self.U_rest) * beta  # Leak
                + I.sum(dim=2) * self.dt / self.C_m  # Recurrent current
                + (
                    I_FF.sum(dim=2) * self.dt / self.C_m if inputs is not None else 0
                )  # Feedforward current
            )

            # Reset membrane potentials where spikes occurred
            v = v * (1 - s) + self.U_reset * s

            # Compute conductance updates
            g = (
                g * alpha  # Decay with synapse time constant
                + torch.einsum(
                    "bi,ijc->bjc", s, self.cell_typed_weights
                )  # Sum over spikes with weights
                * g_scale  # Scale by g_bar and normalization factor for both rise and decay components
            )

            if inputs is not None:
                g_FF = (
                    g_FF * alpha_FF
                    + torch.einsum(
                        "bi,ijc->bjc", inputs[:, t, :], self.cell_typed_weights_FF
                    )
                    * g_scale_FF
                )

            # Store variables
            all_s[:, t, :] = s
            all_v[:, t, :] = v
            all_I[:, t, :, :] = I
            all_I_FF[:, t, :, :] = I_FF

        return all_s, all_v, all_I, all_I_FF

    def _validate(
        self,
        n_steps: int,
        dt: float,
        inputs: FloatArray | None,
        initial_v: FloatArray | None,
        initial_g: FloatArray | None,
        initial_g_FF: FloatArray | None,
    ) -> None:
        """Validate the inputs to the forward method."""

        # Determine batch size
        batch_size = inputs.shape[0] if inputs is not None else 1

        assert isinstance(dt, float), "dt must be a float."

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
                "initial_g must have 4 dimensions (batch_size, n_neurons, 2, n_synapse_types)."
            )
            assert initial_g.shape[0] == batch_size, (
                "initial_g batch size must match batch_size."
            )
            assert initial_g.shape[1] == self.n_neurons, (
                "initial_g must match n_neurons."
            )
            assert initial_g.shape[2] == 2, (
                "initial_g must have 2 for rise/decay components."
            )
            assert initial_g.shape[3] == self.n_synapse_types, (
                "initial_g must match the number of synapse types."
            )

        # Validate initial feedforward conductances if provided
        if initial_g_FF is not None:
            assert initial_g_FF.ndim == 4, (
                "initial_g_FF must have 4 dimensions (batch_size, n_neurons, 2, n_synapse_types_FF)."
            )
            assert initial_g_FF.shape[0] == batch_size, (
                "initial_g_FF batch size must match batch_size."
            )
            assert initial_g_FF.shape[1] == self.n_neurons, (
                "initial_g_FF must match n_neurons."
            )
            assert initial_g_FF.shape[2] == 2, (
                "initial_g_FF must have 2 for rise/decay components."
            )
            assert initial_g_FF.shape[3] == self.n_synapse_types_FF, (
                "initial_g_FF must match the number of feedforward synapse types."
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
