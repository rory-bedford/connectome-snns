"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.conductance_based.model_init import ConductanceLIFNetwork_IO

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class ConductanceLIFNetwork(ConductanceLIFNetwork_IO):
    """Conductance-based LIF network simulator with connectome-constrained weights."""

    def forward(
        self,
        input_spikes: FloatArray | None = None,
        initial_v: FloatArray | None = None,
        initial_g: FloatArray | None = None,
        initial_g_FF: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Args:
            input_spikes (FloatArray | None): External input spikes of shape (batch_size, n_steps, n_inputs).
            initial_v (FloatArray | None): Initial membrane potentials of shape (batch_size, n_neurons). Defaults to resting potentials.
            initial_g (FloatArray | None): Initial synaptic conductances of shape (batch_size, n_neurons, 2, n_cell_types). Defaults to zeros.
            initial_g_FF (FloatArray | None): Initial feedforward synaptic conductances of shape (batch_size, n_neurons, 2, n_cell_types_FF). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - all_s: Spike trains of shape (batch_size, n_steps, n_neurons)
                - all_v: Membrane potentials of shape (batch_size, n_steps, n_neurons)
                - all_I: Synaptic input currents of shape (batch_size, n_steps, n_neurons, n_cell_types)
                - all_I_FF: Feedforward synaptic input currents of shape (batch_size, n_steps, n_neurons, n_cell_types_FF)
                - all_g: Synaptic conductances of shape (batch_size, n_steps, n_neurons, 2, n_cell_types) where dim 3 is [rise, decay]
                - all_g_FF: Feedforward synaptic conductances of shape (batch_size, n_steps, n_neurons, 2, n_cell_types_FF) where dim 3 is [rise, decay]
        """

        # ===============
        # Validate inputs
        # ===============

        self._validate_forward(input_spikes, initial_v, initial_g, initial_g_FF)

        # ==========================
        # Initialize state variables
        # ==========================

        batch_size = input_spikes.shape[0]
        n_steps = input_spikes.shape[1]

        # Membrane potentials (batch_size, n_neurons)
        if initial_v is not None:
            v = initial_v
        else:
            v = self.U_reset.clone().detach()
            v = v.repeat(batch_size, 1)

        # Synaptic conductances (batch_size, n_neurons, 2, n_synapse_types + n_synapse_types_FF)
        if initial_g is not None:
            g = torch.stack(
                [initial_g, initial_g_FF], dim=-1
            )
        else:
            g = torch.zeros(
                (batch_size, self.n_neurons, 2, self.n_synapse_types + self.n_synapse_types_FF),
                dtype=torch.float32,
                device=self.device,
            )

        # Membrane potential storage (batch_size, n_steps, n_neurons)
        all_v = torch.empty(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )

        # Synaptic conductance storage (batch_size, n_steps, n_neurons, 2, n_synapse_types)
        all_g = torch.empty(
            (batch_size, n_steps, self.n_neurons, 2, self.n_synapse_types + self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )

        # Synaptic input current storage (batch_size, n_steps, n_neurons, n_synapse_types)
        all_I = torch.empty(
            (batch_size, n_steps, self.n_neurons, self.n_synapse_types + self.n_synapse_types_FF),
            dtype=torch.float32,
            device=self.device,
        )

        # Spike train storage (batch_size, n_steps, n_neurons)
        all_s = torch.empty(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.bool,
            device=self.device,
        )

        # ==============
        # Run simulation
        # ==============

        iterator = range(n_steps)
        if self.use_tqdm:
            iterator = tqdm(iterator, desc="Simulating network", unit="step")

        for t in iterator:
            # === Simulation Variables ===
            # s:                           (batch_size, n_neurons)                                              -- spikes at current time step
            # I:                           (batch_size, n_neurons, n_synapse_types + n_synapse_types_FF)       -- input current at current time step
            # v:                           (batch_size, n_neurons)                                              -- membrane potentials at current time step
            # g:                           (batch_size, n_neurons, 2, n_synapse_types + n_synapse_types_FF)    -- synaptic conductances at current time step
            # self.U_rest:                 (n_neurons,)                                                         -- resting potentials
            # self.E_L:                    (n_neurons,)                                                         -- leak reversal potentials
            # self.g_L:                    (n_neurons,)                                                         -- leak conductances
            # self.C_m:                    (n_neurons,)                                                         -- membrane capacitances
            # self.alpha:                  (2, n_synapse_types + n_synapse_types_FF)                           -- synaptic conductance decay factors
            # beta:                        (n_neurons,)                                                         -- membrane potential decay factors
            # self.theta:                  (n_neurons,)                                                         -- spike thresholds
            # self.g_scale:                (2, n_synapse_types + n_synapse_types_FF)                           -- maximal synaptic conductances, stacked for easy summation
            # self.E_syn:                  (n_synapse_types + n_synapse_types_FF,)                             -- synaptic reversal potentials (recurrent + feedforward)
            # self.weights_combined:       (n_neurons + n_inputs, n_neurons)                                   -- combined weight matrix (recurrent + feedforward stacked vertically)
            # ============================

            # Compute spikes
            s = self.spike_fn(v - self.theta)

            # Compute currents
            I = (
                g.sum(dim=2) * (v[:, :, None] - self.E_syn[None, None, :])
            )  # Conductance times driving force (conductance is difference of rise and decay here)
            I_FF = (
                g_FF.sum(dim=2) * (v[:, :, None] - self.E_syn_FF[None, None, :])
                if input_spikes is not None
                else 0
            )

            # Update membrane potentials (without reset)
            v = (
                self.E_L  # Resting potential
                + (v - self.E_L) * self.beta  # Leak
                - (I.sum(dim=2) + (I_FF.sum(dim=2) if input_spikes is not None else 0))
                * self.dt
                / self.C_m  # Combined current
            )

            # Reset membrane potentials where spikes occurred
            v = v * (1 - s) + self.U_reset * s

            # Compute conductance updates
            g = (
                g * self.alpha  # Decay with synapse time constant
                + (
                    torch.einsum("bi,cij->bjc", s, self.cell_typed_weights)[
                        :, :, None, :
                    ]  # Sum over spikes with weights
                    * self.g_scale[None, None, :, :]
                )  # Scale by g_bar and normalization factor for both rise and decay components
            )

            if input_spikes is not None:
                g_FF = g_FF * alpha_FF + (
                    torch.einsum(
                        "bi,cij->bjc", input_spikes[:, t, :], self.cell_typed_weights_FF
                    )[:, :, None, :]
                    * g_scale_FF[None, None, :, :]
                )

            # Store variables
            all_s[:, t, :] = s
            all_v[:, t, :] = v
            all_I[:, t, :, :] = I
            all_g[:, t, :, :, :] = g

        return all_s, all_v, all_I[:, :, :, :self.n_synapse_types], all_I[:, :, :, self.n_synapse_types:], all_g[:, :, :, :, :self.n_synapse_types], all_g[:, :, :, :, self.n_synapse_types:]
