"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.conductance_based.model_init import ConductanceLIFNetwork_IO
from optimisation.surrogate_gradients import SurrGradSpike

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
            v = self.U_reset.repeat(batch_size, 1)

        # Synaptic conductances (batch_size, n_neurons, 2, n_synapse_types + n_synapse_types_FF)
        if initial_g is not None:
            g = torch.cat([initial_g, initial_g_FF], dim=-1)
        else:
            g = torch.zeros(
                (
                    batch_size,
                    self.n_neurons,
                    2,
                    self.n_synapse_types + self.n_synapse_types_FF,
                ),
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
            (
                batch_size,
                n_steps,
                self.n_neurons,
                2,
                self.n_synapse_types + self.n_synapse_types_FF,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # Synaptic input current storage (batch_size, n_steps, n_neurons, n_synapse_types)
        all_I = torch.empty(
            (
                batch_size,
                n_steps,
                self.n_neurons,
                self.n_synapse_types + self.n_synapse_types_FF,
            ),
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
            # Call single timestep method
            v, g, s, I = self._step(
                v,
                g,
                input_spikes[:, t, :],
                self.theta,
                self.surrgrad_scale,
                self.dt,
                self.beta,
                self.alpha,
                self.E_syn,
                self.E_L,
                self.C_m,
                self.U_reset,
                self.weights,
                self.weights_FF,
                self.cell_type_masks,
                self.cell_type_masks_FF,
                self.cell_to_synapse_mask,
                self.cell_to_synapse_mask_FF,
                self.scaling_factors,
                self.scaling_factors_FF,
                self.cell_type_indices,
                self.cell_type_indices_FF,
            )

            # Store variables
            all_s[:, t, :] = s.bool()
            all_v[:, t, :] = v
            all_I[:, t, :, :] = I
            all_g[:, t, :, :, :] = g

        return (
            all_s,
            all_v,
            all_I[:, :, :, : self.n_synapse_types],
            all_I[:, :, :, self.n_synapse_types :],
            all_g[:, :, :, :, : self.n_synapse_types],
            all_g[:, :, :, :, self.n_synapse_types :],
        )

    @staticmethod
    def _step(
        v: torch.Tensor,
        g: torch.Tensor,
        input_spikes_t: torch.Tensor,
        theta: torch.Tensor,
        surrgrad_scale: torch.Tensor,
        dt: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        E_syn: torch.Tensor,
        E_L: torch.Tensor,
        C_m: torch.Tensor,
        U_reset: torch.Tensor,
        weights: torch.Tensor,
        weights_FF: torch.Tensor,
        cell_type_masks: list[torch.Tensor],
        cell_type_masks_FF: list[torch.Tensor],
        cell_to_synapse_mask: torch.Tensor,
        cell_to_synapse_mask_FF: torch.Tensor,
        scaling_factors: torch.Tensor | None,
        scaling_factors_FF: torch.Tensor | None,
        cell_type_indices: torch.Tensor,
        cell_type_indices_FF: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single timestep of conductance-based LIF network simulation.

        Computes membrane potential updates, synaptic conductance dynamics, and spike generation
        for one simulation timestep using Euler integration.

        Args:
            v: Membrane potentials (batch_size, n_neurons) in mV.
            g: Synaptic conductances (batch_size, n_neurons, 2, n_synapse_types_total) in nS.
                Dimension 2 represents rise/decay components.
            input_spikes_t: External input spikes (batch_size, n_inputs) for current timestep.
            theta: Spike thresholds (n_neurons,) in mV.
            surrgrad_scale: Surrogate gradient scale parameter for spike function.
            dt: Integration timestep in ms.
            beta: Membrane potential decay factors (n_neurons,) = exp(-dt/tau_m).
            alpha: Synaptic decay factors (2, n_synapse_types_total) = exp(-dt/tau_syn).
            E_syn: Synaptic reversal potentials (n_synapse_types_total,) in mV.
            E_L: Leak reversal potentials (n_neurons,) in mV.
            C_m: Membrane capacitances (n_neurons,) in pF.
            U_reset: Reset potentials (n_neurons,) in mV.
            weights: Recurrent connectivity weights (n_neurons, n_neurons).
            weights_FF: Feedforward connectivity weights (n_inputs, n_neurons).
            cell_type_masks: Boolean masks for grouping recurrent neurons by cell type.
            cell_type_masks_FF: Boolean masks for grouping inputs by cell type.
            cell_to_synapse_mask: Maps recurrent cell types to their synapse types.
            cell_to_synapse_mask_FF: Maps feedforward cell types to their synapse types.
            scaling_factors: Recurrent scaling factors (n_cell_types, n_cell_types) or None.
            scaling_factors_FF: Feedforward scaling factors (n_cell_types_FF, n_cell_types) or None.
            cell_type_indices: Maps each neuron to its cell type (n_neurons,).
            cell_type_indices_FF: Maps each input to its cell type (n_inputs,).

        Returns:
            tuple: (updated_v, updated_g, spikes, synaptic_currents) where:
                - updated_v: New membrane potentials (batch_size, n_neurons).
                - updated_g: New synaptic conductances (batch_size, n_neurons, 2, n_synapse_types_total).
                - spikes: Binary spike indicators (batch_size, n_neurons).
                - synaptic_currents: Total synaptic currents (batch_size, n_neurons, n_synapse_types_total).
        """
        # Compute spikes using surrogate gradient for backpropagation
        s = SurrGradSpike.apply(v - theta, surrgrad_scale)

        # Compute currents
        I = (
            g.sum(dim=2) * (v[:, :, None] - E_syn[None, None, :])
        )  # Conductance times driving force (conductance is difference of rise and decay here)

        # Fused membrane potential update with numerical reset
        v = (E_L + (v - E_L) * beta - I.sum(dim=2) * dt / C_m) * (
            1 - s
        ) + U_reset * s.detach()

        # Compute conductance updates
        g *= alpha  # Decay with synapse time constant

        # Loop over cell types to update conductances, split by presynaptic cell type
        for k, cell_type_mask in enumerate(cell_type_masks):
            # Recurrent synapses - only process if this cell type has synapses
            if cell_to_synapse_mask[k].any():
                s_k = s[:, cell_type_mask]
                W_k = weights[cell_type_mask, :]
                # Apply scaling factors if they exist
                if scaling_factors is not None:
                    # For presynaptic cell type k, get scaling to each postsynaptic neuron
                    neuron_scaling = scaling_factors[
                        k, cell_type_indices
                    ]  # Shape: (n_neurons,)
                    W_k = (
                        W_k * neuron_scaling[None, :]
                    )  # Broadcast over presynaptic neurons
                g[:, :, 0, cell_to_synapse_mask[k]] += torch.matmul(s_k, W_k)[
                    :, :, None
                ]

        for k, cell_type_mask_FF in enumerate(cell_type_masks_FF):
            # Feedforward synapses - only process if this cell type has synapses
            if cell_to_synapse_mask_FF[k].any():
                input_k = input_spikes_t[:, cell_type_mask_FF].float()
                W_k_FF = weights_FF[cell_type_mask_FF, :]
                # Apply scaling factors if they exist
                if scaling_factors_FF is not None:
                    # For feedforward cell type k, get scaling to each postsynaptic neuron
                    neuron_scaling_FF = scaling_factors_FF[
                        k, cell_type_indices
                    ]  # Shape: (n_neurons,)
                    W_k_FF = (
                        W_k_FF * neuron_scaling_FF[None, :]
                    )  # Broadcast over presynaptic inputs
                g[:, :, 0, cell_to_synapse_mask_FF[k]] += torch.matmul(input_k, W_k_FF)[
                    :, :, None
                ]

        return v, g, s, I
