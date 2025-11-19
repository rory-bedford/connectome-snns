"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
from typing import List
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
                - all_I: Synaptic input currents of shape (batch_size, n_steps, n_neurons, n_synapse_types)
                - all_I_FF: Feedforward synaptic input currents of shape (batch_size, n_steps, n_neurons, n_synapse_types_FF)
                - all_I_leak: Leak currents of shape (batch_size, n_steps, n_neurons)
                - all_g: Synaptic conductances of shape (batch_size, n_steps, n_neurons, 2, n_synapse_types) where dim 3 is [rise, decay]
                - all_g_FF: Feedforward synaptic conductances of shape (batch_size, n_steps, n_neurons, 2, n_synapse_types_FF) where dim 3 is [rise, decay]
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
                self.n_synapse_types + self.n_synapse_types_FF + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # Spike train storage (batch_size, n_steps, n_neurons)
        all_s = torch.empty(
            (batch_size, n_steps, self.n_neurons),
            dtype=torch.float32,
            device=self.device,
        )

        # ==============
        # Run simulation
        # ==============

        # Gather cached weight tensors into lists once (outside the loop)
        cached_rec = [
            getattr(self, f"cached_rec_{i}")
            for i in range(len(self.cached_weights_rec_masks))
        ]
        cached_ff = [
            getattr(self, f"cached_ff_{i}")
            for i in range(len(self.cached_weights_ff_masks))
        ]

        iterator = range(n_steps)
        if self.use_tqdm:
            iterator = tqdm(iterator, desc="Simulating network", unit="step")

        for t in iterator:
            # Call the appropriate _step method based on optimization mode
            if self.optimisable is None:
                v, g, s, I, I_leak = self._step_inference(
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
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    cached_ff,
                )
            elif self.optimisable == "weights":
                v, g, s, I, I_leak = self._step_optimize_weights(
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
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    self.cached_weights_rec_indices,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )
            elif self.optimisable == "scaling_factors":
                v, g, s, I, I_leak = self._step_optimize_scaling_factors(
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
                    self.scaling_factors,
                    self.scaling_factors_FF,
                    self.cell_type_indices,
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    self.cached_weights_rec_indices,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )

            # Store variables
            all_s[:, t, :] = s
            all_v[:, t, :] = v
            all_I[:, t, :, :-1] = I
            all_I[:, t, :, -1] = I_leak
            all_g[:, t, :, :, :] = g

        return (
            all_s,
            all_v,
            all_I[:, :, :, : self.n_synapse_types],
            all_I[
                :,
                :,
                :,
                self.n_synapse_types : self.n_synapse_types + self.n_synapse_types_FF,
            ],
            all_I[:, :, :, -1],
            all_g[:, :, :, :, : self.n_synapse_types],
            all_g[:, :, :, :, self.n_synapse_types :],
        )

    @staticmethod
    def _step_inference(
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
        masks_rec: List[torch.Tensor],
        syn_masks_rec: List[torch.Tensor],
        weights_rec: List[torch.Tensor],
        masks_ff: List[torch.Tensor],
        syn_masks_ff: List[torch.Tensor],
        weights_ff: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized _step for inference mode (optimisable=None).

        Uses fully precomputed weights * scaling_factors * g_scale.
        No runtime multiplications except the einsum.
        """
        # Compute spikes
        s = SurrGradSpike.apply(v - theta, surrgrad_scale)

        # Compute currents
        I = g.sum(dim=2) * (v[:, :, None] - E_syn[None, None, :])
        I_leak = (v - E_L) * (1 - beta) * C_m / dt

        # Update membrane potential with reset
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (1 - s) + U_reset * s.detach()

        # Decay conductances
        g *= alpha

        # Update conductances (recurrent) - everything precomputed
        for mask, syn_mask, weights in zip(masks_rec, syn_masks_rec, weights_rec):
            # s[:, mask] → (batch, n_in_type)
            # weights → (n_in_type, n_neurons, 2, n_syn)
            # Result after einsum: (batch, n_neurons, 2, n_syn)
            g[:, :, :, syn_mask] += torch.einsum("bi,ijkl->bjkl", s[:, mask], weights)

        # Update conductances (feedforward) - everything precomputed
        for mask, syn_mask, weights in zip(masks_ff, syn_masks_ff, weights_ff):
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl", input_spikes_t[:, mask].float(), weights
            )

        return v, g, s, I, I_leak

    @staticmethod
    def _step_optimize_weights(
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
        masks_rec: List[torch.Tensor],
        syn_masks_rec: List[torch.Tensor],
        indices_rec: List[int],
        cached_weights_rec: List[torch.Tensor],
        masks_ff: List[torch.Tensor],
        syn_masks_ff: List[torch.Tensor],
        indices_ff: List[int],
        cached_weights_ff: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized _step for training weights (optimisable="weights").

        Uses precomputed scaling_factors * g_scale.
        Weights remain dynamic for gradient flow.
        """
        # Compute spikes
        s = SurrGradSpike.apply(v - theta, surrgrad_scale)

        # Compute currents
        I = g.sum(dim=2) * (v[:, :, None] - E_syn[None, None, :])
        I_leak = (v - E_L) * (1 - beta) * C_m / dt

        # Update membrane potential with reset
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (1 - s) + U_reset * s.detach()

        # Decay conductances
        g *= alpha

        # Update conductances (recurrent) - weights stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_rec, syn_masks_rec, indices_rec, cached_weights_rec
        ):
            # weights[mask, :] → (n_in_type, n_neurons)
            # cached → (n_neurons, 2, n_syn)
            # Multiply them: (n_in_type, n_neurons, 2, n_syn)
            weighted = weights[mask, :][:, :, None, None] * cached[None, :, :, :]
            g[:, :, :, syn_mask] += torch.einsum("bi,ijkl->bjkl", s[:, mask], weighted)

        # Update conductances (feedforward) - weights stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_ff, syn_masks_ff, indices_ff, cached_weights_ff
        ):
            weighted = weights_FF[mask, :][:, :, None, None] * cached[None, :, :, :]
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl", input_spikes_t[:, mask].float(), weighted
            )

        return v, g, s, I, I_leak

    @staticmethod
    def _step_optimize_scaling_factors(
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
        scaling_factors: torch.Tensor,
        scaling_factors_FF: torch.Tensor,
        cell_type_indices: torch.Tensor,
        masks_rec: List[torch.Tensor],
        syn_masks_rec: List[torch.Tensor],
        indices_rec: List[int],
        cached_weights_rec: List[torch.Tensor],
        masks_ff: List[torch.Tensor],
        syn_masks_ff: List[torch.Tensor],
        indices_ff: List[int],
        cached_weights_ff: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized _step for training scaling factors (optimisable="scaling_factors").

        Uses precomputed weights * g_scale.
        Scaling factors remain dynamic for gradient flow.
        """
        # Compute spikes
        s = SurrGradSpike.apply(v - theta, surrgrad_scale)

        # Compute currents
        I = g.sum(dim=2) * (v[:, :, None] - E_syn[None, None, :])
        I_leak = (v - E_L) * (1 - beta) * C_m / dt

        # Update membrane potential with reset
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (1 - s) + U_reset * s.detach()

        # Decay conductances
        g *= alpha

        # Update conductances (recurrent) - scaling_factors stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_rec, syn_masks_rec, indices_rec, cached_weights_rec
        ):
            # cached → (n_in_type, n_neurons, 2, n_syn)
            # scaling_factors[k, cell_type_indices] → (n_neurons,)
            # Multiply them: (n_in_type, n_neurons, 2, n_syn)
            scaled = cached * scaling_factors[k, cell_type_indices][None, :, None, None]
            g[:, :, :, syn_mask] += torch.einsum("bi,ijkl->bjkl", s[:, mask], scaled)

        # Update conductances (feedforward) - scaling_factors stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_ff, syn_masks_ff, indices_ff, cached_weights_ff
        ):
            scaled = (
                cached * scaling_factors_FF[k, cell_type_indices][None, :, None, None]
            )
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl", input_spikes_t[:, mask].float(), scaled
            )

        return v, g, s, I, I_leak
