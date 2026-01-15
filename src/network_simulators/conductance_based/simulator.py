"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
from typing import List
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.conductance_based.model_init import ConductanceLIFNetwork_IO
from training_utils.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class ConductanceLIFNetwork(ConductanceLIFNetwork_IO):
    """
    Conductance-based LIF network simulator with connectome-constrained weights.

    This simulator maintains internal state variables (v, g, g_FF) that automatically
    continue across forward() calls, enabling efficient chunked simulation of long
    time series without explicit state management.

    State Management:
        - Internal state automatically continues between forward() calls
        - Call reset_state() before starting independent simulations
        - Call reset_state(batch_size=N) to change batch size

    Tracking Modes:
        - track_variables=False (default): Returns only spikes, minimal memory
        - track_variables=True: Returns full dict with all variables for analysis/visualization

    Example:
        >>> # Continuous simulation across chunks
        >>> model = ConductanceLIFNetwork(..., batch_size=10, track_variables=False)
        >>> for chunk in input_chunks:
        ...     spikes = model.forward(chunk)  # State continues automatically
        ...
        >>> # Independent simulation with full tracking
        >>> model.reset_state(batch_size=1)
        >>> model.track_variables = True
        >>> output_dict = model.forward(input_spikes)
    """

    def forward(
        self,
        input_spikes: torch.Tensor | FloatArray,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Internal state (v, g, g_FF) is updated in-place and continues across calls.
        Call reset_state() before starting a new independent simulation.

        Args:
            input_spikes (torch.Tensor | FloatArray): External input spikes of shape
                (batch_size, n_steps, n_inputs). Batch size must match self.batch_size.

        Returns:
            When track_variables=False:
                torch.Tensor: Spike trains of shape (batch_size, n_steps, n_neurons)

            When track_variables=True:
                dict[str, torch.Tensor]: Dictionary containing:
                    - "spikes": Spike trains (batch_size, n_steps, n_neurons)
                    - "voltages": Membrane potentials (batch_size, n_steps, n_neurons)
                    - "currents_recurrent": Recurrent synaptic currents (batch_size, n_steps, n_neurons, n_synapse_types)
                    - "currents_feedforward": Feedforward synaptic currents (batch_size, n_steps, n_neurons, n_synapse_types_FF)
                    - "currents_leak": Leak currents (batch_size, n_steps, n_neurons)
                    - "conductances_recurrent": Recurrent conductances (batch_size, n_steps, n_neurons, 2, n_synapse_types)
                    - "conductances_feedforward": Feedforward conductances (batch_size, n_steps, n_neurons, 2, n_synapse_types_FF)

        Raises:
            ValueError: If input_spikes batch size doesn't match self.batch_size
        """
        # Convert to tensor if needed
        if isinstance(input_spikes, np.ndarray):
            input_spikes = torch.from_numpy(input_spikes).float().to(self.device)

        # Validate inputs
        self._validate_forward(input_spikes)

        n_steps = input_spikes.shape[1]

        # Concatenate recurrent and feedforward conductances for unified processing
        # Shape: (batch_size, n_neurons, 2, n_synapse_types + n_synapse_types_FF)
        g = torch.cat([self.g, self.g_FF], dim=-1)

        # ==========================
        # Conditionally allocate tracking arrays
        # ==========================

        if self.track_variables:
            # Determine batch dimension for tracking
            tracking_batch_size = (
                1 if self.track_batch_idx is not None else self.batch_size
            )

            # Allocate storage for all variables
            all_v = torch.empty(
                (tracking_batch_size, n_steps, self.n_neurons),
                dtype=torch.float32,
                device=self.device,
            )
            all_g = torch.empty(
                (
                    tracking_batch_size,
                    n_steps,
                    self.n_neurons,
                    2,
                    self.n_synapse_types + self.n_synapse_types_FF,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            all_I = torch.empty(
                (
                    tracking_batch_size,
                    n_steps,
                    self.n_neurons,
                    self.n_synapse_types + self.n_synapse_types_FF + 1,
                ),
                dtype=torch.float32,
                device=self.device,
            )

        # Always allocate spike storage
        all_s = torch.empty(
            (self.batch_size, n_steps, self.n_neurons),
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
                self.v, g, s, I, I_leak = self._step_inference(
                    self.v,
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
                    self.g_mins,
                    self.g_maxs,
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    cached_ff,
                )
            elif self.optimisable == "weights":
                self.v, g, s, I, I_leak = self._step_optimize_weights(
                    self.v,
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
                    self.g_mins,
                    self.g_maxs,
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    self.cached_weights_rec_indices,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )
            elif self.optimisable in [
                "scaling_factors",
                "scaling_factors_recurrent",
                "scaling_factors_feedforward",
            ]:
                self.v, g, s, I, I_leak = self._step_optimize_scaling_factors(
                    self.v,
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
                    self.g_mins,
                    self.g_maxs,
                    self.cached_weights_rec_masks,
                    self.cached_weights_rec_syn_masks,
                    self.cached_weights_rec_indices,
                    cached_rec,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )

            # Store spike output (spikes need gradients for training!)
            all_s[:, t, :] = s

            # Conditionally store other variables (detach these for logging only)
            if self.track_variables:
                if self.track_batch_idx is not None:
                    # Only track specified batch index
                    all_v[:, t, :] = self.v[
                        self.track_batch_idx : self.track_batch_idx + 1, :
                    ].detach()
                    all_I[:, t, :, :-1] = I[
                        self.track_batch_idx : self.track_batch_idx + 1, :, :
                    ].detach()
                    all_I[:, t, :, -1] = I_leak[
                        self.track_batch_idx : self.track_batch_idx + 1, :
                    ].detach()
                    all_g[:, t, :, :, :] = g[
                        self.track_batch_idx : self.track_batch_idx + 1, :, :, :
                    ].detach()
                else:
                    # Track all batch elements
                    all_v[:, t, :] = self.v.detach()
                    all_I[:, t, :, :-1] = I.detach()
                    all_I[:, t, :, -1] = I_leak.detach()
                    all_g[:, t, :, :, :] = g.detach()

        # Update internal state variables (split g back into recurrent and feedforward)
        self.g = g[:, :, :, : self.n_synapse_types]
        self.g_FF = g[:, :, :, self.n_synapse_types :]

        # CRITICAL: Detach state tensors to prevent carrying computation graph to next chunk
        # Without this, chunk N+1 would try to use state from chunk N's (freed) graph
        self.v = self.v.detach()
        self.g = self.g.detach()
        self.g_FF = self.g_FF.detach()

        # ==============
        # Return results
        # ==============

        if self.track_variables:
            return {
                "spikes": all_s,
                "voltages": all_v,
                "currents_recurrent": all_I[:, :, :, : self.n_synapse_types],
                "currents_feedforward": all_I[
                    :,
                    :,
                    :,
                    self.n_synapse_types : self.n_synapse_types
                    + self.n_synapse_types_FF,
                ],
                "currents_leak": all_I[:, :, :, -1],
                "conductances_recurrent": all_g[:, :, :, :, : self.n_synapse_types],
                "conductances_feedforward": all_g[:, :, :, :, self.n_synapse_types :],
            }
        else:
            return all_s

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
        g_mins: torch.Tensor,
        g_maxs: torch.Tensor,
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
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (
            1 - s.detach()
        ) + U_reset * s.detach()

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

        # Clip rise and decay components to their physiological peaks
        g = torch.clamp(g, min=g_mins[None, None, :, :], max=g_maxs[None, None, :, :])

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
        g_mins: torch.Tensor,
        g_maxs: torch.Tensor,
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
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (
            1 - s.detach()
        ) + U_reset * s.detach()

        # Decay conductances
        g *= alpha

        # Update conductances (recurrent) - weights stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_rec, syn_masks_rec, indices_rec, cached_weights_rec
        ):
            # weights[mask, :] → (n_in_type, n_neurons)
            # cached → (n_neurons, 2, n_syn)
            # Multiply them: (n_in_type, n_neurons, 2, n_syn)
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl",
                s[:, mask],
                weights[mask, :][:, :, None, None] * cached[None, :, :, :],
            )

        # Update conductances (feedforward) - weights stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_ff, syn_masks_ff, indices_ff, cached_weights_ff
        ):
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl",
                input_spikes_t[:, mask].float(),
                weights_FF[mask, :][:, :, None, None] * cached[None, :, :, :],
            )

        # Clip rise and decay components to their physiological peaks
        g = torch.clamp(g, min=g_mins[None, None, :, :], max=g_maxs[None, None, :, :])

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
        g_mins: torch.Tensor,
        g_maxs: torch.Tensor,
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
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (
            1 - s.detach()
        ) + U_reset * s.detach()

        # Decay conductances
        g *= alpha

        # Update conductances (recurrent) - scaling_factors stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_rec, syn_masks_rec, indices_rec, cached_weights_rec
        ):
            # cached → (n_in_type, n_neurons, 2, n_syn)
            # scaling_factors[k, cell_type_indices] → (n_neurons,)
            # Apply scaling after einsum to reduce memory usage
            g[:, :, :, syn_mask] += (
                torch.einsum("bi,ijkl->bjkl", s[:, mask], cached)
                * scaling_factors[k, cell_type_indices][None, :, None, None]
            )

        # Update conductances (feedforward) - scaling_factors stay dynamic
        for mask, syn_mask, k, cached in zip(
            masks_ff, syn_masks_ff, indices_ff, cached_weights_ff
        ):
            g[:, :, :, syn_mask] += (
                torch.einsum("bi,ijkl->bjkl", input_spikes_t[:, mask].float(), cached)
                * scaling_factors_FF[k, cell_type_indices][None, :, None, None]
            )

        # Clip rise and decay components to their physiological peaks
        g = torch.clamp(g, min=g_mins[None, None, :, :], max=g_maxs[None, None, :, :])

        return v, g, s, I, I_leak
