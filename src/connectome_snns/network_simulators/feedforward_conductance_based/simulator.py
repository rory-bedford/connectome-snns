"""Feedforward-only conductance-based LIF network simulator."""

import numpy as np
import torch
from typing import List
from numpy.typing import NDArray
from connectome_snns.network_simulators.feedforward_conductance_based.model_init import (
    FeedforwardConductanceLIFNetwork_IO,
)
from connectome_snns.training_utils.surrogate_gradients import (
    SurrGradSpike,
    SampleSpike,
)

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class FeedforwardConductanceLIFNetwork(FeedforwardConductanceLIFNetwork_IO):
    """
    Feedforward-only conductance-based LIF network simulator.

    This simulator maintains internal state variables (v, g_FF) that automatically
    continue across forward() calls, enabling efficient chunked simulation of long
    time series without explicit state management.

    State Management:
        - Internal state automatically continues between forward() calls
        - Call reset_state() before starting independent simulations
        - Call reset_state(batch_size=N) to change batch size

    Tracking Modes:
        - track_variables=False (default): Returns only spikes, minimal memory
        - track_variables=True: Returns full dict with all variables for analysis/visualization
        - track_gradients=False (default): Don't store gradient-enabled tensors
        - track_gradients=True: Store v, g_FF, s without detaching for gradient analysis

    Example:
        >>> # Continuous simulation across chunks
        >>> model = FeedforwardConductanceLIFNetwork(..., batch_size=10, track_variables=False)
        >>> for chunk in input_chunks:
        ...     spikes = model.forward(chunk)  # State continues automatically
        ...
        >>> # Independent simulation with full tracking
        >>> model.reset_state(batch_size=1)
        >>> model.track_variables = True
        >>> output_dict = model.forward(input_spikes)
        ...
        >>> # Gradient analysis
        >>> model.track_gradients = True
        >>> output_dict = model(input_spikes)
        >>> loss.backward()
        >>> gradients = model.get_tracked_gradients()  # Extract gradient magnitudes
    """

    def forward(
        self,
        input_spikes: torch.Tensor | FloatArray,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        Simulate the feedforward network for a given number of time steps.

        Internal state (v, g_FF) is updated in-place and continues across calls.
        Call reset_state() before starting a new independent simulation.

        Args:
            input_spikes (torch.Tensor | FloatArray): External input of shape
                (batch_size, n_steps, n_inputs). Batch size must match self.batch_size.
                Values are typically binary spikes (0/1), but continuous floats are
                also supported — the conductance update treats them as real-valued
                input activations (e.g. OU latent trajectories).

        Returns:
            When track_variables=False and return_probabilities=False:
                torch.Tensor: Spike trains of shape (batch_size, n_steps, n_neurons)

            When return_probabilities=True (and not tracking):
                tuple(probabilities, spikes): each (batch_size, n_steps, n_neurons);
                probabilities is the escape-noise p = sigmoid(scale*(v-theta)).

            When track_variables=True:
                dict[str, torch.Tensor]: Dictionary containing:
                    - "spikes": Spike trains (batch_size, n_steps, n_neurons)
                    - "voltages": Membrane potentials (batch_size, n_steps, n_neurons)
                    - "currents_feedforward": Feedforward synaptic currents (batch_size, n_steps, n_neurons, n_synapse_types_FF)
                    - "currents_leak": Leak currents (batch_size, n_steps, n_neurons)
                    - "conductances_feedforward": Feedforward conductances (batch_size, n_steps, n_neurons, 2, n_synapse_types_FF)

        Raises:
            ValueError: If input_spikes batch size doesn't match self.batch_size
        """
        # Convert to tensor if needed
        if isinstance(input_spikes, np.ndarray):
            input_spikes = torch.from_numpy(input_spikes).to(self.device)

        # Validate inputs
        self._validate_forward(input_spikes)

        # Ensure float dtype once (avoids per-timestep .float() in step functions)
        input_spikes = input_spikes.float()

        n_steps = input_spikes.shape[1]

        # ==========================
        # Conditionally allocate tracking arrays
        # ==========================

        # Storage for gradient-enabled tensors (if track_gradients=True)
        if self.track_gradients:
            self._tracked_v_list = []
            self._tracked_g_FF_list = []
            self._tracked_s_list = []

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
                    self.n_synapse_types_FF,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            all_I = torch.empty(
                (
                    tracking_batch_size,
                    n_steps,
                    self.n_neurons,
                    self.n_synapse_types_FF + 1,
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

        # Spike-probability storage (only when requested; carries gradient like all_s).
        all_p = (
            torch.empty(
                (self.batch_size, n_steps, self.n_neurons),
                dtype=torch.float32,
                device=self.device,
            )
            if self.return_probabilities
            else None
        )

        # ==============
        # Run simulation
        # ==============

        # ── Per-forward-call precomputation ──────────────────────────────
        # Assemble (connection_weights, synapse_kernel) pairs for each cell
        # type.  Init-cached tensors are looked up; learnable parameters are
        # snapshotted here so the time loop only does matmul + broadcast.
        # See _precompute_weight_products() for the full caching strategy.

        weights_FF = self.weights_FF if self._has_weights_mode else None
        scaling_factors_FF = self.scaling_factors_FF if self._has_scaling_mode else None
        cell_type_masks = self.cell_type_masks_FF

        all_masks: List[torch.Tensor] = []
        all_syn_masks: List[torch.Tensor] = []
        all_conn_weights: List[torch.Tensor] = []  # (n_inputs_k, n_neurons)
        all_kernels: List[torch.Tensor] = []  # (n_neurons, n_rise, n_syn_k)

        for k in self._fixed_cell_types:
            # Both conn_weights and kernel cached at init — nothing to compute
            all_masks.append(cell_type_masks[k])
            all_syn_masks.append(self.cell_to_synapse_mask_FF[k])
            all_conn_weights.append(getattr(self, f"connection_weights_{k}"))
            all_kernels.append(getattr(self, f"synapse_kernel_{k}"))

        for k in self._weights_cell_types:
            # conn_weights from current learnable weights; kernel cached at init
            mask = cell_type_masks[k]
            all_masks.append(mask)
            all_syn_masks.append(self.cell_to_synapse_mask_FF[k])
            all_conn_weights.append(weights_FF[mask, :])
            all_kernels.append(getattr(self, f"synapse_kernel_{k}"))

        for k in self._scaling_cell_types:
            # conn_weights cached at init; kernel scaled by current learnable SF
            sf = scaling_factors_FF[k, self.cell_type_indices]  # (n_neurons,)
            all_masks.append(cell_type_masks[k])
            all_syn_masks.append(self.cell_to_synapse_mask_FF[k])
            all_conn_weights.append(getattr(self, f"connection_weights_{k}"))
            all_kernels.append(getattr(self, f"synapse_kernel_{k}") * sf[:, None, None])

        ff_pathways = list(zip(all_masks, all_syn_masks, all_conn_weights, all_kernels))

        for t in range(n_steps):
            self.v, self.g_FF, s, I, I_leak, prob = self._step_common(
                self.v,
                self.g_FF,
                self.theta,
                self.surrgrad_scale,
                self.dt,
                self.beta,
                self.alpha,
                self.E_syn,
                self.E_L,
                self.C_m,
                self.U_reset,
                self.g_clip,
                self.spike_mode,
                ff_pathways,
                input_spikes[:, t, :],
                self.return_probabilities,
            )

            # Store spike output (spikes need gradients for training!)
            all_s[:, t, :] = s
            if self.return_probabilities:
                all_p[:, t, :] = prob

            # Store gradient-enabled tensors (must retain grad on actual tensors in graph)
            if self.track_gradients:
                # Retain gradients on the actual tensors that are part of the computation graph
                # Only call retain_grad() if tensor has requires_grad=True
                if self.v.requires_grad:
                    self.v.retain_grad()
                if self.g_FF.requires_grad:
                    self.g_FF.retain_grad()
                if s.requires_grad:
                    s.retain_grad()
                # Store references to these tensors (they will be reassigned in next iteration)
                self._tracked_v_list.append(self.v)
                self._tracked_g_FF_list.append(self.g_FF)
                self._tracked_s_list.append(s)

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
                    all_g[:, t, :, :, :] = self.g_FF[
                        self.track_batch_idx : self.track_batch_idx + 1, :, :, :
                    ].detach()
                else:
                    # Track all batch elements
                    all_v[:, t, :] = self.v.detach()
                    all_I[:, t, :, :-1] = I.detach()
                    all_I[:, t, :, -1] = I_leak.detach()
                    all_g[:, t, :, :, :] = self.g_FF.detach()

        # CRITICAL: Detach state tensors to prevent carrying computation graph to next chunk
        # Without this, chunk N+1 would try to use state from chunk N's (freed) graph
        self.v = self.v.detach()
        self.g_FF = self.g_FF.detach()

        # ==============
        # Return results
        # ==============

        if self.track_variables:
            out = {
                "spikes": all_s,
                "voltages": all_v,
                "currents_feedforward": all_I[:, :, :, : self.n_synapse_types_FF],
                "currents_leak": all_I[:, :, :, -1],
                "conductances_feedforward": all_g,
            }
            if self.return_probabilities:
                out["probabilities"] = all_p
            return out
        elif self.return_probabilities:
            # (probabilities, spikes): the trainer / _spikes_from_output unpack this.
            return all_p, all_s
        else:
            return all_s

    def get_tracked_gradients(self) -> dict[str, torch.Tensor]:
        """
        Extract gradient magnitudes from tracked tensors.

        Call this AFTER calling backward() on a loss that depends on the model output.
        Requires track_gradients=True during forward pass.

        Returns:
            dict containing gradient magnitude tensors:
                - "grad_v": Shape (time, batch, neurons) - voltage gradients
                - "grad_g_FF": Shape (time, batch, neurons, 2, n_synapse_types_FF) - conductance gradients
                - "grad_s": Shape (time, batch, neurons) - spike gradients

        Raises:
            RuntimeError: If track_gradients was False or backward() wasn't called
        """
        if not self.track_gradients:
            raise RuntimeError("track_gradients must be True during forward pass")

        if not hasattr(self, "_tracked_v_list") or len(self._tracked_v_list) == 0:
            raise RuntimeError(
                "No tracked tensors found. Did you call forward() with track_gradients=True?"
            )

        # Extract gradients (use .grad if available, else 0)
        grad_v_list = []
        grad_g_FF_list = []
        grad_s_list = []

        for v, g, s in zip(
            self._tracked_v_list, self._tracked_g_FF_list, self._tracked_s_list
        ):
            # Extract gradient if it exists, otherwise zeros
            if v.grad is not None:
                grad_v_list.append(v.grad.detach().clone())
            else:
                grad_v_list.append(torch.zeros_like(v))

            if g.grad is not None:
                grad_g_FF_list.append(g.grad.detach().clone())
            else:
                grad_g_FF_list.append(torch.zeros_like(g))

            if s.grad is not None:
                grad_s_list.append(s.grad.detach().clone())
            else:
                grad_s_list.append(torch.zeros_like(s))

        # Stack into tensors (time, batch, ...)
        return {
            "grad_v": torch.stack(grad_v_list, dim=0),
            "grad_g_FF": torch.stack(grad_g_FF_list, dim=0),
            "grad_s": torch.stack(grad_s_list, dim=0),
        }

    # ======================================================================
    # Factored step functions
    # ======================================================================

    @staticmethod
    def _step_common(
        v: torch.Tensor,
        g: torch.Tensor,
        theta: torch.Tensor,
        surrgrad_scale: torch.Tensor,
        dt: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        E_syn: torch.Tensor,
        E_L: torch.Tensor,
        C_m: torch.Tensor,
        U_reset: torch.Tensor,
        g_clip: torch.Tensor,
        spike_mode: str,
        ff_pathways: list,
        input_t: torch.Tensor,
        return_probabilities: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Single simulation timestep: spike, voltage, conductance update.

        Args:
            ff_pathways: List of (mask, syn_mask, conn_w, kernel) for feedforward inputs.
            input_t: Feedforward input spikes for this timestep (batch, n_ff).
            return_probabilities: also compute the escape-noise spike probability
                ``p = sigmoid(surrgrad_scale*(v-theta))`` (else ``prob`` is None).

        Returns:
            (v, g, s, I, I_leak, prob)
        """
        # Compute spikes
        if spike_mode == "deterministic":
            s = SurrGradSpike.apply(v - theta, surrgrad_scale)
        elif spike_mode == "sample":
            s = SampleSpike.apply(v - theta, surrgrad_scale)
        else:
            s = SurrGradSpike.apply(v - theta, surrgrad_scale)

        # Escape-noise spike probability from the SAME pre-update v as the spike, so
        # E[s] = prob in sample mode. Differentiable in v (a true gradient to the
        # scaling factors, not the surrogate on s) — rate losses can filter this to
        # drop the sampling-variance term that otherwise makes silence optimal.
        prob = (
            torch.sigmoid(surrgrad_scale * (v - theta))
            if return_probabilities
            else None
        )

        # Compute summed conductance and clip to saturation bound
        g_total = g.sum(dim=2)  # (batch, neurons, n_synapse_types)
        g_total = torch.clamp(g_total, min=0.0)
        g_total = torch.minimum(g_total, g_clip[None, None, :])

        # Compute currents from clipped summed conductance
        I = g_total * (v[:, :, None] - E_syn[None, None, :])
        I_leak = (v - E_L) * (1 - beta) * C_m / dt

        # Update membrane potential with reset
        v = (v - (I.sum(dim=2) + I_leak) * dt / C_m) * (
            1 - s.detach()
        ) + U_reset * s.detach()

        # Decay conductances
        g = g * alpha

        # Inject feedforward conductances from input
        for mask, syn_mask, conn_w, kernel in ff_pathways:
            activation = input_t[:, mask] @ conn_w
            g[:, :, :, syn_mask] += activation[:, :, None, None] * kernel[None, :, :, :]

        return v, g, s, I, I_leak, prob
