"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
from numpy.typing import NDArray
from connectome_snns.network_simulators.conductance_based.model_init import (
    ConductanceLIFNetwork_IO,
)
from connectome_snns.training_utils.surrogate_gradients import (
    SurrGradSpike,
    SampleSpike,
)

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
        - track_gradients=False (default): Don't store gradient-enabled tensors
        - track_gradients=True: Store v, g, s without detaching for gradient analysis

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
                    - "currents": Synaptic currents (batch_size, n_steps, n_neurons, n_unified_synapse_types)
                    - "currents_leak": Leak currents (batch_size, n_steps, n_neurons)
                    - "conductances": Unified conductances (batch_size, n_steps, n_neurons, 2, n_unified_synapse_types)

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

        # Use unified conductance array directly (no more separate g and g_FF)
        # Shape: (batch_size, n_neurons, 2, n_unified_synapse_types)
        g = self.g

        # ==========================
        # Conditionally allocate tracking arrays
        # ==========================

        # Storage for gradient-enabled tensors (if track_gradients=True)
        if self.track_gradients:
            self._tracked_v_list = []
            self._tracked_g_list = []
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
                    self.n_unified_synapse_types,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            all_I = torch.empty(
                (
                    tracking_batch_size,
                    n_steps,
                    self.n_neurons,
                    self.n_unified_synapse_types + 1,
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

        # ── Per-forward-call precomputation ──────────────────────────────
        # Assemble (connection_weights, synapse_kernel) pairs for each cell
        # type.  Init-cached tensors are looked up; learnable parameters are
        # snapshotted here so the time loop only does matmul + broadcast.
        # See _precompute_weight_products() for the full caching strategy.

        # Resolve per-cell-type modes for building forward-pass tensors
        rec_modes, ff_modes = self._resolve_modes()

        def _build_cell_type_tensors(
            prefix, masks, syn_masks, cell_type_indices, weights, sf, modes
        ):
            """Build (mask, syn_mask, conn_w, kernel) lists for one pathway."""
            conn_weights_list = []
            kernels_list = []
            for ki, (k, (mask, syn_mask)) in enumerate(
                zip(cell_type_indices, zip(masks, syn_masks))
            ):
                kernel = getattr(self, f"synapse_kernel_{prefix}_{k}")
                mode = modes[ki]
                if mode is None:
                    conn_w = getattr(self, f"connection_weights_{prefix}_{k}")
                elif mode == "weights":
                    conn_w = weights[mask, :]
                elif mode == "scaling_factors":
                    conn_w = getattr(self, f"connection_weights_{prefix}_{k}")
                    s_factor = sf[k, self.cell_type_indices]
                    kernel = kernel * s_factor[:, None, None]
                elif mode == "fixed":
                    conn_w = getattr(self, f"connection_weights_{prefix}_{k}")
                conn_weights_list.append(conn_w)
                kernels_list.append(kernel)
            return conn_weights_list, kernels_list

        rec_conn_weights, rec_kernels = _build_cell_type_tensors(
            "rec",
            self.rec_masks,
            self.rec_syn_masks,
            self.rec_cell_type_indices,
            self.weights,
            self.scaling_factors,
            rec_modes,
        )
        ff_conn_weights, ff_kernels = _build_cell_type_tensors(
            "ff",
            self.ff_masks,
            self.ff_syn_masks,
            self.ff_cell_type_indices,
            self.weights_FF,
            self.scaling_factors_FF,
            ff_modes,
        )

        rec_pathways = list(
            zip(self.rec_masks, self.rec_syn_masks, rec_conn_weights, rec_kernels)
        )
        ff_pathways = list(
            zip(self.ff_masks, self.ff_syn_masks, ff_conn_weights, ff_kernels)
        )

        for t in range(n_steps):
            self.v, g, s, I, I_leak = self._step_common(
                self.v,
                g,
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
                rec_pathways,
                ff_pathways,
                input_spikes[:, t, :],
            )

            # Store spike output (spikes need gradients for training!)
            all_s[:, t, :] = s

            # Store gradient-enabled tensors (must retain grad on actual tensors in graph)
            if self.track_gradients:
                # Retain gradients on the actual tensors that are part of the computation graph
                # Only call retain_grad() if tensor has requires_grad=True
                if self.v.requires_grad:
                    self.v.retain_grad()
                if g.requires_grad:
                    g.retain_grad()
                if s.requires_grad:
                    s.retain_grad()
                # Store references to these tensors (they will be reassigned in next iteration)
                self._tracked_v_list.append(self.v)
                self._tracked_g_list.append(g)
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
                    all_g[:, t, :, :, :] = g[
                        self.track_batch_idx : self.track_batch_idx + 1, :, :, :
                    ].detach()
                else:
                    # Track all batch elements
                    all_v[:, t, :] = self.v.detach()
                    all_I[:, t, :, :-1] = I.detach()
                    all_I[:, t, :, -1] = I_leak.detach()
                    all_g[:, t, :, :, :] = g.detach()

        # Update internal state variable (unified conductance array)
        self.g = g

        # CRITICAL: Detach state tensors to prevent carrying computation graph to next chunk
        # Without this, chunk N+1 would try to use state from chunk N's (freed) graph
        self.v = self.v.detach()
        self.g = self.g.detach()

        # ==============
        # Return results
        # ==============

        if self.track_variables:
            return {
                "spikes": all_s,
                "voltages": all_v,
                "currents": all_I[:, :, :, : self.n_unified_synapse_types],
                "currents_leak": all_I[:, :, :, -1],
                "conductances": all_g,
            }
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
                - "grad_g": Shape (time, batch, neurons, 2, n_unified_synapse_types) - conductance gradients
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
        grad_g_list = []
        grad_s_list = []

        for v, g, s in zip(
            self._tracked_v_list, self._tracked_g_list, self._tracked_s_list
        ):
            # Extract gradient if it exists, otherwise zeros
            if v.grad is not None:
                grad_v_list.append(v.grad.detach().clone())
            else:
                grad_v_list.append(torch.zeros_like(v))

            if g.grad is not None:
                grad_g_list.append(g.grad.detach().clone())
            else:
                grad_g_list.append(torch.zeros_like(g))

            if s.grad is not None:
                grad_s_list.append(s.grad.detach().clone())
            else:
                grad_s_list.append(torch.zeros_like(s))

        # Stack into tensors (time, batch, ...)
        return {
            "grad_v": torch.stack(grad_v_list, dim=0),
            "grad_g": torch.stack(grad_g_list, dim=0),
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
        rec_pathways: list,
        ff_pathways: list,
        input_t: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Single simulation timestep: spike, voltage, conductance update.

        Args:
            rec_pathways: List of (mask, syn_mask, conn_w, kernel) for recurrent inputs.
            ff_pathways: List of (mask, syn_mask, conn_w, kernel) for feedforward inputs.
            input_t: Feedforward input spikes for this timestep (batch, n_ff).

        Returns:
            (v, g, s, I, I_leak)
        """
        # Compute spikes
        if spike_mode == "deterministic":
            s = SurrGradSpike.apply(v - theta, surrgrad_scale)
        elif spike_mode == "sample":
            s = SampleSpike.apply(v - theta, surrgrad_scale)
        else:
            s = SurrGradSpike.apply(v - theta, surrgrad_scale)

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

        # Inject recurrent conductances from current spikes
        for mask, syn_mask, conn_w, kernel in rec_pathways:
            activation = s[:, mask].detach() @ conn_w
            g[:, :, :, syn_mask] += activation[:, :, None, None] * kernel[None, :, :, :]

        # Inject feedforward conductances from input
        for mask, syn_mask, conn_w, kernel in ff_pathways:
            activation = input_t[:, mask] @ conn_w
            g[:, :, :, syn_mask] += activation[:, :, None, None] * kernel[None, :, :, :]

        return v, g, s, I, I_leak
