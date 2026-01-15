"""Defines a straightforward simulator of feedforward-only current-based LIF network"""

import numpy as np
import torch
from typing import List
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.feedforward_conductance_based.model_init import (
    FeedforwardConductanceLIFNetwork_IO,
)
from training_utils.surrogate_gradients import SurrGradSpike

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
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Simulate the feedforward network for a given number of time steps.

        Internal state (v, g_FF) is updated in-place and continues across calls.
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
                    - "currents_feedforward": Feedforward synaptic currents (batch_size, n_steps, n_neurons, n_synapse_types_FF)
                    - "currents_leak": Leak currents (batch_size, n_steps, n_neurons)
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

        # ==============
        # Run simulation
        # ==============

        # Gather cached weight tensors into lists once (outside the loop)
        cached_ff = [
            getattr(self, f"cached_ff_{i}")
            for i in range(len(self.cached_weights_ff_masks))
        ]

        iterator = range(n_steps)
        if self.use_tqdm:
            iterator = tqdm(
                iterator, desc="Simulating feedforward network", unit="step"
            )

        for t in iterator:
            # Call the appropriate _step method based on optimization mode
            if self.optimisable is None:
                self.v, self.g_FF, s, I, I_leak = self._step_inference(
                    self.v,
                    self.g_FF,
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
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    cached_ff,
                )
            elif self.optimisable == "weights":
                self.v, self.g_FF, s, I, I_leak = self._step_optimize_weights(
                    self.v,
                    self.g_FF,
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
                    self.weights_FF,
                    self.g_mins,
                    self.g_maxs,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )
            elif self.optimisable == "scaling_factors":
                self.v, self.g_FF, s, I, I_leak = self._step_optimize_scaling_factors(
                    self.v,
                    self.g_FF,
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
                    self.scaling_factors_FF,
                    self.cell_type_indices,
                    self.g_mins,
                    self.g_maxs,
                    self.cached_weights_ff_masks,
                    self.cached_weights_ff_syn_masks,
                    self.cached_weights_ff_indices,
                    cached_ff,
                )

            # Store spike output (spikes need gradients for training!)
            all_s[:, t, :] = s

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
        # TEMPORARILY DISABLED FOR GRADIENT DEBUGGING
        # self.v = self.v.detach()
        # self.g_FF = self.g_FF.detach()

        # ==============
        # Return results
        # ==============

        if self.track_variables:
            return {
                "spikes": all_s,
                "voltages": all_v,
                "currents_feedforward": all_I[:, :, :, : self.n_synapse_types_FF],
                "currents_leak": all_I[:, :, :, -1],
                "conductances_feedforward": all_g,
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

        # Update conductances (feedforward only) - everything precomputed
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
        weights_FF: torch.Tensor,
        g_mins: torch.Tensor,
        g_maxs: torch.Tensor,
        masks_ff: List[torch.Tensor],
        syn_masks_ff: List[torch.Tensor],
        indices_ff: List[int],
        cached_weights_ff: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimized _step for training weights (optimisable="weights").

        Uses precomputed scaling_factors * g_scale.
        Weights remain dynamic for gradient flow.
        """
        # Decay conductances
        g *= alpha

        # Update conductances (feedforward only)
        for mask, syn_mask, k, cached in zip(
            masks_ff, syn_masks_ff, indices_ff, cached_weights_ff
        ):
            g[:, :, :, syn_mask] += torch.einsum(
                "bi,ijkl->bjkl",
                input_spikes_t[:, mask].float(),
                weights_FF[mask, :][:, :, None, None] * cached[None, :, :, :],
            )

        # Clip conductances
        g = torch.clamp(g, min=g_mins[None, None, :, :], max=g_maxs[None, None, :, :])

        # Compute currents
        I = g.sum(dim=2) * (v[:, :, None] - E_syn[None, None, :])
        I_leak = (v - E_L) * (1 - beta) * C_m / dt

        # Update voltage (without reset)
        v_temp = v - (I.sum(dim=2) + I_leak) * dt / C_m

        # CHANGE: Check spike on v_temp (AFTER update) not v (BEFORE update)
        s = SurrGradSpike.apply(v_temp - theta, surrgrad_scale)

        # Apply reset to v_temp
        v = v_temp * (1 - s) + U_reset * s.detach()

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
        scaling_factors_FF: torch.Tensor,
        cell_type_indices: torch.Tensor,
        g_mins: torch.Tensor,
        g_maxs: torch.Tensor,
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

        # Update conductances (feedforward only) - scaling_factors stay dynamic
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
