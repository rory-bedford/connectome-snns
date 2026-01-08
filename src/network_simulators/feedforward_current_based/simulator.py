"""Defines a straightforward simulator of feedforward-only current-based LIF network"""

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from network_simulators.feedforward_current_based.model_init import (
    FeedforwardCurrentLIFNetwork_IO,
)
from optimisation.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class FeedforwardCurrentLIFNetwork(FeedforwardCurrentLIFNetwork_IO):
    """
    Feedforward-only current-based LIF network simulator.

    This simulator maintains internal state variables (v, I_syn) that automatically
    continue across forward() calls, enabling efficient chunked simulation of long
    time series without explicit state management.

    Uses natural units: membrane potential varies from 0 (reset) to 1 (threshold), R=1.

    State Management:
        - Internal state automatically continues between forward() calls
        - Call reset_state() before starting independent simulations
        - Call reset_state(batch_size=N) to change batch size

    Tracking Modes:
        - track_variables=False (default): Returns only spikes, minimal memory
        - track_variables=True: Returns full dict with all variables for analysis/visualization

    Example:
        >>> # Continuous simulation across chunks
        >>> model = FeedforwardCurrentLIFNetwork(..., batch_size=10, track_variables=False)
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
        Simulate the feedforward network for a given number of time steps.

        Internal state (v, I_syn) is updated in-place and continues across calls.
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
                    - "currents_syn": Synaptic currents (batch_size, n_steps, n_neurons)

        Raises:
            ValueError: If input_spikes batch size doesn't match self.batch_size
        """
        # Convert to tensor if needed
        if isinstance(input_spikes, np.ndarray):
            input_spikes = torch.from_numpy(input_spikes).float().to(self.device)

        # Validate inputs
        self._validate_forward(input_spikes)

        n_steps = input_spikes.shape[1]

        # ======================================
        # Conditionally allocate tracking arrays
        # ======================================

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
            all_I_syn = torch.empty(
                (tracking_batch_size, n_steps, self.n_neurons, self.n_synapse_types_FF),
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
            # Compute spikes (threshold at 1.0 in natural units)
            s = SurrGradSpike.apply(self.v - 1.0, self.surrgrad_scale)

            # Update synaptic currents: I_syn(t+1) = alpha_syn_FF * I_syn(t) + h(t)
            # Decay all synapse types
            # alpha_syn_FF has shape (n_synapse_types_FF,)
            # I_syn has shape (batch_size, n_neurons, n_synapse_types_FF)
            self.I_syn = self.alpha_syn_FF[None, None, :] * self.I_syn

            # Add input currents per synapse type (loop over input cell types)
            for mask, syn_mask, weights in zip(
                self.cached_weights_ff_masks,
                self.cached_weights_ff_syn_masks,
                cached_ff,
            ):
                # input_spikes[:, t, mask] has shape (batch_size, n_inputs_in_type)
                # weights has shape (n_inputs_in_type, n_neurons, n_synapse_types_in_mask)
                # Result: (batch_size, n_neurons, n_synapse_types_in_mask)
                h = torch.einsum(
                    "bi,ijk->bjk",
                    input_spikes[:, t, mask].float(),
                    weights,
                )

                # Apply scaling factors if optimizing them
                if self.optimisable in [
                    "scaling_factors",
                    "scaling_factors_feedforward",
                ]:
                    # Get the cell_id for this input type
                    cell_id = int(self.cell_type_indices_FF[mask][0])
                    # scaling_factors_FF has shape (n_cell_types_FF, n_cell_types)
                    # Apply scaling per neuron based on target cell type
                    scaling = self.scaling_factors_FF[
                        cell_id, self.cell_type_indices
                    ]  # (n_neurons,)
                    h = h * scaling[None, :, None]

                # Add to appropriate synapse types
                self.I_syn[:, :, syn_mask] += h

            # Sum currents across all synapse types with signs applied
            # sign_FF has shape (n_synapse_types_FF,) with +1 for excitatory, -1 for inhibitory
            # I_syn has shape (batch_size, n_neurons, n_synapse_types_FF)
            # Shape: (batch_size, n_neurons)
            I_total = (self.I_syn * self.sign_FF[None, None, :]).sum(dim=2)

            # Update membrane potential: v(t+1) = (beta_mem * v(t) + I_total(t)) * (1 - s(t))
            # Reset mask: (1 - s) zeros out voltage for spiking neurons
            self.v = (self.beta_mem[None, :] * self.v + I_total) * (1.0 - s.detach())

            # Store spike output (spikes need gradients for training!)
            all_s[:, t, :] = s

            # Conditionally store other variables (detach these for logging only)
            if self.track_variables:
                if self.track_batch_idx is not None:
                    # Only track specified batch index
                    all_v[:, t, :] = self.v[
                        self.track_batch_idx : self.track_batch_idx + 1, :
                    ].detach()
                    all_I_syn[:, t, :, :] = self.I_syn[
                        self.track_batch_idx : self.track_batch_idx + 1, :, :
                    ].detach()
                else:
                    # Track all batch elements
                    all_v[:, t, :] = self.v.detach()
                    all_I_syn[:, t, :, :] = self.I_syn.detach()

        # CRITICAL: Detach state tensors to prevent carrying computation graph to next chunk
        # Without this, chunk N+1 would try to use state from chunk N's (freed) graph
        self.v = self.v.detach()
        self.I_syn = self.I_syn.detach()

        # ==============
        # Return results
        # ==============

        if self.track_variables:
            return {
                "spikes": all_s,
                "voltages": all_v,
                "currents_syn": all_I_syn,
            }
        else:
            return all_s
