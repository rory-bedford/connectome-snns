"""Recurrent current-based LIF network."""

from __future__ import annotations

import numpy as np
import torch

from connectome_snns.network_simulators.current.network import (
    CurrentNetwork,
    CurrentSide,
    OptimisableParams,
)


class CurrentLIFNetwork(CurrentNetwork):
    """Recurrent current-based LIF network in natural units (reset 0, threshold 1).

    See ``network_simulators.current.network`` for the dynamics and what
    ``optimisable`` trains. The state (``v``, ``I_syn``, ``I_syn_FF``) continues across
    ``forward`` calls; call ``reset_state()`` before an independent simulation.
    """

    def __init__(
        self,
        dt: float,
        weights: np.ndarray,
        weights_FF: np.ndarray,
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        synapse_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors: np.ndarray | None = None,
        scaling_factors_FF: np.ndarray | None = None,
        optimisable: OptimisableParams = None,
        connectome_mask: np.ndarray | None = None,
        feedforward_mask: np.ndarray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = True,
    ):
        """
        Args:
            dt: Timestep (ms).
            weights, weights_FF: Recurrent ``(n_neurons, n_neurons)`` and feedforward
                ``(n_inputs, n_neurons)`` non-negative weights.
            cell_type_indices, cell_type_indices_FF: Cell type of each neuron / input.
            cell_params: Neuron cell types: 'name', 'cell_id', 'tau_mem'.
            synapse_params, synapse_params_FF: Synapses made by each side's cell types:
                'synapse_id', 'cell_id', 'name', 'tau_syn', 'type' (excitatory or
                inhibitory, the sign of the current).
            surrgrad_scale: Scale of the surrogate gradient.
            batch_size: Number of trials simulated in parallel.
            scaling_factors, scaling_factors_FF: ``(source type, target type)``
                multipliers (default ones).
            optimisable: "weights", "scaling_factors" (both sides),
                "scaling_factors_recurrent", "scaling_factors_feedforward", or None.
            connectome_mask, feedforward_mask: Where weights may be non-zero; required
                when ``optimisable="weights"`` (default: the non-zero weights).
            track_variables: Return voltages and synaptic currents per step.
            track_gradients: Retain per-step gradients (see ``get_tracked_gradients``).
            track_batch_idx: Track only this trial (all if None).
            use_tqdm: Show a progress bar over the steps of each forward call.
        """
        self.optimisable = optimisable
        self.synapse_params = synapse_params
        self.synapse_params_FF = synapse_params_FF
        self.n_synapse_types = len(synapse_params)
        self.n_synapse_types_FF = len(synapse_params_FF)
        self.n_cell_types_FF = int(cell_type_indices_FF.max()) + 1
        self.n_inputs = int(weights_FF.shape[0])
        learn_weights = optimisable == "weights"
        super().__init__(
            dt=dt,
            cell_params=cell_params,
            cell_type_indices=cell_type_indices,
            sides=[
                CurrentSide(
                    "", "I_syn", weights, cell_type_indices, synapse_params,
                    scaling_factors, connectome_mask, learn_weights,
                    optimisable in ("scaling_factors", "scaling_factors_recurrent"),
                    recurrent=True,
                ),
                CurrentSide(
                    "_FF", "I_syn_FF", weights_FF, cell_type_indices_FF, synapse_params_FF,
                    scaling_factors_FF, feedforward_mask, learn_weights,
                    optimisable in ("scaling_factors", "scaling_factors_feedforward"),
                    recurrent=False,
                ),
            ],
            surrgrad_scale=surrgrad_scale,
            batch_size=batch_size,
            track_variables=track_variables,
            track_gradients=track_gradients,
            track_batch_idx=track_batch_idx,
            use_tqdm=use_tqdm,
        )  # fmt: skip

    def forward(
        self, input_spikes: torch.Tensor | np.ndarray
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Simulate ``input_spikes.shape[1]`` steps, continuing from the current state.

        Returns spikes ``(batch_size, n_steps, n_neurons)``; or, with
        ``track_variables``, a dict of ``spikes``, ``voltages``,
        ``currents_syn_recurrent`` and ``currents_syn_feedforward``.
        """
        spikes, tracked = self._simulate(input_spikes)
        if tracked is None:
            return spikes
        recurrent, feedforward = tracked["currents"]
        return {
            "spikes": spikes,
            "voltages": tracked["voltages"],
            "currents_syn_recurrent": recurrent,
            "currents_syn_feedforward": feedforward,
        }

    @property
    def weights(self) -> torch.Tensor:
        return self._weights(0)

    @property
    def weights_FF(self) -> torch.Tensor:
        return self._weights(1)

    @property
    def scaling_factors(self) -> torch.Tensor:
        return self._scaling(0)

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        return self._scaling(1)


CurrentLIFNetwork_IO = CurrentNetwork  # the former base-class name
