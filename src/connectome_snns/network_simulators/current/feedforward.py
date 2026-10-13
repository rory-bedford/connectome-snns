"""Feedforward-only current-based LIF network."""

from __future__ import annotations

import numpy as np
import torch

from connectome_snns.network_simulators.current.network import (
    CurrentNetwork,
    CurrentSide,
    OptimisableParams,
)


class FeedforwardCurrentLIFNetwork(CurrentNetwork):
    """Current-based LIF neurons driven only by input rows (natural units).

    See ``network_simulators.current.network`` for the dynamics. The state (``v``,
    ``I_syn``) continues across ``forward`` calls; call ``reset_state()`` before an
    independent simulation.
    """

    def __init__(
        self,
        dt: float,
        weights_FF: np.ndarray,
        cell_type_indices: np.ndarray,
        cell_type_indices_FF: np.ndarray,
        cell_params: list[dict],
        synapse_params_FF: list[dict],
        surrgrad_scale: float,
        batch_size: int,
        scaling_factors_FF: np.ndarray | None = None,
        optimisable: OptimisableParams = None,
        feedforward_mask: np.ndarray | None = None,
        track_variables: bool = False,
        track_gradients: bool = False,
        track_batch_idx: int | None = None,
        use_tqdm: bool = True,
    ):
        """
        Args: as ``CurrentLIFNetwork``, for the feedforward side only. Any of the
        scaling-factor modes of ``optimisable`` trains ``scaling_factors_FF``.
        """
        self.optimisable = optimisable
        self.synapse_params_FF = synapse_params_FF
        self.n_synapse_types_FF = len(synapse_params_FF)
        self.n_cell_types_FF = int(cell_type_indices_FF.max()) + 1
        self.n_inputs = int(weights_FF.shape[0])
        super().__init__(
            dt=dt,
            cell_params=cell_params,
            cell_type_indices=cell_type_indices,
            sides=[
                CurrentSide(
                    "_FF", "I_syn", weights_FF, cell_type_indices_FF, synapse_params_FF,
                    scaling_factors_FF, feedforward_mask, optimisable == "weights",
                    optimisable in (
                        "scaling_factors", "scaling_factors_feedforward",
                        "scaling_factors_recurrent",
                    ),
                    recurrent=False,
                )
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
        ``track_variables``, a dict of ``spikes``, ``voltages`` and ``currents_syn``.
        """
        spikes, tracked = self._simulate(input_spikes)
        if tracked is None:
            return spikes
        return {
            "spikes": spikes,
            "voltages": tracked["voltages"],
            "currents_syn": tracked["currents"][0],
        }

    @property
    def weights_FF(self) -> torch.Tensor:
        return self._weights(0)

    @property
    def scaling_factors_FF(self) -> torch.Tensor:
        return self._scaling(0)
