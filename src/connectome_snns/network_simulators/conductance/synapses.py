"""Physiology tables: per-neuron cell parameters and the synapse registry.

A conductance-based network has one conductance per *synapse type* per neuron, each
with a rise and a decay component. Synapse types come from the cell types that make
them, on each input side (recurrent neurons, feedforward inputs). The registry gives
every synapse type a single index used throughout the simulator; with ``unify=True``
types with the same signature — (source cell type name, synapse name) — share an index,
so e.g. excitatory AMPA from recurrent and from teacher-forced inputs is one
conductance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

NEURON_PARAMETERS = ("tau_mem", "theta", "U_reset", "E_L", "g_L")
SYNAPSE_PARAMETERS = ("tau_rise", "tau_decay", "E_syn", "g_bar")


def neuron_tables(
    cell_params: list[dict], cell_type_indices: np.ndarray
) -> dict[str, torch.Tensor]:
    """Per-neuron float32 tensors of each cell parameter, plus ``C_m = tau_mem * g_L``."""
    tables = {}
    for name in NEURON_PARAMETERS:
        lookup = np.zeros(len(cell_params), dtype=np.float32)
        for cell in cell_params:
            lookup[cell["cell_id"]] = cell[name]
        tables[name] = torch.from_numpy(lookup[cell_type_indices])
    tables["C_m"] = tables["tau_mem"] * tables["g_L"]
    return tables


@dataclass(frozen=True)
class SynapseRegistry:
    """Synapse types of every input side, under one index.

    Attributes:
        synapses: one parameter dict per synapse type, in index order, each with its
            ``unified_synapse_id`` and ``signature``.
        to_index: per input side, ``synapse_id -> index``.
    """

    synapses: list[dict]
    to_index: list[dict[int, int]]

    @classmethod
    def build(
        cls, sides: list[tuple[list[dict], list[dict]]], unify: bool
    ) -> SynapseRegistry:
        """Index the synapse types of each ``(cell_params, synapse_params)`` side.

        Sides are indexed in order, so the first side's types come first.
        """
        synapses, to_index, by_signature = [], [], {}
        for side, (cells, side_synapses) in enumerate(sides):
            names = {c["cell_id"]: c["name"] for c in cells}
            mapping = {}
            if not unify:  # then each side's indices follow its synapse ids
                side_synapses = sorted(side_synapses, key=lambda s: s["synapse_id"])
            for synapse in side_synapses:
                signature = (names[synapse["cell_id"]], synapse["name"])
                key = signature if unify else (side, synapse["synapse_id"])
                if key not in by_signature:
                    by_signature[key] = len(synapses)
                    synapses.append(
                        {
                            **synapse,
                            "unified_synapse_id": len(synapses),
                            "signature": signature,
                        }
                    )
                mapping[synapse["synapse_id"]] = by_signature[key]
            to_index.append(mapping)
        return cls(synapses, to_index)

    def __len__(self) -> int:
        return len(self.synapses)

    def tables(self) -> dict[str, torch.Tensor]:
        """Float32 tensors of each synapse parameter by index; ``g_clip`` is inf if unset."""
        tables = {
            name: torch.tensor([s[name] for s in self.synapses], dtype=torch.float32)
            for name in SYNAPSE_PARAMETERS
        }
        tables["g_clip"] = torch.tensor(
            [s.get("g_clip", float("inf")) for s in self.synapses], dtype=torch.float32
        )
        return tables

    def made_by(self, side: int, synapse_params: list[dict]) -> dict[int, torch.Tensor]:
        """For each source cell type on ``side``, a bool mask of the types it makes."""
        masks = {}
        for synapse in synapse_params:
            mask = masks.setdefault(
                synapse["cell_id"], torch.zeros(len(self), dtype=torch.bool)
            )
            mask[self.to_index[side][synapse["synapse_id"]]] = True
        return dict(sorted(masks.items()))


def conductance_kernel(
    tau_rise: torch.Tensor, tau_decay: torch.Tensor, g_bar: torch.Tensor
) -> torch.Tensor:
    """``(2, n_synapse_types)`` jump of the rise and decay components per unit input.

    The rise component jumps by ``-g_bar`` and the decay component by ``+g_bar``,
    normalised so the difference-of-exponentials waveform peaks at ``g_bar``.
    """
    r = tau_rise / tau_decay
    peak = r ** (r / (1 - r)) - r ** (1 / (1 - r))
    return torch.stack([-g_bar, g_bar], dim=0) / peak
