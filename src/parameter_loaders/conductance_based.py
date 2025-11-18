"""Parameter loader for basic conductance-based network simulations.

For simulations without training - just running a network with fixed connectivity.
"""

from typing import Dict, List
from pydantic import BaseModel
from .base_configs import (
    SimulationConfig,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    PhysiologyConfig,
    SynapseConfig,
    ActivityConfig,
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)


# =============================================================================
# LAYER CONFIGS
# =============================================================================


class RecurrentConfig(BaseModel):
    """Recurrent layer (matches [recurrent] section in TOML)."""

    cell_types: CellTypesConfig
    topology: TopologyConfig
    weights: WeightsConfig
    physiology: Dict[str, PhysiologyConfig]
    synapses: Dict[str, SynapseConfig]

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_mem', 'theta',
            'U_reset', 'E_L', 'g_L', 'tau_ref'.
        """
        cell_params = []
        for i, cell_type_name in enumerate(self.cell_types.names):
            physiology = self.physiology[cell_type_name]
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": i,
                    "tau_mem": physiology.tau_mem,
                    "theta": physiology.theta,
                    "U_reset": physiology.U_reset,
                    "E_L": physiology.E_L,
                    "g_L": physiology.g_L,
                    "tau_ref": physiology.tau_ref,
                }
            )
        return cell_params

    def get_synapse_params(self) -> List[Dict[str, float | int | str]]:
        """Convert synapse config to list of synapse parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'synapse_id', 'cell_id',
            'tau_rise', 'tau_decay', 'E_syn', 'g_bar'.
        """
        cell_params = self.get_cell_params()
        synapse_params = []
        for cell_type_name, synapse_config in self.synapses.items():
            cell_id = next(
                cp["cell_id"] for cp in cell_params if cp["name"] == cell_type_name
            )
            for i, synapse_name in enumerate(synapse_config.names):
                synapse_params.append(
                    {
                        "name": synapse_name,
                        "synapse_id": len(synapse_params),
                        "cell_id": cell_id,
                        "tau_rise": synapse_config.tau_rise[i],
                        "tau_decay": synapse_config.tau_decay[i],
                        "E_syn": synapse_config.E_syn[i],
                        "g_bar": synapse_config.g_bar[i],
                    }
                )
        return synapse_params

    def get_g_bar_by_type(self) -> Dict[str, float]:
        """Get total maximal conductance for each cell type.

        Returns:
            Dict mapping cell type names to total g_bar values.
        """
        return {
            cell_type: float(sum(synapse_config.g_bar))
            for cell_type, synapse_config in self.synapses.items()
        }

    def get_neuron_params_for_plotting(self) -> Dict[int, Dict[str, float | str]]:
        """Get neuron parameters formatted for visualization functions.

        Returns:
            Dict mapping cell type indices to parameter dicts with keys:
            'threshold', 'rest', 'name', 'sign'.
        """
        return {
            idx: {
                "threshold": self.physiology[cell_name].theta,
                "rest": self.physiology[cell_name].U_reset,
                "name": cell_name,
                "sign": 1 if "excit" in cell_name.lower() else -1,
            }
            for idx, cell_name in enumerate(self.cell_types.names)
        }

    def get_synapse_names(self) -> Dict[str, List[str]]:
        """Get synapse names for each cell type.

        Returns:
            Dict mapping cell type names to lists of synapse names.
        """
        return {
            cell_type_name: synapse_config.names
            for cell_type_name, synapse_config in self.synapses.items()
        }

    def get_excitatory_synapse_indices(self) -> List[int]:
        """Get indices of excitatory synapses across all cell types.

        Returns:
            List of synapse indices (flattened across all cell types) that are excitatory.
        """
        excitatory_indices = []
        synapse_idx = 0
        for cell_type_name in self.cell_types.names:
            if cell_type_name in self.synapses:
                synapse_config = self.synapses[cell_type_name]
                for syn_name in synapse_config.names:
                    if syn_name in EXCITATORY_SYNAPSE_TYPES:
                        excitatory_indices.append(synapse_idx)
                    synapse_idx += 1
        return excitatory_indices

    def get_inhibitory_synapse_indices(self) -> List[int]:
        """Get indices of inhibitory synapses across all cell types.

        Returns:
            List of synapse indices (flattened across all cell types) that are inhibitory.
        """
        inhibitory_indices = []
        synapse_idx = 0
        for cell_type_name in self.cell_types.names:
            if cell_type_name in self.synapses:
                synapse_config = self.synapses[cell_type_name]
                for syn_name in synapse_config.names:
                    if syn_name in INHIBITORY_SYNAPSE_TYPES:
                        inhibitory_indices.append(synapse_idx)
                    synapse_idx += 1
        return inhibitory_indices


class FeedforwardConfig(BaseModel):
    """Feedforward layer (matches [feedforward] section in TOML)."""

    cell_types: CellTypesConfig
    topology: TopologyConfig
    weights: WeightsConfig
    activity: Dict[str, ActivityConfig]
    synapses: Dict[str, SynapseConfig]

    def get_cell_params(self) -> List[Dict[str, int | str]]:
        """Convert cell types to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id'.
        """
        return [
            {"name": name, "cell_id": i} for i, name in enumerate(self.cell_types.names)
        ]

    def get_synapse_params(self) -> List[Dict[str, float | int | str]]:
        """Convert synapse config to list of synapse parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'synapse_id', 'cell_id',
            'tau_rise', 'tau_decay', 'E_syn', 'g_bar'.
        """
        cell_params = self.get_cell_params()
        synapse_params = []
        for cell_type_name, synapse_config in self.synapses.items():
            cell_id = next(
                cp["cell_id"] for cp in cell_params if cp["name"] == cell_type_name
            )
            for i, synapse_name in enumerate(synapse_config.names):
                synapse_params.append(
                    {
                        "name": synapse_name,
                        "synapse_id": len(synapse_params),
                        "cell_id": cell_id,
                        "tau_rise": synapse_config.tau_rise[i],
                        "tau_decay": synapse_config.tau_decay[i],
                        "E_syn": synapse_config.E_syn[i],
                        "g_bar": synapse_config.g_bar[i],
                    }
                )
        return synapse_params

    def get_g_bar_by_type(self) -> Dict[str, float]:
        """Get total maximal conductance for each cell type.

        Returns:
            Dict mapping cell type names to total g_bar values.
        """
        return {
            cell_type: float(sum(synapse_config.g_bar))
            for cell_type, synapse_config in self.synapses.items()
        }

    def get_synapse_names(self) -> Dict[str, List[str]]:
        """Get synapse names for each cell type.

        Returns:
            Dict mapping cell type names to lists of synapse names.
        """
        return {
            cell_type_name: synapse_config.names
            for cell_type_name, synapse_config in self.synapses.items()
        }

    def get_excitatory_synapse_indices(self) -> List[int]:
        """Get indices of excitatory synapses across all cell types.

        Returns:
            List of synapse indices (flattened across all cell types) that are excitatory.
        """
        excitatory_indices = []
        synapse_idx = 0
        for cell_type_name in self.cell_types.names:
            if cell_type_name in self.synapses:
                synapse_config = self.synapses[cell_type_name]
                for syn_name in synapse_config.names:
                    if syn_name in EXCITATORY_SYNAPSE_TYPES:
                        excitatory_indices.append(synapse_idx)
                    synapse_idx += 1
        return excitatory_indices

    def get_inhibitory_synapse_indices(self) -> List[int]:
        """Get indices of inhibitory synapses across all cell types.

        Returns:
            List of synapse indices (flattened across all cell types) that are inhibitory.
        """
        inhibitory_indices = []
        synapse_idx = 0
        for cell_type_name in self.cell_types.names:
            if cell_type_name in self.synapses:
                synapse_config = self.synapses[cell_type_name]
                for syn_name in synapse_config.names:
                    if syn_name in INHIBITORY_SYNAPSE_TYPES:
                        inhibitory_indices.append(synapse_idx)
                    synapse_idx += 1
        return inhibitory_indices


# =============================================================================
# TOP-LEVEL MODEL
# =============================================================================


class ConductanceBasedParams(BaseModel):
    """Conductance-based simulation (no training)."""

    simulation: SimulationConfig
    recurrent: RecurrentConfig
    feedforward: FeedforwardConfig
