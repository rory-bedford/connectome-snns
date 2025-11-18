"""Parameter loader for teacher-student fitting activity training.

For training networks to match target activity patterns without fixed connectivity.
"""

from typing import Dict, List
from pydantic import BaseModel, model_validator
from .base_configs import (
    TrainingConfig,
    Hyperparameters,
    CellTypesConfig,
    PhysiologyConfig,
    SynapseConfig,
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)


# =============================================================================
# FITTING-ACTIVITY-SPECIFIC CONFIGS
# =============================================================================


class FittingActivitySimulationConfig(BaseModel):
    """Simulation parameters for fitting activity."""

    dt: float
    duration: float
    seed: int
    chunk_size: int
    batch_size: int

    @property
    def num_chunks(self) -> int:
        """Total number of chunks"""
        return int(self.duration / (self.chunk_size * self.dt))

    @property
    def chunk_duration_s(self) -> float:
        """Duration of a single chunk in seconds."""
        return self.chunk_size * self.dt / 1000.0

    @property
    def total_duration_s(self) -> float:
        """Total simulation duration in seconds."""
        return self.num_chunks * self.chunk_duration_s


class FittingActivityRecurrentConfig(BaseModel):
    """Recurrent layer config for fitting activity (no topology/weights)."""

    cell_types: CellTypesConfig
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


class FittingActivityFeedforwardConfig(BaseModel):
    """Feedforward layer config for fitting activity (no topology/weights/activity)."""

    cell_types: CellTypesConfig
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


class FittingActivityParams(BaseModel):
    """Teacher-student training for fitting activity."""

    simulation: FittingActivitySimulationConfig
    training: TrainingConfig
    hyperparameters: Hyperparameters
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig

    @property
    def log_interval_s(self) -> float:
        """Duration of log interval in seconds."""
        return self.training.log_interval * self.simulation.chunk_duration_s

    @property
    def checkpoint_interval_s(self) -> float:
        """Duration of checkpoint interval in seconds."""
        return self.training.checkpoint_interval * self.simulation.chunk_duration_s

    @model_validator(mode="after")
    def validate_checkpoint_alignment(self) -> "FittingActivityParams":
        """Validate that simulation duration aligns with checkpoint interval."""
        num_chunks = int(
            self.simulation.duration / (self.simulation.chunk_size * self.simulation.dt)
        )

        if num_chunks % self.training.checkpoint_interval != 0:
            raise ValueError(
                f"Number of chunks ({num_chunks}) must be a multiple of checkpoint_interval "
                f"({self.training.checkpoint_interval}). Either adjust duration or checkpoint_interval."
            )

        return self
