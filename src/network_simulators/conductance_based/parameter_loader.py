"""Parameter loaders for conductance-based network models.

Pydantic models that directly validate TOML configuration files.
"""

import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# SIMPLE CONFIG MODELS (match TOML structure exactly)
# =============================================================================

# Hardcoding synapse types for simplicity
EXCITATORY_SYNAPSE_TYPES = ["AMPA", "NMDA"]
INHIBITORY_SYNAPSE_TYPES = ["GABA_A", "GABA_B"]


class SimulationConfig(BaseModel):
    """Simulation parameters."""

    dt: float
    duration: float
    seed: int
    chunk_size: int

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


class TrainingConfig(BaseModel):
    """Training parameters."""

    chunks_per_update: int
    log_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    plot_size: int
    batch_size: int


class ActivityTargetConfig(BaseModel):
    """Target values for a single cell type."""

    firing_rate: float


class Targets(BaseModel):
    """Target values for all cell types."""

    firing_rate: Dict[str, float]


class Hyperparameters(BaseModel):
    """Optimization hyperparameters."""

    surrgrad_scale: float
    learning_rate: float


class CellTypesConfig(BaseModel):
    """Cell types."""

    names: List[str]
    proportion_list: List[float] = Field(alias="proportion")

    @property
    def proportion(self) -> np.ndarray:
        """Cell type proportions as numpy array."""
        return np.array(self.proportion_list)

    class Config:
        populate_by_name = True


class TopologyConfig(BaseModel):
    """Network topology."""

    num_neurons: int
    num_assemblies: Optional[int] = None
    neurons_per_assembly: Optional[int] = None
    conn_within_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_within"
    )
    conn_between_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_between"
    )
    conn_inputs_list: Optional[List[List[float]]] = Field(
        default=None, alias="conn_inputs"
    )

    @property
    def conn_within(self) -> Optional[np.ndarray]:
        """Within-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_within_list)
            if self.conn_within_list is not None
            else None
        )

    @property
    def conn_between(self) -> Optional[np.ndarray]:
        """Between-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_between_list)
            if self.conn_between_list is not None
            else None
        )

    @property
    def conn_inputs(self) -> Optional[np.ndarray]:
        """Input connectivity as numpy array."""
        return (
            np.array(self.conn_inputs_list)
            if self.conn_inputs_list is not None
            else None
        )

    class Config:
        populate_by_name = True


class WeightsConfig(BaseModel):
    """Weight parameters."""

    w_mu_list: List[List[float]] = Field(alias="w_mu")
    w_sigma_list: List[List[float]] = Field(alias="w_sigma")

    @property
    def w_mu(self) -> np.ndarray:
        """Mean weights as numpy array."""
        return np.array(self.w_mu_list)

    @property
    def w_sigma(self) -> np.ndarray:
        """Weight standard deviations as numpy array."""
        return np.array(self.w_sigma_list)

    class Config:
        populate_by_name = True


class PhysiologyConfig(BaseModel):
    """Neuronal physiology."""

    tau_mem: float
    theta: float
    U_reset: float
    E_L: float
    g_L: float
    tau_ref: float


class SynapseConfig(BaseModel):
    """Synapse parameters."""

    names: List[str]
    tau_rise_list: List[float] = Field(alias="tau_rise")
    tau_decay_list: List[float] = Field(alias="tau_decay")
    E_syn_list: List[float] = Field(alias="E_syn")
    g_bar_list: List[float] = Field(alias="g_bar")

    @property
    def tau_rise(self) -> np.ndarray:
        """Rise time constants as numpy array."""
        return np.array(self.tau_rise_list)

    @property
    def tau_decay(self) -> np.ndarray:
        """Decay time constants as numpy array."""
        return np.array(self.tau_decay_list)

    @property
    def E_syn(self) -> np.ndarray:
        """Reversal potentials as numpy array."""
        return np.array(self.E_syn_list)

    @property
    def g_bar(self) -> np.ndarray:
        """Maximal conductances as numpy array."""
        return np.array(self.g_bar_list)

    class Config:
        populate_by_name = True


class ActivityConfig(BaseModel):
    """Input activity."""

    firing_rate: float


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

    @property
    def firing_rates(self) -> np.ndarray:
        """Firing rates as numpy array."""
        return np.array(
            [self.activity[name].firing_rate for name in self.cell_types.names]
        )

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
# TOP-LEVEL MODELS
# =============================================================================


class ConductanceBasedParams(BaseModel):
    """Conductance-based simulation (no training)."""

    simulation: SimulationConfig
    recurrent: RecurrentConfig
    feedforward: FeedforwardConfig


class HomeostaticPlasticityParams(BaseModel):
    """Homeostatic plasticity training."""

    simulation: SimulationConfig
    training: TrainingConfig
    targets: Targets
    hyperparameters: Hyperparameters
    recurrent: RecurrentConfig
    feedforward: FeedforwardConfig

    @property
    def log_interval_s(self) -> float:
        """Duration of log interval in seconds."""
        return self.training.log_interval * self.simulation.chunk_duration_s

    @property
    def checkpoint_interval_s(self) -> float:
        """Duration of checkpoint interval in seconds."""
        return self.training.checkpoint_interval * self.simulation.chunk_duration_s

    @model_validator(mode="after")
    def validate_checkpoint_alignment(self) -> "HomeostaticPlasticityParams":
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

    @model_validator(mode="after")
    def validate_target_cell_types(self) -> "HomeostaticPlasticityParams":
        """Validate that target keys match recurrent cell types."""
        recurrent_cell_types = set(self.recurrent.cell_types.names)
        target_cell_types = set(self.targets.firing_rate.keys())

        if target_cell_types != recurrent_cell_types:
            missing_in_targets = recurrent_cell_types - target_cell_types
            extra_in_targets = target_cell_types - recurrent_cell_types

            error_msg = "Target cell types must match recurrent cell types."
            if missing_in_targets:
                error_msg += f"\n  Missing in targets: {missing_in_targets}"
            if extra_in_targets:
                error_msg += f"\n  Extra in targets: {extra_in_targets}"

            raise ValueError(error_msg)

        return self
