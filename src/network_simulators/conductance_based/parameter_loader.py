"""Parameter loaders for conductance-based network models.

Pydantic models that directly validate TOML configuration files.
"""

import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


# =============================================================================
# SIMPLE CONFIG MODELS (match TOML structure exactly)
# =============================================================================


class SimulationConfig(BaseModel):
    """Simulation parameters."""

    dt: float
    duration: float
    seed: int
    chunk_size: int


class TrainingConfig(BaseModel):
    """Training parameters."""

    chunks_per_loss: int
    losses_per_update: int
    log_interval: int
    mixed_precision: bool
    plot_size: int


class Targets(BaseModel):
    """Target values."""

    firing_rates_list: List[float] = Field(alias="firing_rates")
    cvs_list: List[float] = Field(alias="cvs")

    @computed_field
    @property
    def firing_rates(self) -> np.ndarray:
        """Target firing rates as numpy array."""
        return np.array(self.firing_rates_list)

    @computed_field
    @property
    def cvs(self) -> np.ndarray:
        """Target CVs as numpy array."""
        return np.array(self.cvs_list)

    class Config:
        populate_by_name = True


class Hyperparameters(BaseModel):
    """Optimization hyperparameters."""

    surrgrad_scale: float
    cv_high_loss: float
    loss_ratio: float
    learning_rate: float


class CellTypesConfig(BaseModel):
    """Cell types."""

    names: List[str]
    proportion_list: List[float] = Field(alias="proportion")

    @computed_field
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

    @computed_field
    @property
    def conn_within(self) -> Optional[np.ndarray]:
        """Within-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_within_list)
            if self.conn_within_list is not None
            else None
        )

    @computed_field
    @property
    def conn_between(self) -> Optional[np.ndarray]:
        """Between-assembly connectivity as numpy array."""
        return (
            np.array(self.conn_between_list)
            if self.conn_between_list is not None
            else None
        )

    @computed_field
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

    @computed_field
    @property
    def w_mu(self) -> np.ndarray:
        """Mean weights as numpy array."""
        return np.array(self.w_mu_list)

    @computed_field
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

    @computed_field
    @property
    def tau_rise(self) -> np.ndarray:
        """Rise time constants as numpy array."""
        return np.array(self.tau_rise_list)

    @computed_field
    @property
    def tau_decay(self) -> np.ndarray:
        """Decay time constants as numpy array."""
        return np.array(self.tau_decay_list)

    @computed_field
    @property
    def E_syn(self) -> np.ndarray:
        """Reversal potentials as numpy array."""
        return np.array(self.E_syn_list)

    @computed_field
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


class FeedforwardConfig(BaseModel):
    """Feedforward layer (matches [feedforward] section in TOML)."""

    cell_types: CellTypesConfig
    topology: TopologyConfig
    weights: WeightsConfig
    activity: Dict[str, ActivityConfig]
    synapses: Dict[str, SynapseConfig]

    @computed_field
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
