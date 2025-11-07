"""Parameter loaders for conductance-based network models.

Pydantic models that directly validate TOML configuration files.
"""

import numpy as np
from typing import Dict, List, Optional
from pydantic import BaseModel


# =============================================================================
# SIMPLE CONFIG MODELS (match TOML structure exactly)
# =============================================================================


class SimulationConfig(BaseModel):
    """Simulation parameters."""

    dt: float
    duration: float
    seed: int
    chunk_size: int
    num_chunks: int
    record_v_m: bool


class TrainingConfig(BaseModel):
    """Training parameters."""

    chunks_per_loss: int
    losses_per_update: int
    log_interval: int
    mixed_precision: bool
    plot_size: int


class Targets(BaseModel):
    """Target values."""

    _firing_rates: List[float]
    _cvs: List[float]

    @property
    def firing_rates(self) -> np.ndarray:
        return np.array(self._firing_rates)

    @property
    def cvs(self) -> np.ndarray:
        return np.array(self._cvs)

    class Config:
        populate_by_name = True
        fields = {
            "_firing_rates": {"alias": "firing_rates"},
            "_cvs": {"alias": "cvs"},
        }


class Hyperparameters(BaseModel):
    """Optimization hyperparameters."""

    surrgrad_scale: float
    cv_high_loss: float
    loss_ratio: float
    learning_rate: float


class CellTypesConfig(BaseModel):
    """Cell types."""

    names: List[str]
    _proportion: List[float]

    @property
    def proportion(self) -> np.ndarray:
        return np.array(self._proportion)

    class Config:
        populate_by_name = True
        fields = {"_proportion": {"alias": "proportion"}}


class TopologyConfig(BaseModel):
    """Network topology."""

    num_neurons: int
    num_assemblies: Optional[int] = None
    neurons_per_assembly: Optional[int] = None
    _conn_within: Optional[List[List[float]]] = None
    _conn_between: Optional[List[List[float]]] = None
    _conn_inputs: Optional[List[List[float]]] = None

    @property
    def conn_within(self) -> Optional[np.ndarray]:
        return np.array(self._conn_within) if self._conn_within is not None else None

    @property
    def conn_between(self) -> Optional[np.ndarray]:
        return np.array(self._conn_between) if self._conn_between is not None else None

    @property
    def conn_inputs(self) -> Optional[np.ndarray]:
        return np.array(self._conn_inputs) if self._conn_inputs is not None else None

    class Config:
        populate_by_name = True
        fields = {
            "_conn_within": {"alias": "conn_within"},
            "_conn_between": {"alias": "conn_between"},
            "_conn_inputs": {"alias": "conn_inputs"},
        }


class WeightsConfig(BaseModel):
    """Weight parameters."""

    _w_mu: List[List[float]]
    _w_sigma: List[List[float]]

    @property
    def w_mu(self) -> np.ndarray:
        return np.array(self._w_mu)

    @property
    def w_sigma(self) -> np.ndarray:
        return np.array(self._w_sigma)

    class Config:
        populate_by_name = True
        fields = {
            "_w_mu": {"alias": "w_mu"},
            "_w_sigma": {"alias": "w_sigma"},
        }


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
    _tau_rise: List[float]
    _tau_decay: List[float]
    _E_syn: List[float]
    _g_bar: List[float]

    @property
    def tau_rise(self) -> np.ndarray:
        return np.array(self._tau_rise)

    @property
    def tau_decay(self) -> np.ndarray:
        return np.array(self._tau_decay)

    @property
    def E_syn(self) -> np.ndarray:
        return np.array(self._E_syn)

    @property
    def g_bar(self) -> np.ndarray:
        return np.array(self._g_bar)

    class Config:
        populate_by_name = True
        fields = {
            "_tau_rise": {"alias": "tau_rise"},
            "_tau_decay": {"alias": "tau_decay"},
            "_E_syn": {"alias": "E_syn"},
            "_g_bar": {"alias": "g_bar"},
        }


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

    @property
    def firing_rates(self) -> np.ndarray:
        """Extract firing rates as numpy array."""
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
