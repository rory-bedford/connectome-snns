"""Parameter loaders for conductance-based network models.

This module provides Pydantic models and loaders for reading and validating
conductance-based network configuration from TOML files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel

from typing import Optional
import toml
from pydantic import Field, model_validator


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================


class SimulationConfig(BaseModel):
    """Basic simulation timing parameters."""

    dt: float = Field(gt=0, description="Time step in ms")
    duration: float = Field(gt=0, description="Total simulation duration in ms")
    seed: Optional[int] = Field(default=None, description="Random seed")
    chunk_size: Optional[int] = Field(
        default=None, gt=0, description="Timesteps per chunk"
    )


class TrainingConfig(BaseModel):
    """Training configuration for homeostatic plasticity."""

    chunks_per_loss: int = Field(gt=0)
    losses_per_update: int = Field(gt=0)
    log_interval: int = Field(gt=0)
    mixed_precision: bool = Field(default=True)
    plot_size: int = Field(gt=0)


class Targets(BaseModel):
    """Target values for homeostatic plasticity."""

    firing_rates: float = Field(ge=0)
    cvs: float = Field(ge=0)


class Hyperparameters(BaseModel):
    """Hyperparameters for training/optimization."""

    surrgrad_scale: float = Field(gt=0)
    cv_high_loss: Optional[float] = Field(default=None, gt=0)
    loss_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    learning_rate: Optional[float] = Field(default=None, gt=0)


# =============================================================================
# LAYER STRUCTURE (RECURRENT AND FEEDFORWARD)
# =============================================================================


class TopologyConfig(BaseModel):
    """Network topology configuration."""

    num_neurons: int
    num_assemblies: Optional[int] = None
    _conn_within: Optional[List[List[float]]] = None
    _conn_between: Optional[List[List[float]]] = None
    _conn_inputs: Optional[List[List[float]]] = None

    @property
    def conn_within(self) -> Optional[np.ndarray]:
        """Return connectivity within assemblies as numpy array."""
        return np.array(self._conn_within) if self._conn_within is not None else None

    @property
    def conn_between(self) -> Optional[np.ndarray]:
        """Return connectivity between assemblies as numpy array."""
        return np.array(self._conn_between) if self._conn_between is not None else None

    @property
    def conn_inputs(self) -> Optional[np.ndarray]:
        """Return input connectivity as numpy array."""
        return np.array(self._conn_inputs) if self._conn_inputs is not None else None

    class Config:
        populate_by_name = True
        fields = {
            "_conn_within": {"alias": "conn_within"},
            "_conn_between": {"alias": "conn_between"},
            "_conn_inputs": {"alias": "conn_inputs"},
        }


class CellTypesConfig(BaseModel):
    """Cell type configuration."""

    names: List[str]
    _proportion: List[float]

    @property
    def proportion(self) -> np.ndarray:
        """Return cell type proportions as numpy array."""
        return np.array(self._proportion)

    class Config:
        populate_by_name = True
        fields = {"_proportion": {"alias": "proportion"}}


class WeightsConfig(BaseModel):
    """Synaptic weight configuration."""

    _w_mu: List[List[float]]
    _w_sigma: List[List[float]]

    @property
    def w_mu(self) -> np.ndarray:
        """Return mean weights as numpy array."""
        return np.array(self._w_mu)

    @property
    def w_sigma(self) -> np.ndarray:
        """Return weight standard deviations as numpy array."""
        return np.array(self._w_sigma)

    class Config:
        populate_by_name = True
        fields = {
            "_w_mu": {"alias": "w_mu"},
            "_w_sigma": {"alias": "w_sigma"},
        }


class PhysiologyConfig(BaseModel):
    """Neuronal physiology parameters (conductance-based)."""

    tau_mem: float = Field(gt=0)
    theta: float
    U_reset: float
    E_L: float
    g_L: float = Field(gt=0)
    tau_ref: float = Field(ge=0)


class SynapseConfig(BaseModel):
    """Synapse parameters (conductance-based)."""

    names: List[str]
    tau_rise: List[float]
    tau_decay: List[float]
    E_syn: List[float]
    g_bar: List[float]


class ActivityConfig(BaseModel):
    """Input activity configuration."""

    _firing_rates: List[float]

    @property
    def firing_rates(self) -> np.ndarray:
        """Return firing rates as numpy array."""
        return np.array(self._firing_rates)

    class Config:
        populate_by_name = True
        fields = {"_firing_rates": {"alias": "firing_rates"}}


class RecurrentLayerConfig(BaseModel):
    """Configuration for recurrent layer."""

    topology: TopologyConfig
    cell_types: CellTypesConfig
    weights: WeightsConfig
    physiology: Dict[str, Dict[str, float]]  # cell_type -> {param: value}
    synapses: Dict[
        str, Dict[str, Dict[str, float]]
    ]  # cell_type -> synapse_name -> {param: value}

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology config to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_mem', 'theta',
            'U_reset', 'E_L', 'g_L', 'tau_ref'.
        """
        cell_params = []
        for cell_type_name, physiology in self.physiology.items():
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": len(cell_params),
                    "tau_mem": physiology["tau_mem"],
                    "theta": physiology["theta"],
                    "U_reset": physiology["U_reset"],
                    "E_L": physiology["E_L"],
                    "g_L": physiology["g_L"],
                    "tau_ref": physiology["tau_ref"],
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
        for cell_type_name, synapse_dict in self.synapses.items():
            for synapse_name, synapse_props in synapse_dict.items():
                synapse_params.append(
                    {
                        "name": synapse_name,
                        "synapse_id": len(synapse_params),
                        "cell_id": next(
                            i
                            for i, cp in enumerate(cell_params)
                            if cp["name"] == cell_type_name
                        ),
                        "tau_rise": synapse_props["tau_rise"],
                        "tau_decay": synapse_props["tau_decay"],
                        "E_syn": synapse_props["E_syn"],
                        "g_bar": synapse_props["g_bar"],
                    }
                )
        return synapse_params


class FeedforwardLayerConfig(BaseModel):
    """Configuration for feedforward layer."""

    topology: TopologyConfig
    cell_types: CellTypesConfig
    weights: WeightsConfig
    activity: ActivityConfig
    synapses: Dict[
        str, Dict[str, Dict[str, float]]
    ]  # cell_type -> synapse_name -> {param: value}

    def get_cell_params(self) -> List[Dict[str, int | str]]:
        """Convert cell types to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id'.
        """
        cell_params = []
        for cell_type_name in self.cell_types.names:
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": len(cell_params),
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
        for cell_type_name, synapse_dict in self.synapses.items():
            for synapse_name, synapse_props in synapse_dict.items():
                synapse_params.append(
                    {
                        "name": synapse_name,
                        "synapse_id": len(synapse_params),
                        "cell_id": next(
                            i
                            for i, cp in enumerate(cell_params)
                            if cp["name"] == cell_type_name
                        ),
                        "tau_rise": synapse_props["tau_rise"],
                        "tau_decay": synapse_props["tau_decay"],
                        "E_syn": synapse_props["E_syn"],
                        "g_bar": synapse_props["g_bar"],
                    }
                )
        return synapse_params


# =============================================================================
# TOP-LEVEL PARAMETER MODELS
# =============================================================================


class ConductanceBasedParams(BaseModel):
    """Parameters for standard conductance-based simulation."""

    simulation: SimulationConfig
    recurrent: RecurrentLayerConfig
    feedforward: FeedforwardLayerConfig


class HomeostaticPlasticityParams(BaseModel):
    """Parameters for homeostatic plasticity training."""

    simulation: SimulationConfig
    training: TrainingConfig
    targets: Targets
    hyperparameters: Hyperparameters
    recurrent: RecurrentLayerConfig
    feedforward: FeedforwardLayerConfig

    @model_validator(mode="after")
    def validate_duration_alignment(self):
        """Validate that duration is an integer number of losses_per_update."""
        # Calculate chunk duration in ms
        chunk_duration_ms = self.simulation.chunk_size * self.simulation.dt

        # Calculate loss duration (chunks_per_loss * chunk_duration)
        loss_duration_ms = self.training.chunks_per_loss * chunk_duration_ms

        # Calculate update duration (losses_per_update * loss_duration)
        update_duration_ms = self.training.losses_per_update * loss_duration_ms

        # Check if duration is an integer multiple of update_duration
        if not np.isclose(self.simulation.duration % update_duration_ms, 0.0):
            raise ValueError(
                f"simulation.duration ({self.simulation.duration}ms) must be an integer "
                f"multiple of losses_per_update * chunks_per_loss * chunk_size * dt "
                f"({update_duration_ms}ms). Got remainder: {self.simulation.duration % update_duration_ms}ms"
            )

        return self


# =============================================================================
# LOADER FUNCTIONS
# =============================================================================


def load_conductance_based_params(toml_path: Path) -> ConductanceBasedParams:
    """Load parameters for standard conductance-based simulation.

    Args:
        toml_path: Path to conductance-based-Dp.toml file

    Returns:
        ConductanceBasedParams: Validated parameter configuration
    """
    with open(toml_path, "r") as f:
        data = toml.load(f)

    return ConductanceBasedParams(
        simulation=SimulationConfig(**data["simulation"]),
        recurrent=_parse_recurrent_layer(data["recurrent"]),
        feedforward=_parse_feedforward_layer(data["feedforward"]),
    )


def load_homeostatic_plasticity_params(toml_path: Path) -> HomeostaticPlasticityParams:
    """Load parameters for homeostatic plasticity training.

    Args:
        toml_path: Path to homeostatic-plasticity.toml file

    Returns:
        HomeostaticPlasticityParams: Validated parameter configuration
    """
    with open(toml_path, "r") as f:
        data = toml.load(f)

    return HomeostaticPlasticityParams(
        simulation=SimulationConfig(**data["simulation"]),
        training=TrainingConfig(**data["training"]),
        targets=Targets(**data["targets"]),
        hyperparameters=Hyperparameters(**data["hyperparameters"]),
        recurrent=_parse_recurrent_layer(data["recurrent"]),
        feedforward=_parse_feedforward_layer(data["feedforward"]),
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _parse_recurrent_layer(data: Dict) -> RecurrentLayerConfig:
    """Parse recurrent layer from TOML data."""
    topology = TopologyConfig(**data["topology"])
    cell_types = CellTypesConfig(**data["cell_types"])
    weights = WeightsConfig(**data["weights"])

    physiology = {}
    for cell_type in cell_types.names:
        physiology[cell_type] = PhysiologyConfig(**data["physiology"][cell_type])

    synapses = {}
    for cell_type in cell_types.names:
        synapses[cell_type] = SynapseConfig(**data["synapses"][cell_type])

    return RecurrentLayerConfig(
        topology=topology,
        cell_types=cell_types,
        weights=weights,
        physiology=physiology,
        synapses=synapses,
    )


def _parse_feedforward_layer(data: Dict) -> FeedforwardLayerConfig:
    """Parse feedforward layer from TOML data."""
    topology = TopologyConfig(**data["topology"])
    cell_types = CellTypesConfig(**data["cell_types"])
    weights = WeightsConfig(**data["weights"])

    activity = {}
    for cell_type in cell_types.names:
        activity[cell_type] = ActivityConfig(**data["activity"][cell_type])

    synapses = {}
    for cell_type in cell_types.names:
        synapses[cell_type] = SynapseConfig(**data["synapses"][cell_type])

    return FeedforwardLayerConfig(
        topology=topology,
        cell_types=cell_types,
        weights=weights,
        activity=activity,
        synapses=synapses,
    )
