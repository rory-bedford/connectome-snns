"""Parameter loader for current-based LIF networks.

Loads and validates parameters from TOML configuration files using Pydantic.
Returns structured parameter objects that can be used by simulation scripts.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import toml
from pydantic import BaseModel, Field


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================


class SimulationConfig(BaseModel):
    """Basic simulation timing parameters."""

    dt: float = Field(gt=0, description="Time step in ms")
    duration: float = Field(gt=0, description="Total simulation duration in ms")
    seed: Optional[int] = Field(default=None, description="Random seed")


class Hyperparameters(BaseModel):
    """Hyperparameters for optimization."""

    surrgrad_scale: float = Field(gt=0)


# =============================================================================
# LAYER STRUCTURE (CONNECTOME AND INPUTS)
# =============================================================================


class TopologyConfig(BaseModel):
    """Network topology parameters."""

    num_neurons: int = Field(gt=0)
    num_assemblies: Optional[int] = Field(default=None, ge=0)
    _conn_within: Optional[List[List[float]]] = Field(default=None, alias="conn_within")
    _conn_between: Optional[List[List[float]]] = Field(
        default=None, alias="conn_between"
    )
    _conn_inputs: Optional[List[List[float]]] = Field(default=None, alias="conn_inputs")

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


class CellTypesConfig(BaseModel):
    """Cell type distribution."""

    names: List[str]
    _signs: List[int]
    _proportion: List[float]

    @property
    def signs(self) -> np.ndarray:
        """Return cell type signs as numpy array."""
        return np.array(self._signs)

    @property
    def proportion(self) -> np.ndarray:
        """Return cell type proportions as numpy array."""
        return np.array(self._proportion)

    class Config:
        populate_by_name = True
        fields = {
            "_signs": {"alias": "signs"},
            "_proportion": {"alias": "proportion"},
        }


class WeightsConfig(BaseModel):
    """Synaptic weight parameters."""

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
    """Neuronal physiology parameters (current-based)."""

    tau_mem: float = Field(gt=0)
    tau_syn: float = Field(gt=0)
    R: float = Field(gt=0)
    U_rest: float
    theta: float
    U_reset: float


class ActivityConfig(BaseModel):
    """Input activity parameters."""

    _firing_rates: List[float]

    @property
    def firing_rates(self) -> np.ndarray:
        """Return firing rates as numpy array."""
        return np.array(self._firing_rates)

    class Config:
        populate_by_name = True
        fields = {"_firing_rates": {"alias": "firing_rates"}}


class ConnectomeLayerConfig(BaseModel):
    """Complete connectome (recurrent) layer configuration."""

    topology: TopologyConfig
    cell_types: CellTypesConfig
    weights: WeightsConfig


class InputLayerConfig(BaseModel):
    """Complete input layer configuration."""

    topology: TopologyConfig
    cell_types: CellTypesConfig
    weights: WeightsConfig
    activity: ActivityConfig
    physiology: Dict[str, PhysiologyConfig]


# =============================================================================
# TOP-LEVEL PARAMETER MODEL
# =============================================================================


class CurrentBasedParams(BaseModel):
    """Parameters for current-based simulation."""

    simulation: SimulationConfig
    connectome: ConnectomeLayerConfig
    inputs: InputLayerConfig
    physiology: Dict[str, PhysiologyConfig]
    hyperparameters: Hyperparameters


# =============================================================================
# LOADER FUNCTION
# =============================================================================


def load_current_based_params(toml_path: Path) -> CurrentBasedParams:
    """Load parameters for current-based simulation.

    Args:
        toml_path: Path to current-based-Dp.toml file

    Returns:
        CurrentBasedParams: Validated parameter configuration
    """
    with open(toml_path, "r") as f:
        data = toml.load(f)

    # Parse connectome layer
    connectome = ConnectomeLayerConfig(
        topology=TopologyConfig(**data["connectome"]["topology"]),
        cell_types=CellTypesConfig(**data["connectome"]["cell_types"]),
        weights=WeightsConfig(**data["connectome"]["weights"]),
    )

    # Parse input layer
    input_topology = TopologyConfig(**data["inputs"]["topology"])
    input_cell_types = CellTypesConfig(**data["inputs"]["cell_types"])
    input_weights = WeightsConfig(**data["inputs"]["weights"])
    input_activity = ActivityConfig(**data["inputs"]["activity"])

    input_physiology = {}
    for cell_type in input_cell_types.names:
        input_physiology[cell_type] = PhysiologyConfig(
            **data["inputs"]["physiology"][cell_type]
        )

    inputs = InputLayerConfig(
        topology=input_topology,
        cell_types=input_cell_types,
        weights=input_weights,
        activity=input_activity,
        physiology=input_physiology,
    )

    # Parse top-level physiology for connectome
    physiology = {}
    for cell_type in connectome.cell_types.names:
        physiology[cell_type] = PhysiologyConfig(**data["physiology"][cell_type])

    return CurrentBasedParams(
        simulation=SimulationConfig(**data["simulation"]),
        connectome=connectome,
        inputs=inputs,
        physiology=physiology,
        hyperparameters=Hyperparameters(**data["hyperparameters"]),
    )
