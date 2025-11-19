"""Parameter loader for basic conductance-based network simulations.

For simulations without training - just running a network with fixed connectivity.
"""

from typing import Dict
from pydantic import BaseModel
from .base_configs import (
    SimulationConfig,
    TopologyConfig,
    WeightsConfig,
    ActivityConfig,
    BaseRecurrentLayerConfig,
    BaseFeedforwardLayerConfig,
)


# =============================================================================
# LAYER CONFIGS
# =============================================================================


class RecurrentConfig(BaseRecurrentLayerConfig):
    """Recurrent layer (matches [recurrent] section in TOML)."""

    topology: TopologyConfig
    weights: WeightsConfig
    # cell_types, physiology, synapses inherited from BaseRecurrentLayerConfig
    # All methods inherited from BaseRecurrentLayerConfig


class FeedforwardConfig(BaseFeedforwardLayerConfig):
    """Feedforward layer (matches [feedforward] section in TOML)."""

    topology: TopologyConfig
    weights: WeightsConfig
    activity: Dict[str, ActivityConfig]
    # cell_types, synapses inherited from BaseFeedforwardLayerConfig
    # All methods inherited from BaseFeedforwardLayerConfig


# =============================================================================
# TOP-LEVEL MODEL
# =============================================================================


class ConductanceBasedParams(BaseModel):
    """Conductance-based simulation (no training)."""

    simulation: SimulationConfig
    recurrent: RecurrentConfig
    feedforward: FeedforwardConfig
