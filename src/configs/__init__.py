"""Configuration modules for spiking neural network simulations.

This module provides a compositional configuration system where scripts load only
the configuration sections they need, rather than requiring monolithic parameter classes.

Example usage:
    import toml
    from configs import SimulationConfig, TrainingConfig
    from configs.conductance_based import RecurrentLayerConfig, FeedforwardLayerConfig

    data = toml.load(params_file)
    simulation = SimulationConfig(**data['simulation'])
    recurrent = RecurrentLayerConfig(**data['recurrent'])
"""

# Shared configs
from .simulation import SimulationConfig, StudentSimulationConfig
from .training import (
    TrainingConfig,
    StudentTrainingConfig,
    EMTrainingConfig,
    LossWeights,
    StudentLossWeights,
    Hyperparameters,
    StudentHyperparameters,
    Targets,
)
from .network import (
    SimpleCellTypesConfig,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    ActivityConfig,
)

# Model-specific configs are imported from their respective modules:
# - configs.conductance_based
# - configs.current_based

__all__ = [
    # Simulation
    "SimulationConfig",
    "StudentSimulationConfig",
    # Training
    "TrainingConfig",
    "StudentTrainingConfig",
    "EMTrainingConfig",
    "LossWeights",
    "StudentLossWeights",
    "Hyperparameters",
    "StudentHyperparameters",
    "Targets",
    # Network
    "SimpleCellTypesConfig",
    "CellTypesConfig",
    "TopologyConfig",
    "WeightsConfig",
    "ActivityConfig",
]
