"""Parameter loaders for conductance-based network models.

Pydantic models that validate and load TOML configuration files for different
simulation types.
"""

from .base_configs import (
    SimulationConfig,
    TrainingConfig,
    LossWeights,
    Hyperparameters,
    Targets,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    PhysiologyConfig,
    SynapseConfig,
    ActivityConfig,
    EXCITATORY_SYNAPSE_TYPES,
    INHIBITORY_SYNAPSE_TYPES,
)

from .conductance_based import (
    ConductanceBasedParams,
    RecurrentConfig,
    FeedforwardConfig,
)

from .homeostatic_plasticity import HomeostaticPlasticityParams

from .fitting_activity import (
    FittingActivityParams,
    FittingActivitySimulationConfig,
    FittingActivityRecurrentConfig,
    FittingActivityFeedforwardConfig,
)

__all__ = [
    # Base configs
    "SimulationConfig",
    "TrainingConfig",
    "LossWeights",
    "Hyperparameters",
    "Targets",
    "CellTypesConfig",
    "TopologyConfig",
    "WeightsConfig",
    "PhysiologyConfig",
    "SynapseConfig",
    "ActivityConfig",
    "EXCITATORY_SYNAPSE_TYPES",
    "INHIBITORY_SYNAPSE_TYPES",
    # Conductance-based
    "ConductanceBasedParams",
    "RecurrentConfig",
    "FeedforwardConfig",
    # Homeostatic plasticity
    "HomeostaticPlasticityParams",
    # Fitting activity
    "FittingActivityParams",
    "FittingActivitySimulationConfig",
    "FittingActivityRecurrentConfig",
    "FittingActivityFeedforwardConfig",
]
