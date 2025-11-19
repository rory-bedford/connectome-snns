"""Parameter loader for teacher-student fitting activity training.

For training networks to match target activity patterns without fixed connectivity.
"""

from typing import Dict
from pydantic import BaseModel
from .base_configs import (
    SimulationConfig,
    TrainingConfig,
    Hyperparameters,
    ActivityConfig,
    BaseRecurrentLayerConfig,
    BaseFeedforwardLayerConfig,
)


# =============================================================================
# FITTING-ACTIVITY-SPECIFIC CONFIGS
# =============================================================================


class FittingActivityRecurrentConfig(BaseRecurrentLayerConfig):
    """Recurrent layer config for fitting activity (no topology/weights)."""

    # cell_types, physiology, synapses inherited from BaseRecurrentLayerConfig
    # All methods inherited from BaseRecurrentLayerConfig
    pass


class FittingActivityFeedforwardConfig(BaseFeedforwardLayerConfig):
    """Feedforward layer config for fitting activity (includes activity but no topology/weights)."""

    activity: Dict[str, ActivityConfig]
    # cell_types, synapses inherited from BaseFeedforwardLayerConfig
    # All methods inherited from BaseFeedforwardLayerConfig


# =============================================================================
# TOP-LEVEL MODELS
# =============================================================================


class TeacherActivityParams(BaseModel):
    """Parameters for generating teacher activity from a trained network.

    Loads pre-trained network structure and generates activity for use as training target.
    Does not include training or optimization parameters.
    """

    simulation: SimulationConfig
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig


class StudentTrainingParams(BaseModel):
    """Parameters for training student network to match teacher activity.

    Includes training configuration and hyperparameters for optimizing network
    to reproduce target activity patterns.
    """

    simulation: SimulationConfig
    training: TrainingConfig
    hyperparameters: Hyperparameters
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig
