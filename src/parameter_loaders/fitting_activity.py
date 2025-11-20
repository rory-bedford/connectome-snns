"""Parameter loader for teacher-student fitting activity training.

For training networks to match target activity patterns without fixed connectivity.
"""

from typing import Dict
from pydantic import BaseModel
from .base_configs import (
    SimulationConfig,
    TrainingConfig,
    Hyperparameters,
    BaseRecurrentLayerConfig,
    BaseFeedforwardLayerConfig,
)


# =============================================================================
# FITTING-ACTIVITY-SPECIFIC CONFIGS
# =============================================================================


class SimulationConfigWithOdours(SimulationConfig):
    """Simulation config extended with odour stimulus parameters."""

    num_odours: int


class OdourInputConfig(BaseModel):
    """Configuration for modulated Poisson input activity.

    Defines baseline and modulated firing rates for input cells responding to odours.
    A fraction of cells are modulated up/down from baseline when odour is present.
    """

    baseline_rate: float
    modulation_rate: float
    modulation_fraction: float

    def get_modulated_rates(self) -> tuple[float, float]:
        """Get the up-modulated and down-modulated rates.

        Returns:
            Tuple of (up_rate, down_rate) in Hz.
        """
        up_rate = self.baseline_rate + self.modulation_rate
        down_rate = self.baseline_rate - self.modulation_rate
        return (up_rate, down_rate)

    def get_n_modulated(self, n_neurons: int) -> int:
        """Get number of neurons modulated in each direction (up or down).

        Args:
            n_neurons: Total number of neurons of this cell type.

        Returns:
            Number of neurons to modulate up (same number will be modulated down).
        """
        return int(n_neurons * self.modulation_fraction / 2.0)


class FittingActivityRecurrentConfig(BaseRecurrentLayerConfig):
    """Recurrent layer config for fitting activity (no topology/weights)."""

    # cell_types, physiology, synapses inherited from BaseRecurrentLayerConfig
    # All methods inherited from BaseRecurrentLayerConfig
    pass


class FittingActivityFeedforwardConfig(BaseFeedforwardLayerConfig):
    """Feedforward layer config for fitting activity (no topology/weights)."""

    # cell_types, synapses inherited from BaseFeedforwardLayerConfig
    # All methods inherited from BaseFeedforwardLayerConfig
    pass


# =============================================================================
# TOP-LEVEL MODELS
# =============================================================================


class TeacherActivityParams(BaseModel):
    """Parameters for generating teacher activity from a trained network.

    Loads pre-trained network structure and generates activity for use as training target.
    Does not include training or optimization parameters.
    """

    simulation: SimulationConfigWithOdours
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig
    odours: Dict[str, OdourInputConfig]


class StudentTrainingParams(BaseModel):
    """Parameters for training student network to match teacher activity.

    Includes training configuration and hyperparameters for optimizing network
    to reproduce target activity patterns.
    """

    simulation: SimulationConfigWithOdours
    training: TrainingConfig
    hyperparameters: Hyperparameters
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig
    odours: Dict[str, OdourInputConfig]
