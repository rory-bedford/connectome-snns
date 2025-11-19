"""Parameter loader for teacher-student fitting activity training.

For training networks to match target activity patterns without fixed connectivity.
"""

from pydantic import BaseModel, model_validator
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


class FittingActivityRecurrentConfig(BaseRecurrentLayerConfig):
    """Recurrent layer config for fitting activity (no topology/weights)."""

    # cell_types, physiology, synapses inherited from BaseRecurrentLayerConfig
    # All methods inherited from BaseRecurrentLayerConfig
    pass


class FittingActivityFeedforwardConfig(BaseFeedforwardLayerConfig):
    """Feedforward layer config for fitting activity (no topology/weights/activity)."""

    # cell_types, synapses inherited from BaseFeedforwardLayerConfig
    # All methods inherited from BaseFeedforwardLayerConfig
    pass


# =============================================================================
# TOP-LEVEL MODEL
# =============================================================================


class FittingActivityParams(BaseModel):
    """Teacher-student training for fitting activity.

    Supports both teacher generation (without training/hyperparameters) and
    student training (with training/hyperparameters).
    """

    simulation: SimulationConfig
    training: TrainingConfig | None = None
    hyperparameters: Hyperparameters | None = None
    recurrent: FittingActivityRecurrentConfig
    feedforward: FittingActivityFeedforwardConfig

    @property
    def log_interval_s(self) -> float | None:
        """Duration of log interval in seconds (None if training not configured)."""
        if self.training is None:
            return None
        return self.training.log_interval * self.simulation.chunk_duration_s

    @property
    def checkpoint_interval_s(self) -> float | None:
        """Duration of checkpoint interval in seconds (None if training not configured)."""
        if self.training is None:
            return None
        return self.training.checkpoint_interval * self.simulation.chunk_duration_s

    @model_validator(mode="after")
    def validate_checkpoint_alignment(self) -> "FittingActivityParams":
        """Validate that simulation duration aligns with checkpoint interval (only for training)."""
        # Skip validation if no training configuration
        if self.training is None:
            return self

        num_chunks = int(
            self.simulation.duration / (self.simulation.chunk_size * self.simulation.dt)
        )

        if num_chunks % self.training.checkpoint_interval != 0:
            raise ValueError(
                f"Number of chunks ({num_chunks}) must be a multiple of checkpoint_interval "
                f"({self.training.checkpoint_interval}). Either adjust duration or checkpoint_interval."
            )

        return self
