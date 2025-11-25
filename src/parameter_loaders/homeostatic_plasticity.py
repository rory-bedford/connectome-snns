"""Parameter loader for homeostatic plasticity training.

For training networks with homeostatic mechanisms to achieve target firing rates.
"""

from pydantic import BaseModel, model_validator
from .base_configs import (
    SimulationConfig,
    TrainingConfig,
    Targets,
    Hyperparameters,
)
from .conductance_based import RecurrentConfig, FeedforwardConfig


# =============================================================================
# CUSTOM CONFIGS FOR HOMEOSTATIC PLASTICITY
# =============================================================================


class HomeostaticTrainingConfig(TrainingConfig):
    """Training config with batch_size for homeostatic plasticity."""

    batch_size: int


# =============================================================================
# TOP-LEVEL MODEL
# =============================================================================


class HomeostaticPlasticityParams(BaseModel):
    """Homeostatic plasticity training."""

    simulation: SimulationConfig
    training: HomeostaticTrainingConfig
    targets: Targets
    hyperparameters: Hyperparameters
    recurrent: RecurrentConfig
    feedforward: FeedforwardConfig

    @property
    def log_interval_s(self) -> float:
        """Duration of log interval in seconds."""
        return self.training.log_interval * self.simulation.chunk_duration_s

    @property
    def checkpoint_interval_s(self) -> float:
        """Duration of checkpoint interval in seconds."""
        return self.training.checkpoint_interval * self.simulation.chunk_duration_s

    @model_validator(mode="after")
    def validate_checkpoint_alignment(self) -> "HomeostaticPlasticityParams":
        """Validate that simulation duration aligns with checkpoint interval."""
        num_chunks = int(
            self.simulation.duration / (self.simulation.chunk_size * self.simulation.dt)
        )

        if num_chunks % self.training.checkpoint_interval != 0:
            raise ValueError(
                f"Number of chunks ({num_chunks}) must be a multiple of checkpoint_interval "
                f"({self.training.checkpoint_interval}). Either adjust duration or checkpoint_interval."
            )

        return self

    @model_validator(mode="after")
    def validate_target_cell_types(self) -> "HomeostaticPlasticityParams":
        """Validate that target keys match recurrent cell types."""
        recurrent_cell_types = set(self.recurrent.cell_types.names)

        # Check all target dictionaries
        target_dicts = {
            "firing_rate": set(self.targets.firing_rate.keys()),
            "alpha": set(self.targets.alpha.keys()),
            "threshold_ratio": set(self.targets.threshold_ratio.keys()),
        }

        for target_name, target_cell_types in target_dicts.items():
            if target_cell_types != recurrent_cell_types:
                missing_in_targets = recurrent_cell_types - target_cell_types
                extra_in_targets = target_cell_types - recurrent_cell_types

                error_msg = f"Target '{target_name}' cell types must match recurrent cell types."
                if missing_in_targets:
                    error_msg += (
                        f"\n  Missing in targets.{target_name}: {missing_in_targets}"
                    )
                if extra_in_targets:
                    error_msg += (
                        f"\n  Extra in targets.{target_name}: {extra_in_targets}"
                    )

                raise ValueError(error_msg)

        return self
