"""SNN Runners for training and inference."""

from .trainer import SNNTrainer
from .inference_runner import SNNInference

__all__ = ["SNNTrainer", "SNNInference"]
