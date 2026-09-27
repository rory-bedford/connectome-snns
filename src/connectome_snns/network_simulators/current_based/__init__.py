"""Recurrent current-based LIF network simulator"""

from .simulator import CurrentLIFNetwork
from .model_init import CurrentLIFNetwork_IO

__all__ = ["CurrentLIFNetwork", "CurrentLIFNetwork_IO"]
