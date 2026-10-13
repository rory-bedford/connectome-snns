"""Current-based LIF networks: ``recurrent`` and ``feedforward``, on the shared base in
``network``."""

from connectome_snns.network_simulators.current.feedforward import (
    FeedforwardCurrentLIFNetwork,
)
from connectome_snns.network_simulators.current.recurrent import CurrentLIFNetwork

__all__ = ["CurrentLIFNetwork", "FeedforwardCurrentLIFNetwork"]
