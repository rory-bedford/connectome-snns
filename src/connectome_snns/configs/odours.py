"""Odour input configuration for olfactory network simulations."""

from pydantic import BaseModel


class OdourInputConfig(BaseModel):
    """Configuration for modulated Poisson input activity.

    Defines baseline and modulated firing rates for input cells responding to odours.
    A fraction of cells are modulated up/down from baseline when odour is present.

    If ``baseline_variance`` > 0, each odour pattern's mean firing rate is sampled
    from N(baseline_rate, baseline_variance) instead of being fixed at baseline_rate.
    This relaxes the perfect-whitening constraint and introduces per-pattern variation
    in total feedforward drive.
    """

    baseline_rate: float
    modulation_rate: float
    modulation_fraction: float
    baseline_variance: float = 0.0

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

    def to_dict(self) -> dict:
        """Convert to a standard dictionary.

        Returns:
            Dictionary with baseline_rate, modulation_rate, and modulation_fraction.
        """
        return {
            "baseline_rate": self.baseline_rate,
            "modulation_rate": self.modulation_rate,
            "modulation_fraction": self.modulation_fraction,
            "baseline_variance": self.baseline_variance,
        }
