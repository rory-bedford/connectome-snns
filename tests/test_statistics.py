"""At scale, on the GPU, the simulator produces the same science.

Large networks are chaotic: after a few tens of milliseconds any change in arithmetic
gives a different spike train. So the recurrent network is compared statistically, with
the old simulator's own variability as the yardstick: the recording was made with two
independent inputs, and a new run on the first input must agree with the recording at
least as well as the two recordings agree with each other.

The feedforward (teacher-forced) network is not chaotic — a rounding flip changes only
that neuron's next few spikes — so its spike trains must agree almost exactly.

Both are checked eagerly and under ``torch.compile``, which is how experiments run, and
with conductances and recurrent weights stored in float16 (``storage_dtype``).
"""

import numpy as np
import pytest

import fixture_io
from fixtures.generate import STATS_INPUT_SEEDS, stats_run

pytestmark = pytest.mark.gpu

A, B = STATS_INPUT_SEEDS


def population_means(rates, types):
    return np.array([rates[types == t].mean() for t in np.unique(types)])


@pytest.fixture(scope="module")
def recorded():
    return fixture_io.load("statistics")[0]


RUNS = [
    pytest.param(False, "float32", id="eager"),
    pytest.param(True, "float32", id="compiled"),
    pytest.param(False, "float16", id="float16-storage"),
]


@pytest.mark.parametrize("compile, storage_dtype", RUNS)
def test_recurrent_rates(recorded, compile, storage_dtype):
    built, spikes = stats_run(
        "recurrent", A, compile=compile, storage_dtype=storage_dtype
    )
    types = built.structure["types"]
    rates = spikes.mean(axis=(0, 1)) * 1000.0
    rec_a, rec_b = recorded[f"recurrent.rates.{A}"], recorded[f"recurrent.rates.{B}"]

    # Per-type mean rates: within twice the input-to-input difference, or 2%.
    reference = population_means(rec_a, types)
    allowed = np.maximum(
        2 * np.abs(population_means(rec_b, types) - reference), 0.02 * reference
    )
    assert np.all(np.abs(population_means(rates, types) - reference) <= allowed), (
        population_means(rates, types),
        reference,
    )
    # Which neurons fire fast is set by the network: rank agreement at least as good.
    floor = np.corrcoef(rec_a, rec_b)[0, 1]
    assert np.corrcoef(rates, rec_a)[0, 1] >= floor - 0.01


@pytest.mark.parametrize("compile, storage_dtype", RUNS)
def test_feedforward_spikes(recorded, compile, storage_dtype):
    _, spikes = stats_run(
        "feedforward", A, compile=compile, storage_dtype=storage_dtype
    )
    reference = recorded["feedforward.spikes"]
    mismatched = (spikes != reference).sum() / reference.sum()
    assert mismatched < 1e-3, f"{mismatched:.2%} of spikes differ from the recording"
