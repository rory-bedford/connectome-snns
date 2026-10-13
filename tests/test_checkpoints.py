"""Checkpoints written before the rewrite still load, with the same values.

``state_dicts.pt`` holds every scenario's ``state_dict`` as master's simulators wrote
it (``generate.py state_dicts``). The rewritten simulators must accept each one
strictly — the same keys, shapes and values (floats to a few ulps) — so existing runs
can be resumed. Buffers since removed (``LEGACY_BUFFERS``) are dropped on load.
"""

import pytest
import torch

import fixture_io
from connectome_snns.network_simulators.checkpoint_compat import LEGACY_BUFFERS
from networks import BATCH_SIZE, SCENARIOS

RECORDED = torch.load(fixture_io.FIXTURES / "state_dicts.pt", weights_only=False)


@pytest.mark.parametrize("name", SCENARIOS)
def test_old_state_dict_loads(name):
    recorded = RECORDED["states"][name]
    kept = {k: v for k, v in recorded.items() if k.split(".")[-1] not in LEGACY_BUFFERS}
    model = SCENARIOS[name].build(BATCH_SIZE).model
    current = model.state_dict()
    assert sorted(current) == sorted(kept), (
        f"missing: {sorted(set(kept) - set(current))}, "
        f"new: {sorted(set(current) - set(kept))}"
    )
    for key, value in kept.items():
        assert current[key].dtype == value.dtype, key
        # Floats to a few ulps: values derived with transcendental functions (log
        # weights) round differently on other CPUs' vector units.
        exact = not value.is_floating_point()
        torch.testing.assert_close(
            current[key], value, rtol=0 if exact else 1e-6, atol=0, msg=key
        )
    model.load_state_dict(recorded, strict=True)


@pytest.mark.parametrize(
    "former, name, now",
    [
        ("conductance_based.simulator", "ConductanceLIFNetwork", "conductance"),
        ("conductance_based", "ConductanceLIFNetwork", "conductance"),
        (
            "feedforward_conductance_based.simulator",
            "FeedforwardConductanceLIFNetwork",
            "conductance",
        ),
        (
            "feedforward_conductance_based",
            "FeedforwardConductanceLIFNetwork",
            "conductance",
        ),
        ("current_based.simulator", "CurrentLIFNetwork", "current"),
        ("current_based", "CurrentLIFNetwork", "current"),
        ("feedforward_current_based", "FeedforwardCurrentLIFNetwork", "current"),
    ],
)
def test_former_network_import_paths(former, name, now):
    """Experiments (and pickles) naming the networks' former modules still find them."""
    import importlib

    package = "connectome_snns.network_simulators"
    old = importlib.import_module(f"{package}.{former}")
    assert getattr(old, name) is getattr(
        importlib.import_module(f"{package}.{now}"), name
    )


def test_former_current_base_class_name():
    from connectome_snns.network_simulators.current.network import CurrentNetwork
    from connectome_snns.network_simulators.current_based import CurrentLIFNetwork_IO

    assert CurrentLIFNetwork_IO is CurrentNetwork
