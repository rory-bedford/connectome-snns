"""Layer configs derive the same parameter tables as when the fixture was recorded
(less the refractory period ``tau_ref``, removed since: it was never simulated)."""

import json

import fixture_io
from fixtures.generate import config_outputs


def test_layer_config_outputs():
    recorded = json.loads((fixture_io.FIXTURES / "configs.json").read_text())["outputs"]
    for cell in recorded["recurrent.get_cell_params"]:
        del cell["tau_ref"]
    assert config_outputs() == recorded


def test_old_configs_with_tau_ref_still_parse():
    from connectome_snns.configs.conductance_based import PhysiologyConfig

    config = PhysiologyConfig(
        tau_mem=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0, tau_ref=2.0
    )
    assert "tau_ref" not in config.model_dump()


def test_unknown_keys_are_errors():
    import pytest
    from pydantic import ValidationError

    from connectome_snns.configs import SimulationConfig
    from connectome_snns.configs.conductance_based import PhysiologyConfig

    with pytest.raises(ValidationError, match="tau_memm"):
        PhysiologyConfig(
            tau_mem=20.0, tau_memm=20.0, theta=-50.0, U_reset=-60.0, E_L=-60.0, g_L=1.0
        )
    with pytest.raises(ValidationError, match="sead"):
        SimulationConfig(dt=1.0, duration=100.0, seed=1, sead=2)
