"""I/O utilities for LIF network parameters"""

import pandas as pd
import torch
from pathlib import Path


def load_params_from_csv(csv_path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load and process all parameters from CSV file.

    Args:
        csv_path: Path to CSV parameter file

    Returns:
        Dictionary of parameter names to torch tensors
    """
    params = pd.read_csv(csv_path, comment="#")
    param_dict = dict(zip(params["parameter"], params["value"]))

    # Validate that all required parameters are present
    required_params = [
        "simulation_timestep",
        "membrane_time_constant_exc",
        "membrane_time_constant_inh",
        "synaptic_time_constant_exc",
        "synaptic_time_constant_inh",
        "membrane_resistance_exc",
        "membrane_resistance_inh",
        "resting_potential_exc",
        "resting_potential_inh",
        "spike_threshold_exc",
        "spike_threshold_inh",
        "reset_potential_exc",
        "reset_potential_inh",
        "refractory_period_exc",
        "refractory_period_inh",
    ]

    missing_params = [p for p in required_params if p not in param_dict]
    if missing_params:
        raise ValueError(f"Missing required parameters in CSV: {missing_params}")

    # Validate that all values are numeric and positive where appropriate
    for param_name in required_params:
        value = param_dict[param_name]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Parameter '{param_name}' must be numeric, got {type(value)}"
            )

        # Time constants, resistances, and refractory periods must be positive
        if any(
            x in param_name
            for x in ["time_constant", "timestep", "resistance", "refractory"]
        ):
            if value <= 0:
                raise ValueError(
                    f"Parameter '{param_name}' must be positive, got {value}"
                )

    # Convert to tensors with proper units (ms -> s where needed)
    processed = {
        "dt": torch.tensor(param_dict["simulation_timestep"] * 1e-3),
        "tau_mem_exc": torch.tensor(param_dict["membrane_time_constant_exc"] * 1e-3),
        "tau_mem_inh": torch.tensor(param_dict["membrane_time_constant_inh"] * 1e-3),
        "tau_syn_exc": torch.tensor(param_dict["synaptic_time_constant_exc"] * 1e-3),
        "tau_syn_inh": torch.tensor(param_dict["synaptic_time_constant_inh"] * 1e-3),
        "R_exc": torch.tensor(param_dict["membrane_resistance_exc"]),
        "R_inh": torch.tensor(param_dict["membrane_resistance_inh"]),
        "U_rest_exc": torch.tensor(param_dict["resting_potential_exc"]),
        "U_rest_inh": torch.tensor(param_dict["resting_potential_inh"]),
        "theta_exc": torch.tensor(param_dict["spike_threshold_exc"]),
        "theta_inh": torch.tensor(param_dict["spike_threshold_inh"]),
        "U_reset_exc": torch.tensor(param_dict["reset_potential_exc"]),
        "U_reset_inh": torch.tensor(param_dict["reset_potential_inh"]),
        "tau_ref_exc": torch.tensor(param_dict["refractory_period_exc"] * 1e-3),
        "tau_ref_inh": torch.tensor(param_dict["refractory_period_inh"] * 1e-3),
    }

    # Precompute decay factors
    processed["alpha_exc"] = torch.exp(-processed["dt"] / processed["tau_syn_exc"])
    processed["alpha_inh"] = torch.exp(-processed["dt"] / processed["tau_syn_inh"])
    processed["beta_exc"] = torch.exp(-processed["dt"] / processed["tau_mem_exc"])
    processed["beta_inh"] = torch.exp(-processed["dt"] / processed["tau_mem_inh"])

    return processed


def export_params_to_csv(network, csv_path: str | Path):
    """
    Export network parameters to CSV file.

    Args:
        network: LIFNetwork instance
        csv_path: Path where CSV file will be saved
    """
    csv_path = Path(csv_path)

    # Build list of parameter rows
    rows = [
        ["simulation_timestep", "delta_t", network.dt.item() * 1e3, "ms"],
        [
            "membrane_time_constant_exc",
            "tau_mem_exc",
            network.tau_mem_exc.item() * 1e3,
            "ms",
        ],
        [
            "membrane_time_constant_inh",
            "tau_mem_inh",
            network.tau_mem_inh.item() * 1e3,
            "ms",
        ],
        [
            "synaptic_time_constant_exc",
            "tau_syn_exc",
            network.tau_syn_exc.item() * 1e3,
            "ms",
        ],
        [
            "synaptic_time_constant_inh",
            "tau_syn_inh",
            network.tau_syn_inh.item() * 1e3,
            "ms",
        ],
        ["membrane_resistance_exc", "R_exc", network.R_exc.item(), "MOhm"],
        ["membrane_resistance_inh", "R_inh", network.R_inh.item(), "MOhm"],
        ["resting_potential_exc", "U_rest_exc", network.U_rest_exc.item(), "mV"],
        ["resting_potential_inh", "U_rest_inh", network.U_rest_inh.item(), "mV"],
        ["spike_threshold_exc", "theta_exc", network.theta_exc.item(), "mV"],
        ["spike_threshold_inh", "theta_inh", network.theta_inh.item(), "mV"],
        ["reset_potential_exc", "U_reset_exc", network.U_reset_exc.item(), "mV"],
        ["reset_potential_inh", "U_reset_inh", network.U_reset_inh.item(), "mV"],
        [
            "refractory_period_exc",
            "tau_ref_exc",
            network.tau_ref_exc.item() * 1e3,
            "ms",
        ],
        [
            "refractory_period_inh",
            "tau_ref_inh",
            network.tau_ref_inh.item() * 1e3,
            "ms",
        ],
    ]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["parameter", "symbol", "value", "unit"])

    # Write with comment header
    with open(csv_path, "w") as f:
        f.write("# LIFNetwork parameters exported\n")
        df.to_csv(f, index=False)

    print(f"Parameters exported to {csv_path}")
