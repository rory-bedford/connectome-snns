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
    param_dict = dict(zip(params["symbol"], params["value"]))

    # Validate that all required parameters are present
    required_params = [
        "delta_t",
        "tau_mem_E",
        "tau_mem_I",
        "tau_syn_E",
        "tau_syn_I",
        "R_E",
        "R_I",
        "U_rest_E",
        "U_rest_I",
        "theta_E",
        "theta_I",
        "U_reset_E",
        "U_reset_I",
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
        "dt": torch.tensor(param_dict["delta_t"] * 1e-3),
        "tau_mem_E": torch.tensor(param_dict["tau_mem_E"] * 1e-3),
        "tau_mem_I": torch.tensor(param_dict["tau_mem_I"] * 1e-3),
        "tau_syn_E": torch.tensor(param_dict["tau_syn_E"] * 1e-3),
        "tau_syn_I": torch.tensor(param_dict["tau_syn_I"] * 1e-3),
        "R_E": torch.tensor(param_dict["R_E"]),
        "R_I": torch.tensor(param_dict["R_I"]),
        "U_rest_E": torch.tensor(param_dict["U_rest_E"]),
        "U_rest_I": torch.tensor(param_dict["U_rest_I"]),
        "theta_E": torch.tensor(param_dict["theta_E"]),
        "theta_I": torch.tensor(param_dict["theta_I"]),
        "U_reset_E": torch.tensor(param_dict["U_reset_E"]),
        "U_reset_I": torch.tensor(param_dict["U_reset_I"]),
    }

    # Precompute decay factors
    processed["alpha_E"] = torch.exp(-processed["dt"] / processed["tau_syn_E"])
    processed["alpha_I"] = torch.exp(-processed["dt"] / processed["tau_syn_I"])
    processed["beta_E"] = torch.exp(-processed["dt"] / processed["tau_mem_E"])
    processed["beta_I"] = torch.exp(-processed["dt"] / processed["tau_mem_I"])

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
        ["delta_t", "ms", "Time resolution of the simulation", network.dt.item() * 1e3],
        [
            "tau_mem_E",
            "ms",
            "Membrane time constant for excitatory neurons",
            network.tau_mem_E.item() * 1e3,
        ],
        [
            "tau_mem_I",
            "ms",
            "Membrane time constant for inhibitory neurons",
            network.tau_mem_I.item() * 1e3,
        ],
        [
            "tau_syn_E",
            "ms",
            "Synaptic time constant for excitatory connections",
            network.tau_syn_E.item() * 1e3,
        ],
        [
            "tau_syn_I",
            "ms",
            "Synaptic time constant for inhibitory connections",
            network.tau_syn_I.item() * 1e3,
        ],
        [
            "R_E",
            "MOhm",
            "Membrane resistance for excitatory neurons",
            network.R_E.item(),
        ],
        [
            "R_I",
            "MOhm",
            "Membrane resistance for inhibitory neurons",
            network.R_I.item(),
        ],
        [
            "U_rest_E",
            "mV",
            "Resting membrane potential for excitatory neurons",
            network.U_rest_E.item(),
        ],
        [
            "U_rest_I",
            "mV",
            "Resting membrane potential for inhibitory neurons",
            network.U_rest_I.item(),
        ],
        [
            "theta_E",
            "mV",
            "Spike threshold voltage for excitatory neurons",
            network.theta_E.item(),
        ],
        [
            "theta_I",
            "mV",
            "Spike threshold voltage for inhibitory neurons",
            network.theta_I.item(),
        ],
        [
            "U_reset_E",
            "mV",
            "Reset voltage after spike for excitatory neurons",
            network.U_reset_E.item(),
        ],
        [
            "U_reset_I",
            "mV",
            "Reset voltage after spike for inhibitory neurons",
            network.U_reset_I.item(),
        ],
    ]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["symbol", "unit", "description", "value"])

    # Write with comment header
    with open(csv_path, "w") as f:
        f.write("# LIFNetwork parameters exported\n")
        df.to_csv(f, index=False)

    print(f"Parameters exported to {csv_path}")
