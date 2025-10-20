"""I/O utilities for LIF network parameters"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pathlib import Path
from optimisation.surrogate_gradients import SurrGradSpike

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class CurrentLIFNetwork_IO(nn.Module):
    """Base class for LIF network with I/O functionality and parameter management."""
    
    def __init__(
        self,
        csv_path: str | Path,
        neuron_types: IntArray,
        recurrent_weights: FloatArray,
        feedforward_weights: FloatArray | None = None,
    ):
        """
        Initialize the LIF network parameters from CSV file.

        Args:
            csv_path (str | Path): Path to CSV parameter file.
            neuron_types (IntArray): Array of shape (n_neurons,) with +1 (excitatory) or -1 (inhibitory).
            recurrent_weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            feedforward_weights (FloatArray | None): Feedforward weight matrix of shape (n_inputs, n_neurons) or None.
        """
        super(CurrentLIFNetwork_IO, self).__init__()

        # Load fixed parameters from CSV
        fixed_params = self.load_fixed_params_from_csv(csv_path)
        
        # Load optimizable parameters from CSV
        optimizable_params = self.load_optimizable_params_from_csv(csv_path)
        
        # Load hyperparameters from CSV
        hyperparams = self.load_hyperparams_from_csv(csv_path)

        # Basic assertions
        assert neuron_types.ndim == 1
        assert recurrent_weights.ndim == 2
        assert recurrent_weights.shape[0] == recurrent_weights.shape[1]
        assert neuron_types.shape[0] == recurrent_weights.shape[0]

        if feedforward_weights is not None:
            assert feedforward_weights.ndim == 2
            assert feedforward_weights.shape[1] == neuron_types.shape[0]

        # Extract indices before registering them
        exc_indices = torch.from_numpy(np.where(neuron_types == 1)[0]).long()
        inh_indices = torch.from_numpy(np.where(neuron_types == -1)[0]).long()

        # Register network structure
        self.register_buffer("neuron_types", torch.from_numpy(neuron_types).long())
        self.register_buffer("exc_indices", exc_indices)
        self.register_buffer("inh_indices", inh_indices)
        self.register_buffer("n_neurons", torch.tensor(neuron_types.shape[0]))
        self.register_buffer(
            "recurrent_weights", torch.from_numpy(recurrent_weights).float()
        )
        if feedforward_weights is not None:
            self.register_buffer(
                "feedforward_weights", torch.from_numpy(feedforward_weights).float()
            )
            self.register_buffer("n_inputs", torch.tensor(feedforward_weights.shape[0]))
        else:
            self.feedforward_weights = None
            self.n_inputs = None

        # Register all fixed loaded parameters
        for name, value in fixed_params.items():
            self.register_buffer(name, value)

        # Initialize optimizable parameters directly - no defaults, must exist in CSV
        if "scaling_factor_E" not in optimizable_params:
            raise ValueError("scaling_factor_E must be specified in CSV file")
        if "scaling_factor_I" not in optimizable_params:
            raise ValueError("scaling_factor_I must be specified in CSV file")
        if "scaling_factor_FF" not in optimizable_params:
            raise ValueError("scaling_factor_FF must be specified in CSV file")
            
        E_weight = optimizable_params["scaling_factor_E"]
        I_weight = optimizable_params["scaling_factor_I"]
        FF_weight = optimizable_params["scaling_factor_FF"]
        
        assert E_weight > 0, "E_weight must be positive"
        assert I_weight > 0, "I_weight must be positive"
        assert FF_weight > 0, "FF_weight must be positive"

        self.scaling_factor_E = nn.Parameter(
            torch.tensor(E_weight, dtype=torch.float32, device=self.device)
        )
        self.scaling_factor_I = nn.Parameter(
            torch.tensor(I_weight, dtype=torch.float32, device=self.device)
        )
        self.scaling_factor_FF = nn.Parameter(
            torch.tensor(FF_weight, dtype=torch.float32, device=self.device)
        )
        
        # Load hyperparameters directly - no defaults, must exist in CSV
        if "surrgrad_scale" not in hyperparams:
            raise ValueError("surrgrad_scale must be specified in CSV file")
            
        surrgrad_scale = hyperparams["surrgrad_scale"]
        assert surrgrad_scale > 0, "surrgrad_scale must be positive"
        self.surrgrad_scale = surrgrad_scale

    def load_fixed_params_from_csv(self, csv_path: str | Path) -> dict[str, torch.Tensor]:
        """
        Load and process fixed (non-optimizable) parameters from CSV file.

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

        return processed

    def load_optimizable_params_from_csv(self, csv_path: str | Path) -> dict[str, float]:
        """
        Load optimizable parameters from CSV file.

        Args:
            csv_path: Path to CSV parameter file

        Returns:
            Dictionary of optimizable parameter names to their values
        """
        params = pd.read_csv(csv_path, comment="#")
        param_dict = dict(zip(params["symbol"], params["value"]))

        # Define optimizable parameters
        optimizable_params = [
            "scaling_factor_E",
            "scaling_factor_I", 
            "scaling_factor_FF",
        ]

        # Extract optimizable parameters that exist in the CSV
        optimizable_values = {}
        for param_name in optimizable_params:
            if param_name in param_dict:
                value = param_dict[param_name]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Optimizable parameter '{param_name}' must be numeric, got {type(value)}"
                    )
                if value <= 0:
                    raise ValueError(
                        f"Optimizable parameter '{param_name}' must be positive, got {value}"
                    )
                optimizable_values[param_name] = float(value)

        return optimizable_values

    def load_hyperparams_from_csv(self, csv_path: str | Path) -> dict[str, float]:
        """
        Load hyperparameters from CSV file.

        Args:
            csv_path: Path to CSV parameter file

        Returns:
            Dictionary of hyperparameter names to their values
        """
        params = pd.read_csv(csv_path, comment="#")
        param_dict = dict(zip(params["symbol"], params["value"]))

        # Define hyperparameters
        hyperparams = [
            "surrgrad_scale",
        ]

        # Extract hyperparameters that exist in the CSV
        hyperparam_values = {}
        for param_name in hyperparams:
            if param_name in param_dict:
                value = param_dict[param_name]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Hyperparameter '{param_name}' must be numeric, got {type(value)}"
                    )
                if value <= 0:
                    raise ValueError(
                        f"Hyperparameter '{param_name}' must be positive, got {value}"
                    )
                hyperparam_values[param_name] = float(value)

        return hyperparam_values

    def export_to_csv(self, csv_path: str | Path):
        """
        Export network parameters to CSV file.

        Args:
            csv_path: Path where CSV file will be saved
        """
        csv_path = Path(csv_path)

        # Build list of parameter rows
        rows = [
            [
                "tau_mem_E",
                "ms",
                "Membrane time constant for excitatory neurons",
                self.tau_mem_E.item() * 1e3,
            ],
            [
                "tau_mem_I",
                "ms",
                "Membrane time constant for inhibitory neurons",
                self.tau_mem_I.item() * 1e3,
            ],
            [
                "tau_syn_E",
                "ms",
                "Synaptic time constant for excitatory connections",
                self.tau_syn_E.item() * 1e3,
            ],
            [
                "tau_syn_I",
                "ms",
                "Synaptic time constant for inhibitory connections",
                self.tau_syn_I.item() * 1e3,
            ],
            [
                "R_E",
                "MOhm",
                "Membrane resistance for excitatory neurons",
                self.R_E.item(),
            ],
            [
                "R_I",
                "MOhm",
                "Membrane resistance for inhibitory neurons",
                self.R_I.item(),
            ],
            [
                "U_rest_E",
                "mV",
                "Resting membrane potential for excitatory neurons",
                self.U_rest_E.item(),
            ],
            [
                "U_rest_I",
                "mV",
                "Resting membrane potential for inhibitory neurons",
                self.U_rest_I.item(),
            ],
            [
                "theta_E",
                "mV",
                "Spike threshold voltage for excitatory neurons",
                self.theta_E.item(),
            ],
            [
                "theta_I",
                "mV",
                "Spike threshold voltage for inhibitory neurons",
                self.theta_I.item(),
            ],
            [
                "U_reset_E",
                "mV",
                "Reset voltage after spike for excitatory neurons",
                self.U_reset_E.item(),
            ],
            [
                "U_reset_I",
                "mV",
                "Reset voltage after spike for inhibitory neurons",
                self.U_reset_I.item(),
            ],
        ]

        # Add optimizable parameters if they exist
        if hasattr(self, "scaling_factor_E"):
            rows.append([
                "scaling_factor_E",
                "pA/voxel",
                "Scaling factor for excitatory weights",
                self.scaling_factor_E.item(),
            ])
        
        if hasattr(self, "scaling_factor_I"):
            rows.append([
                "scaling_factor_I", 
                "pA/voxel",
                "Scaling factor for inhibitory weights",
                self.scaling_factor_I.item(),
            ])
            
        if hasattr(self, "scaling_factor_FF"):
            rows.append([
                "scaling_factor_FF",
                "pA/voxel", 
                "Scaling factor for feedforward weights",
                self.scaling_factor_FF.item(),
            ])

        # Add hyperparameters if they exist
        if hasattr(self, "surrgrad_scale"):
            rows.append([
                "surrgrad_scale",
                "",
                "Scale for surrogate gradient fast sigmoid function",
                self.surrgrad_scale,
            ])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=["symbol", "unit", "description", "value"])

        # Write with comment header
        with open(csv_path, "w") as f:
            f.write("# LIFNetwork parameters exported\n")
            df.to_csv(f, index=False)

        print(f"Parameters exported to {csv_path}")

    @property
    def device(self):
        """Get the device the model is on"""
        return self.recurrent_weights.device

    @property
    def scaled_recurrent_weights(self) -> torch.Tensor:
        """
        Get the recurrent weights scaled by scaling_factor_E and scaling_factor_I.

        From our connectome weights, we want to scale all E->I and E->E connections by scaling_factor_E,
        and all I->E and I->I connections by scaling_factor_I.

        Returns:
            torch.Tensor: Scaled recurrent weight matrix of shape (n_neurons, n_neurons).
        """
        assert hasattr(self, "scaling_factor_E") and hasattr(self, "scaling_factor_I"), (
            "Parameters must be initialized first"
        )
        
        scaling_factors = torch.where(
            self.neuron_types == 1, self.scaling_factor_E, self.scaling_factor_I
        )
        return self.recurrent_weights * scaling_factors.unsqueeze(0)

    @property  
    def scaled_feedforward_weights(self) -> torch.Tensor:
        """
        Get the feedforward weights scaled by scaling_factor_FF.

        Returns:
            torch.Tensor: Scaled feedforward weight matrix of shape (n_inputs, n_neurons).
            
        Raises:
            ValueError: If no feedforward weights exist.
        """
        if self.feedforward_weights is None:
            raise ValueError("No feedforward weights available")
            
        assert hasattr(self, "scaling_factor_FF"), "scaling_factor_FF must be initialized"
        return self.feedforward_weights * self.scaling_factor_FF

    @property
    def spike_fn(self):
        """
        Get the surrogate gradient spike function with the current scale parameter.
        
        Returns:
            Callable: Partial function for SurrGradSpike with configured scale.
        """
        return lambda x: (x >= 0).float()
        #return lambda x: SurrGradSpike.apply(x, self.surrgrad_scale)
