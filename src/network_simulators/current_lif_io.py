"""I/O utilities for LIF network parameters"""

import numpy as np
import toml
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
        params_file: str | Path,
        neuron_types: IntArray,
        recurrent_weights: FloatArray,
        feedforward_weights: FloatArray | None = None,
    ):
        """
        Initialize the LIF network parameters from parameter file.

        Args:
            params_file (str | Path): Path to TOML parameter file.
            neuron_types (IntArray): Array of shape (n_neurons,) with +1 (excitatory) or -1 (inhibitory).
            recurrent_weights (FloatArray): Recurrent weight matrix of shape (n_neurons, n_neurons).
            feedforward_weights (FloatArray | None): Feedforward weight matrix of shape (n_inputs, n_neurons) or None.
        """
        super(CurrentLIFNetwork_IO, self).__init__()

        # Load fixed parameters from file
        fixed_params = self.load_fixed_params(params_file)
        
        # Load optimizable parameters from file
        optimizable_params = self.load_optimizable_params(params_file)
        
        # Load hyperparameters from file
        hyperparams = self.load_hyperparams(params_file)

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

        # Initialize optimizable parameters directly
        if "scaling_factor_E" not in optimizable_params:
            raise ValueError("scaling_factor_E must be specified")
        if "scaling_factor_I" not in optimizable_params:
            raise ValueError("scaling_factor_I must be specified")
        if "scaling_factor_FF" not in optimizable_params:
            raise ValueError("scaling_factor_FF must be specified")
            
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
        
        # Load hyperparameters directly
        if "surrgrad_scale" not in hyperparams:
            raise ValueError("surrgrad_scale must be specified")
            
        surrgrad_scale = hyperparams["surrgrad_scale"]
        assert surrgrad_scale > 0, "surrgrad_scale must be positive"
        self.surrgrad_scale = surrgrad_scale

    def load_fixed_params(self, file_path: str | Path) -> dict[str, torch.Tensor]:
        """Load fixed neuron parameters from TOML file.
        
        Args:
            file_path (str | Path): Path to TOML parameter file.
            
        Returns:
            Dictionary of parameter names to torch tensors
        """
        with open(file_path, 'r') as f:
            toml_data = toml.load(f)

        # Extract cell parameters with E/I suffixes
        param_dict = {}
        if 'cells' in toml_data:
            if 'excitatory' in toml_data['cells']:
                for key, value in toml_data['cells']['excitatory'].items():
                    param_dict[f"{key}_E"] = value
            if 'inhibitory' in toml_data['cells']:
                for key, value in toml_data['cells']['inhibitory'].items():
                    param_dict[f"{key}_I"] = value

        # Validate that all required parameters are present
        required_params = [
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
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Convert to torch tensors
        fixed_params = {}
        for param_name in required_params:
            fixed_params[param_name] = torch.tensor(param_dict[param_name], dtype=torch.float32)
        
        return fixed_params

    def load_optimizable_params(self, file_path: str | Path) -> dict[str, float]:
        """
        Load optimizable parameters.

        Args:
            file_path: Path to TOML parameter file

        Returns:
            Dictionary of optimizable parameter names to their values
        """
        with open(file_path, 'r') as f:
            toml_data = toml.load(f)

        # Extract scaling parameters
        param_dict = {}
        if 'scaling' in toml_data:
            param_dict.update(toml_data['scaling'])

        # Define optimizable parameters
        optimizable_params = [
            "scaling_factor_E",
            "scaling_factor_I", 
            "scaling_factor_FF",
        ]

        # Extract optimizable parameters that exist in the file
        optimizable_values = {}
        for param_name in optimizable_params:
            if param_name in param_dict:
                value = param_dict[param_name]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{param_name}' must be numeric, got {type(value)}")
                optimizable_values[param_name] = float(value)

        return optimizable_values

    def load_hyperparams(self, file_path: str | Path) -> dict[str, float]:
        """
        Load hyperparameters.

        Args:
            file_path: Path to TOML parameter file

        Returns:
            Dictionary of hyperparameter names to their values
        """
        with open(file_path, 'r') as f:
            toml_data = toml.load(f)

        # Extract hyperparameters
        param_dict = {}
        if 'hyperparameters' in toml_data:
            param_dict.update(toml_data['hyperparameters'])

        # Define hyperparameters
        hyperparams = ["surrgrad_scale"]

        # Extract hyperparameters that exist in the file
        hyperparam_values = {}
        for param_name in hyperparams:
            if param_name in param_dict:
                value = param_dict[param_name]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Hyperparameter '{param_name}' must be numeric, got {type(value)}")
                hyperparam_values[param_name] = float(value)

        return hyperparam_values

    def export_to_toml(self, toml_path: str | Path):
        """
        Export all current parameter values to a TOML file.

        Args:
            toml_path: Path where the TOML file will be saved
        """
        toml_path = Path(toml_path)
        
        # Build the structured TOML data
        toml_data = {}
        
        # Cell parameters (split E and I)
        toml_data['cells'] = {
            'excitatory': {
                'tau_mem': float(self.tau_mem_E.item()),
                'tau_syn': float(self.tau_syn_E.item()),
                'R': float(self.R_E.item()),
                'U_rest': float(self.U_rest_E.item()),
                'theta': float(self.theta_E.item()),
                'U_reset': float(self.U_reset_E.item()),
            },
            'inhibitory': {
                'tau_mem': float(self.tau_mem_I.item()),
                'tau_syn': float(self.tau_syn_I.item()),
                'R': float(self.R_I.item()),
                'U_rest': float(self.U_rest_I.item()),
                'theta': float(self.theta_I.item()),
                'U_reset': float(self.U_reset_I.item()),
            }
        }
        
        # Scaling parameters (optimizable)
        toml_data['scaling'] = {
            'scaling_factor_E': float(self.scaling_factor_E.item()),
            'scaling_factor_I': float(self.scaling_factor_I.item()),
            'scaling_factor_FF': float(self.scaling_factor_FF.item()),
        }
        
        # Hyperparameters
        toml_data['hyperparameters'] = {
            'surrgrad_scale': float(self.surrgrad_scale)
        }
        
        # Write to file
        with open(toml_path, 'w') as f:
            toml.dump(toml_data, f)
        
        print(f"Parameters exported to {toml_path}")

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
            torch.Tensor: Scaled recurrent weight matrix.
        """
        assert hasattr(self, "scaling_factor_E"), "scaling_factor_E must be initialized"
        assert hasattr(self, "scaling_factor_I"), "scaling_factor_I must be initialized"
        
        # Create scaling matrix
        scaling = torch.ones_like(self.recurrent_weights)
        
        # Scale excitatory connections (columns with exc_indices)
        scaling[:, self.exc_indices] *= self.scaling_factor_E
        
        # Scale inhibitory connections (columns with inh_indices)  
        scaling[:, self.inh_indices] *= self.scaling_factor_I
        
        return self.recurrent_weights * scaling

    @property  
    def scaled_feedforward_weights(self) -> torch.Tensor:
        """
        Get the feedforward weights scaled by scaling_factor_FF.
        
        Returns:
            torch.Tensor: Scaled feedforward weight matrix.
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
        return lambda x: SurrGradSpike.apply(x, self.surrgrad_scale)