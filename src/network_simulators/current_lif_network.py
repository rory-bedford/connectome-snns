"""Defines a straightforward simulator of recurrent current-based LIF network"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from pathlib import Path
from tqdm import tqdm
from network_simulators.current_lif_io import load_params_from_csv, export_params_to_csv

# Type aliases for clarity
IntArray = NDArray[np.int_]
FloatArray = NDArray[np.float64]


class CurrentLIFNetwork(nn.Module):
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
        super(CurrentLIFNetwork, self).__init__()

        # Load all parameters from CSV
        params = load_params_from_csv(csv_path)

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

        # Register all loaded parameters
        for name, value in params.items():
            self.register_buffer(name, value)

    def export_to_csv(self, csv_path: str | Path):
        """
        Export network parameters to CSV file.

        Args:
            csv_path: Path where CSV file will be saved
        """
        export_params_to_csv(self, csv_path)

    @property
    def device(self):
        """Get the device the model is on"""
        return self.recurrent_weights.device

    def initialise_parameters(
        self,
        E_weight: float,
        I_weight: float,
    ):
        """
        Initialize optimisable parameters.

        Args:
            E_weight (float): Scaling factor for excitatory weights (pA/voxel).
            I_weight (float): Scaling factor for inhibitory weights (pA/voxel).
        """
        assert E_weight > 0, "E_weight must be positive"
        assert I_weight > 0, "I_weight must be positive"

        self.E_weight = nn.Parameter(
            torch.tensor(E_weight, dtype=torch.float32, device=self.device)
        )
        self.I_weight = nn.Parameter(
            torch.tensor(I_weight, dtype=torch.float32, device=self.device)
        )

    @property
    def scaled_recurrent_weights(self) -> torch.Tensor:
        """
        Get the recurrent weights scaled by E_weight and I_weight

        From our connectome weights, we want to scale all E->I and E->E connections by E_weight,
        and all I->E and I->I connections by I_weight. This is done by multiplying each column of the
        recurrent weight matrix by the appropriate scaling factor based on the neuron types.

        Returns:
            torch.Tensor: Scaled recurrent weight matrix of shape (n_neurons, n_neurons).
        """
        assert hasattr(self, "E_weight") and hasattr(self, "I_weight"), (
            "Parameters must be initialized first"
        )
        scaling_factors = torch.where(
            self.neuron_types == 1, self.E_weight, self.I_weight
        )
        return self.recurrent_weights * scaling_factors.unsqueeze(0)

    def forward(
        self,
        n_steps: int,
        inputs: FloatArray | None = None,
        initial_v: FloatArray | None = None,
        initial_I: FloatArray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate the network for a given number of time steps.

        Args:
            n_steps (int): Number of time steps to simulate.
            inputs (FloatArray | None): External input spikes of shape (batches, n_steps, n_inputs).
            initial_v (FloatArray | None): Initial membrane potentials of shape (batches, n_neurons). Defaults to zeros.
            initial_I (FloatArray | None): Initial synaptic currents of shape (batches, n_neurons). Defaults to zeros.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Spike trains, voltages, and currents.
        """

        # Default initial membrane potentials to resting potential if not provided
        if initial_v is None:
            initial_v = torch.zeros(
                (1, self.n_neurons), dtype=torch.float32, device=self.device
            )
            # Set resting potentials based on neuron type
            initial_v[:, self.exc_indices] = self.U_rest_E
            initial_v[:, self.inh_indices] = self.U_rest_I
        else:
            initial_v = torch.as_tensor(
                initial_v, dtype=torch.float32, device=self.device
            )

        # Default initial currents to zero if not provided
        if initial_I is None:
            initial_I = torch.zeros(
                (1, self.n_neurons), dtype=torch.float32, device=self.device
            )
        else:
            initial_I = torch.as_tensor(
                initial_I, dtype=torch.float32, device=self.device
            )

        if inputs is not None:
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)

        batch_size = initial_v.shape[0]

        # Validate dimensions of inputs
        assert n_steps > 0
        assert initial_v.ndim == 2
        assert initial_v.shape[1] == self.n_neurons
        assert initial_I.ndim == 2
        assert initial_I.shape[1] == self.n_neurons
        assert initial_I.shape[0] == batch_size

        if inputs is not None:
            assert self.feedforward_weights is not None
            assert inputs.ndim == 3
            assert inputs.shape[0] == batch_size
            assert inputs.shape[1] == n_steps
            assert inputs.shape[2] == self.n_inputs

        # Create neuron-type subpopulations
        v_exc = initial_v[:, self.exc_indices]
        v_inh = initial_v[:, self.inh_indices]
        I_exc = initial_I[:, self.exc_indices]
        I_inh = initial_I[:, self.inh_indices]
        s_inh = torch.zeros((batch_size, len(self.inh_indices)), device=self.device)
        s_exc = torch.zeros((batch_size, len(self.exc_indices)), device=self.device)

        # Preallocate output tensors
        all_v = torch.zeros((batch_size, n_steps, self.n_neurons), device=self.device)
        all_I = torch.zeros((batch_size, n_steps, self.n_neurons), device=self.device)
        all_s = torch.zeros((batch_size, n_steps, self.n_neurons), device=self.device)

        # Run simulation
        for t in tqdm(range(n_steps), desc="Simulating network", unit="step"):
            # Update membrane potentials (without reset)
            v_exc = (
                self.U_rest_E  # Resting potential
                + (v_exc - self.U_rest_E) * self.beta_E  # Leak
                + I_exc * self.R_E * (1 - self.beta_E)  # Input current
            )

            v_inh = (
                self.U_rest_I  # Resting potential
                + (v_inh - self.U_rest_I) * self.beta_I  # Leak
                + I_inh * self.R_I * (1 - self.beta_I)  # Input current
            )

            # Generate spikes based on threshold
            s_exc = (v_exc >= self.theta_E).float()
            s_inh = (v_inh >= self.theta_I).float()

            # Reset neurons that spiked
            v_exc = v_exc * (1 - s_exc) + self.U_reset_E * s_exc
            v_inh = v_inh * (1 - s_inh) + self.U_reset_I * s_inh

            # Update synaptic currents
            I_exc = (
                I_exc * self.alpha_E  # Decay
                + s_exc
                @ self.scaled_recurrent_weights[
                    self.exc_indices[:, None], self.exc_indices
                ]  # E→E
                + s_inh
                @ self.scaled_recurrent_weights[
                    self.inh_indices[:, None], self.exc_indices
                ]  # I→E
            )

            I_inh = (
                I_inh * self.alpha_I  # Decay
                + s_exc
                @ self.scaled_recurrent_weights[
                    self.exc_indices[:, None], self.inh_indices
                ]  # E→I
                + s_inh
                @ self.scaled_recurrent_weights[
                    self.inh_indices[:, None], self.inh_indices
                ]  # I→I
            )

            # Add feedforward input if present
            if inputs is not None:
                I_exc = (
                    I_exc
                    + inputs[:, t, :] @ self.feedforward_weights[:, self.exc_indices]
                )
                I_inh = (
                    I_inh
                    + inputs[:, t, :] @ self.feedforward_weights[:, self.inh_indices]
                )

            # Store results
            all_v[:, t, self.exc_indices] = v_exc
            all_v[:, t, self.inh_indices] = v_inh
            all_I[:, t, self.exc_indices] = I_exc
            all_I[:, t, self.inh_indices] = I_inh
            all_s[:, t, self.exc_indices] = s_exc
            all_s[:, t, self.inh_indices] = s_inh

        return all_s, all_v, all_I
