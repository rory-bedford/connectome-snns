"""
Simulating Dp with homeostatic plasticity to achieve target activity regime.

This script sets up and runs a conductance-based leaky integrate-and-fire
network model of the zebrafish dorsal pallium (Dp) with homeostatic plasticity
mechanisms to regulate neuron firing rates and spike train statistics.

Overview:
1. First we generate a biologically plausible recurrent weight matrix with a
   Dp-inspired assembly structure.
2. Next we generate excitatory mitral cell inputs from the OB with Poisson
   statistics and sparse projections.
3. Then we initialise our network with parameters adapted from
   Meissner-Bernard et al. (2025) https://doi.org/10.1016/j.celrep.2025.115330
4. We then run our network simulation in a training loop, applying updates to
   the connectome-constrained weights every so often to optimise the activity
   towards target firing rates and spike train CVs.
"""

import numpy as np
import toml
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
    spike_generators,
)
from network_simulators.conductance_lif_network import ConductanceLIFNetwork
import torch
import sys
from pathlib import Path
from optimisation.loss_functions import CVLoss, FiringRateLoss

# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))


def main(output_dir, params_file):
    """Main execution function for Dp network homeostatic training.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
    """
    # =============================================
    # SETUP: Device selection and parameter loading
    # =============================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        params = toml.load(f)

    # ========== OPTIMISATION TARGETS ==========
    target_firing_rate = float(params["targets"]["firing_rates"])
    target_cv = float(params["targets"]["cvs"])

    # ========== SIMULATION CONFIGURATION ==========
    dt = float(params["simulation"]["dt"])
    chunk_size = float(params["simulation"]["chunk_size"])
    seed = params["simulation"].get("seed", None)
    epochs = int(params["simulation"]["epochs"])
    accumulation_interval = int(params["simulation"]["accumulation_interval"])
    n_steps = int(chunk_size / dt)

    # Set global random seed for reproducibility (only if specified in config)
    if seed is not None:
        np.random.seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ========== HYPERPARAMETERS ==========
    surrgrad_scale = float(params["hyperparameters"]["surrgrad_scale"])
    cv_high_loss = float(params["hyperparameters"]["cv_high_loss"])
    loss_ratio = float(params["hyperparameters"]["loss_ratio"])
    learning_rate = float(params["hyperparameters"]["learning_rate"])

    # ========== FEEDFORWARD LAYER TOPOLOGY ==========
    # Feedforward topology
    input_num_neurons = int(params["feedforward"]["topology"]["num_neurons"])
    input_conn_inputs = np.array(
        params["feedforward"]["topology"]["conn_inputs"], dtype=float
    )

    # Feedforward cell types
    input_cell_type_names = params["feedforward"]["cell_types"]["names"]
    input_cell_type_proportions = np.array(
        params["feedforward"]["cell_types"]["proportion"], dtype=float
    )

    # Feedforward weights
    input_w_mu = np.array(params["feedforward"]["weights"]["w_mu"], dtype=float)
    input_w_sigma = np.array(params["feedforward"]["weights"]["w_sigma"], dtype=float)

    # ========== FEEDFORWARD LAYER ACTIVITY AND SYNAPSES ==========
    # Feedforward cell parameters (as list of dicts)
    # Note: Feedforward cells don't have physiological parameters, just name and id
    cell_params_FF = []
    input_firing_rates = {}
    for cell_id, ct in enumerate(input_cell_type_names):
        firing_rate = float(params["feedforward"]["activity"][ct]["firing_rate"])
        input_firing_rates[ct] = firing_rate  # Store separately for spike generation
        cell_params_FF.append(
            {
                "name": ct,
                "cell_id": cell_id,
            }
        )

    # Feedforward synapse parameters (as list of dicts, flattened from all cell types)
    synapse_params_FF = []
    synapse_id_FF = 0
    for cell_id, ct in enumerate(input_cell_type_names):
        synapse_names = params["feedforward"]["synapses"][ct]["names"]
        tau_rise = params["feedforward"]["synapses"][ct]["tau_rise"]
        tau_decay = params["feedforward"]["synapses"][ct]["tau_decay"]
        E_syn = params["feedforward"]["synapses"][ct]["E_syn"]
        g_bar = params["feedforward"]["synapses"][ct]["g_bar"]

        # Each synapse type for this cell type gets its own entry
        for i, syn_name in enumerate(synapse_names):
            synapse_params_FF.append(
                {
                    "name": syn_name,
                    "synapse_id": synapse_id_FF,
                    "cell_id": cell_id,
                    "tau_rise": float(tau_rise[i]),
                    "tau_decay": float(tau_decay[i]),
                    "E_syn": float(E_syn[i]),
                    "g_bar": float(g_bar[i]),
                }
            )
            synapse_id_FF += 1

    # ========== RECURRENT LAYER TOPOLOGY ==========
    # Recurrent topology
    num_neurons = int(params["recurrent"]["topology"]["num_neurons"])
    num_assemblies = int(params["recurrent"]["topology"]["num_assemblies"])
    conn_within = np.array(params["recurrent"]["topology"]["conn_within"])
    conn_between = np.array(params["recurrent"]["topology"]["conn_between"])

    # Recurrent cell types
    cell_type_names = params["recurrent"]["cell_types"]["names"]
    cell_type_proportions = np.array(
        params["recurrent"]["cell_types"]["proportion"], dtype=float
    )

    # Recurrent weights
    w_mu = np.array(params["recurrent"]["weights"]["w_mu"], dtype=float)
    w_sigma = np.array(params["recurrent"]["weights"]["w_sigma"], dtype=float)

    # ========== RECURRENT LAYER PHYSIOLOGY AND SYNAPSES ==========
    # Recurrent cell parameters (as list of dicts)
    cell_params = []
    for cell_id, ct in enumerate(cell_type_names):
        cell_params.append(
            {
                "name": ct,
                "cell_id": cell_id,
                "tau_mem": float(params["recurrent"]["physiology"][ct]["tau_mem"]),
                "theta": float(params["recurrent"]["physiology"][ct]["theta"]),
                "U_reset": float(params["recurrent"]["physiology"][ct]["U_reset"]),
                "E_L": float(params["recurrent"]["physiology"][ct]["E_L"]),
                "g_L": float(params["recurrent"]["physiology"][ct]["g_L"]),
                "tau_ref": float(params["recurrent"]["physiology"][ct]["tau_ref"]),
            }
        )

    # Recurrent synapse parameters (as list of dicts, flattened from all cell types)
    synapse_params = []
    synapse_id = 0
    for cell_id, ct in enumerate(cell_type_names):
        synapse_names = params["recurrent"]["synapses"][ct]["names"]
        tau_rise = params["recurrent"]["synapses"][ct]["tau_rise"]
        tau_decay = params["recurrent"]["synapses"][ct]["tau_decay"]
        E_syn = params["recurrent"]["synapses"][ct]["E_syn"]
        g_bar = params["recurrent"]["synapses"][ct]["g_bar"]

        # Each synapse type for this cell type gets its own entry
        for i, syn_name in enumerate(synapse_names):
            synapse_params.append(
                {
                    "name": syn_name,
                    "synapse_id": synapse_id,
                    "cell_id": cell_id,
                    "tau_rise": float(tau_rise[i]),
                    "tau_decay": float(tau_decay[i]),
                    "E_syn": float(E_syn[i]),
                    "g_bar": float(g_bar[i]),
                }
            )
            synapse_id += 1

    # ==========================================================
    # STEP 1: Generate Assembly-Based Topology and Visualization
    # ==========================================================

    # First assign cell types to source and target neurons (same for recurrent)
    cell_type_indices = cell_types.assign_cell_types(
        num_neurons=num_neurons,
        cell_type_proportions=cell_type_proportions,
    )

    # Generate assembly-based connectivity graph
    connectivity_graph = topology_generators.assembly_generator(
        source_cell_types=cell_type_indices,
        target_cell_types=cell_type_indices,  # Same for recurrent connections
        num_assemblies=num_assemblies,
        conn_within=conn_within,
        conn_between=conn_between,
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # ===============================
    # STEP 2: Assign Synaptic Weights
    # ===============================

    # Assign log-normal weights to connectivity graph (no signs for conductance-based)
    weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=connectivity_graph,
        source_cell_indices=cell_type_indices,
        target_cell_indices=cell_type_indices,  # Same for recurrent connections
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=w_mu,
        w_sigma_matrix=w_sigma,
        parameter_space="linear",
    )

    # =================================
    # STEP 3: Create Feedforward Inputs
    # =================================

    # Assign cell types to input layer
    input_source_indices = cell_types.assign_cell_types(
        num_neurons=input_num_neurons,
        cell_type_proportions=input_cell_type_proportions,
    )

    # Generate feedforward connectivity graph
    feedforward_connectivity_graph = topology_generators.sparse_graph_generator(
        source_cell_types=input_source_indices,
        target_cell_types=cell_type_indices,  # Connectome layer cell assignments
        conn_matrix=input_conn_inputs,  # N_input_types x N_recurrent_types matrix
        allow_self_loops=True,  # Allow for feedforward connections
        method="configuration",  # Ensures exact in-degree/out-degree distributions
    )

    # Assign log-normal weights to feedforward connectivity (no signs for conductance-based)
    feedforward_weights = weight_assigners.assign_weights_lognormal(
        connectivity_graph=feedforward_connectivity_graph,
        source_cell_indices=input_source_indices,
        target_cell_indices=cell_type_indices,
        cell_type_signs=None,  # No signs for conductance-based model
        w_mu_matrix=input_w_mu,
        w_sigma_matrix=input_w_sigma,
        parameter_space="linear",
    )

    # ==============================
    # STEP 4: Initialize LIF Network
    # ==============================

    # Initialize conductance-based LIF network model
    model = ConductanceLIFNetwork(
        weights=weights,
        cell_type_indices=cell_type_indices,
        cell_params=cell_params,
        synapse_params=synapse_params,
        surrgrad_scale=surrgrad_scale,
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=cell_params_FF,
        synapse_params_FF=synapse_params_FF,
        optimisable="weights",
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # =======================
    # STEP 5: Setup Optimiser
    # =======================

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize target tensors for loss functions
    target_cv_tensor = torch.ones(num_neurons, device=device) * target_cv
    target_rate_tensor = torch.ones(num_neurons, device=device) * target_firing_rate

    # Initialize loss functions
    cv_loss_fn = CVLoss(target_cv=target_cv_tensor, penalty_value=cv_high_loss)
    firing_rate_loss_fn = FiringRateLoss(target_rate=target_rate_tensor, delta_t=dt)

    initial_v = None
    initial_g = None
    initial_g_FF = None

    # =========================
    # STEP 6: Run Training Loop
    # =========================

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}...")

        # Generate new feedforward spikes for this epoch
        input_spikes = spike_generators.generate_poisson_spikes(
            n_steps=n_steps,
            dt=dt,
            num_neurons=input_num_neurons,
            cell_type_indices=input_source_indices,
            cell_type_names=input_cell_type_names,
            firing_rates=input_firing_rates,
            batch_size=1,
            device=device,
        )

        # Run network simulation for this epoch
        chunk_s, chunk_v, _, _, chunk_g, chunk_g_FF = model.forward(
            n_steps=n_steps,
            dt=dt,
            inputs=input_spikes,
            initial_v=initial_v,
            initial_g=initial_g,
            initial_g_FF=initial_g_FF,
        )

        # Compute losses
        cv_loss = cv_loss_fn(chunk_s)
        fr_loss = firing_rate_loss_fn(chunk_s)
        total_loss = (
            loss_ratio * fr_loss + (1 - loss_ratio) * cv_loss
        ) / accumulation_interval

        # Compute gradients
        total_loss.backward()

        # Perform optimisation step every accumulation_interval epochs
        if (epoch + 1) % accumulation_interval == 0:
            optimiser.step()
            optimiser.zero_grad()

        # Create inputs to next chunk - detached so gradients don't flow across chunks
        initial_v = chunk_v[:, -1, :].detach()
        initial_g = chunk_g[:, -1, :].detach()
        initial_g_FF = chunk_g_FF[:, -1, :].detach()
