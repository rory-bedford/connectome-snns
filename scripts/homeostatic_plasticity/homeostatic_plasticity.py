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
from collections import deque
from synthetic_connectome import (
    topology_generators,
    weight_assigners,
    cell_types,
    spike_generators,
)
from network_simulators.conductance_lif_network import ConductanceLIFNetwork
import torch
from torch.amp import autocast, GradScaler
import sys
import signal
from pathlib import Path
from tqdm import tqdm
from optimisation.loss_functions import CVLoss, FiringRateLoss
import wandb
from matplotlib import pyplot as plt
from torch.utils.checkpoint import checkpoint
from visualization.connectivity import (
    plot_assembly_graph,
    plot_weighted_connectivity,
    plot_input_count_histogram,
    plot_feedforward_connectivity,
    plot_synaptic_input_histogram,
)


# Import local scripts
sys.path.append(str(Path(__file__).resolve().parent))
import homeostatic_plots


def run_chunk_with_checkpoint(
    model: ConductanceLIFNetwork,
    n_steps: int,
    dt: float,
    input_spikes: torch.Tensor,
    initial_v: torch.Tensor,
    initial_g: torch.Tensor,
    initial_g_FF: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Run a single chunk through the model with gradient checkpointing.

    This wrapper allows PyTorch to checkpoint the forward pass and recompute
    it during the backward pass, trading memory for compute.

    Args:
        model (ConductanceLIFNetwork): The network model
        n_steps (int): Number of timesteps in this chunk
        dt (float): Timestep size in milliseconds
        input_spikes (torch.Tensor): Input spike trains for this chunk
        initial_v (torch.Tensor): Initial membrane potentials
        initial_g (torch.Tensor): Initial recurrent conductances
        initial_g_FF (torch.Tensor): Initial feedforward conductances

    Returns:
        tuple: (spikes, voltages, currents, currents_FF, conductances, conductances_FF)
    """
    return model.forward(
        n_steps=n_steps,
        dt=dt,
        inputs=input_spikes,
        initial_v=initial_v,
        initial_g=initial_g,
        initial_g_FF=initial_g_FF,
    )


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    initial_v: np.ndarray,
    initial_g: np.ndarray,
    initial_g_FF: np.ndarray,
    input_spikes: np.ndarray,
    cv_loss: float,
    fr_loss: float,
    total_loss: float,
    best_loss: float,
) -> None:
    """Save model checkpoint to disk.

    Args:
        output_dir (Path): Directory where checkpoint will be saved
        epoch (int): Current epoch number
        model (torch.nn.Module): Model to checkpoint
        optimiser (torch.optim.Optimizer): Optimizer state to save
        scaler (GradScaler): Mixed precision scaler state to save
        initial_v (np.ndarray): Current membrane potentials
        initial_g (np.ndarray): Current recurrent conductances
        initial_g_FF (np.ndarray): Current feedforward conductances
        input_spikes (np.ndarray): Input spike trains
        cv_loss (float): Current CV loss value
        fr_loss (float): Current firing rate loss value
        total_loss (float): Current total loss value
        best_loss (float): Best loss seen so far
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "initial_v": initial_v,
        "initial_g": initial_g,
        "initial_g_FF": initial_g_FF,
        "input_spikes": input_spikes,
        "cv_loss": cv_loss,
        "fr_loss": fr_loss,
        "total_loss": total_loss,
        "best_loss": best_loss,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }

    # Save as latest checkpoint (for resumption)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save as best if this is the best model
    is_best = total_loss <= best_loss
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ✓ New best model saved (loss: {total_loss:.6f})")

    return is_best


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Load model checkpoint from disk.

    Args:
        checkpoint_path (Path): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimiser (torch.optim.Optimizer): Optimizer to load state into
        scaler (GradScaler): Mixed precision scaler to load state into
        device (str): Device to load tensors onto

    Returns:
        tuple: (epoch, initial_v, initial_g, initial_g_FF, best_loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    # Load scaler state if available (for backward compatibility)
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = checkpoint["epoch"]

    # Convert numpy arrays to tensors if needed, then move to device
    if checkpoint["initial_v"] is not None:
        initial_v = checkpoint["initial_v"]
        if isinstance(initial_v, np.ndarray):
            initial_v = torch.from_numpy(initial_v).to(device)
        else:
            initial_v = initial_v.to(device)
    else:
        initial_v = None

    if checkpoint["initial_g"] is not None:
        initial_g = checkpoint["initial_g"]
        if isinstance(initial_g, np.ndarray):
            initial_g = torch.from_numpy(initial_g).to(device)
        else:
            initial_g = initial_g.to(device)
    else:
        initial_g = None

    if checkpoint["initial_g_FF"] is not None:
        initial_g_FF = checkpoint["initial_g_FF"]
        if isinstance(initial_g_FF, np.ndarray):
            initial_g_FF = torch.from_numpy(initial_g_FF).to(device)
        else:
            initial_g_FF = initial_g_FF.to(device)
    else:
        initial_g_FF = None

    best_loss = checkpoint.get("best_loss", float("inf"))

    # Restore random states (convert from numpy if needed)
    rng_state = checkpoint["rng_state"]
    if isinstance(rng_state, np.ndarray):
        # Convert numpy array to ByteTensor
        rng_state = torch.ByteTensor(rng_state.tobytes())
    elif not isinstance(rng_state, torch.Tensor):
        # If it's some other type, try to convert
        rng_state = torch.ByteTensor(rng_state)

    # Ensure it's a ByteTensor (uint8) on CPU
    if rng_state.dtype != torch.uint8:
        rng_state = rng_state.to(torch.uint8)
    if rng_state.device.type != "cpu":
        rng_state = rng_state.cpu()

    torch.set_rng_state(rng_state)
    np.random.set_state(checkpoint["numpy_rng_state"])

    print(f"  ✓ Resumed from epoch {epoch}, best loss: {best_loss:.6f}")
    return epoch, initial_v, initial_g, initial_g_FF, best_loss


def main(
    output_dir, params_file, resume_from=None, use_wandb=True, resumed_output_dir=None
):
    """Main execution function for Dp network homeostatic training.

    Args:
        output_dir (Path): Directory where output files will be saved
        params_file (Path): Path to the file containing network parameters
        resume_from (Path, optional): Path to checkpoint to resume from
        use_wandb (bool): Whether to use Weights & Biases for logging
        resumed_output_dir (Path, optional): Separate directory for plots when resuming training
    """
    # =============================================
    # SETUP: Device selection and parameter loading
    # =============================================

    # Select device (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load all network parameters from TOML file
    with open(params_file, "r") as f:
        params = toml.load(f)

    # ========== OPTIMISATION TARGETS ==========
    target_firing_rate = float(params["targets"]["firing_rates"])
    target_cv = float(params["targets"]["cvs"])

    # ========== SIMULATION CONFIGURATION ==========
    dt = float(params["simulation"]["dt"])
    n_steps = int(params["simulation"]["chunk_size"])
    seed = params["simulation"].get("seed", None)
    num_chunks = int(params["simulation"]["num_chunks"])
    chunks_per_loss = int(params["simulation"]["chunks_per_loss"])
    losses_per_update = int(params["simulation"]["losses_per_update"])
    log_interval = int(params["simulation"]["log_interval"])
    plot_size = int(params["simulation"]["plot_size"])

    # Enable mixed precision training (only beneficial on CUDA)
    use_mixed_precision = params["simulation"].get("mixed_precision", True)

    # Calculate derived values
    chunk_duration = n_steps * dt  # Duration in ms
    chunks_per_update = chunks_per_loss * losses_per_update

    # Validate configuration
    if log_interval % chunks_per_loss != 0:
        raise ValueError(
            f"log_interval ({log_interval}) must be a multiple of chunks_per_loss ({chunks_per_loss}) "
            "to ensure loss values are always available when checkpointing."
        )

    # Set global random seed for reproducibility (only if specified in config)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")
    else:
        print("No seed specified - using random initialization")

    # ========== HYPERPARAMETERS ==========
    surrgrad_scale = float(params["hyperparameters"]["surrgrad_scale"])
    cv_high_loss = float(params["hyperparameters"]["cv_high_loss"])
    loss_ratio = float(params["hyperparameters"]["loss_ratio"])
    learning_rate = float(params["hyperparameters"]["learning_rate"])

    # ========== WANDB CONFIGURATION ==========
    wandb_config = params.get("wandb", {})
    use_wandb_from_config = wandb_config.get("enabled", True)
    wandb_project = wandb_config.get("project", "connectome-snns-homeostatic")
    wandb_entity = wandb_config.get("entity", None)
    wandb_tags = wandb_config.get("tags", [])
    wandb_notes = wandb_config.get("notes", "")

    # Override with function parameter if explicitly set
    use_wandb = use_wandb and use_wandb_from_config

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
        scaling_factors=None,  # No scaling factors for homeostatic plasticity
        weights_FF=feedforward_weights,
        cell_type_indices_FF=input_source_indices,
        cell_params_FF=cell_params_FF,
        synapse_params_FF=synapse_params_FF,
        scaling_factors_FF=None,  # No scaling factors for feedforward
        optimisable="weights",
        use_tqdm=False,  # Disable tqdm progress bar for training loop
    )

    # Move model to device for GPU acceleration
    model.to(device)

    # ========================================================
    # STEP 4.5: Save Initial Network Structure and Weights
    # ========================================================

    # Use resumed_output_dir for plots if provided (when resuming training)
    plots_output_dir = (
        resumed_output_dir if resumed_output_dir is not None else output_dir
    )

    print("\nSaving initial network structure and weights...")
    initial_dir = plots_output_dir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)

    # Save initial weights to disk
    np.save(initial_dir / "recurrent_weights.npy", weights)
    np.save(initial_dir / "feedforward_weights.npy", feedforward_weights)
    np.save(initial_dir / "connectivity_graph.npy", connectivity_graph)
    print(f"✓ Initial weights saved to {initial_dir}")

    # Generate and save initial structure plots
    print("  Generating initial structure plots...")

    # Plot assembly graph
    fig_assembly = plot_assembly_graph(
        connectivity_graph=connectivity_graph,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig_assembly.savefig(
        initial_dir / "assembly_graph.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_assembly)

    # Plot feedforward connectivity
    fig_ff_conn = plot_feedforward_connectivity(
        feedforward_weights=feedforward_weights,
        input_cell_type_indices=input_source_indices,
    )
    fig_ff_conn.savefig(
        initial_dir / "feedforward_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_ff_conn)

    # Plot weighted connectivity
    fig_weighted = plot_weighted_connectivity(
        weights=weights,
        cell_type_indices=cell_type_indices,
        num_assemblies=num_assemblies,
    )
    fig_weighted.savefig(
        initial_dir / "weighted_connectivity.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_weighted)

    # Plot input count histogram
    fig_input_hist = plot_input_count_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
    )
    fig_input_hist.savefig(
        initial_dir / "input_count_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_input_hist)

    # Plot synaptic input histogram (conductance-weighted)
    # Prepare g_bar dictionaries
    recurrent_g_bar_by_type = {}
    for cell_type in cell_type_names:
        g_bar_values = params["recurrent"]["synapses"][cell_type]["g_bar"]
        recurrent_g_bar_by_type[cell_type] = sum(g_bar_values)

    feedforward_g_bar_by_type = {}
    for cell_type in input_cell_type_names:
        g_bar_values = params["feedforward"]["synapses"][cell_type]["g_bar"]
        feedforward_g_bar_by_type[cell_type] = sum(g_bar_values)

    fig_synaptic_input = plot_synaptic_input_histogram(
        weights=weights,
        feedforward_weights=feedforward_weights,
        cell_type_indices=cell_type_indices,
        input_cell_type_indices=input_source_indices,
        cell_type_names=cell_type_names,
        input_cell_type_names=input_cell_type_names,
        recurrent_g_bar_by_type=recurrent_g_bar_by_type,
        feedforward_g_bar_by_type=feedforward_g_bar_by_type,
    )
    fig_synaptic_input.savefig(
        initial_dir / "synaptic_input_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_synaptic_input)

    print(f"✓ Initial structure plots saved to {initial_dir}\n")

    # =======================
    # STEP 5: Setup Optimiser
    # =======================

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler("cuda", enabled=use_mixed_precision and device == "cuda")

    # Initialize target tensors for loss functions
    target_cv_tensor = torch.ones(num_neurons, device=device) * target_cv
    target_rate_tensor = torch.ones(num_neurons, device=device) * target_firing_rate

    # Initialize loss functions
    cv_loss_fn = CVLoss(target_cv=target_cv_tensor, penalty_value=cv_high_loss)
    firing_rate_loss_fn = FiringRateLoss(target_rate=target_rate_tensor, dt=dt)

    # Initial arrays for chunk inputs
    initial_v = None
    initial_g = None
    initial_g_FF = None

    # ========================================
    # STEP 6: Initialize wandb (if requested)
    # ========================================

    if use_wandb:
        # Initialize wandb with config from TOML
        wandb_init_kwargs = {
            "project": wandb_project,
            "name": output_dir.name,
            "config": {
                **params,  # Log all parameters
                "output_dir": str(output_dir),
                "device": device,
            },
            "dir": str(output_dir),  # Save wandb files to output directory
        }

        # Add optional parameters if specified
        if wandb_entity:
            wandb_init_kwargs["entity"] = wandb_entity
        if wandb_tags:
            wandb_init_kwargs["tags"] = wandb_tags
        if wandb_notes:
            wandb_init_kwargs["notes"] = wandb_notes

        wandb.init(**wandb_init_kwargs)
        wandb.watch(model, log="all", log_freq=log_interval)

    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float("inf")

    if resume_from is not None:
        start_epoch, initial_v, initial_g, initial_g_FF, best_loss = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimiser=optimiser,
            scaler=scaler,
            device=device,
        )

    print(f"\nStarting training from chunk {start_epoch}...")
    print(f"Target firing rate: {target_firing_rate} Hz")
    print(f"Target CV: {target_cv}")
    print(f"Total chunks: {num_chunks} ({num_chunks * chunk_duration / 1000:.2f} s)")
    print(
        f"Chunks per loss: {chunks_per_loss} chunks ({chunks_per_loss * chunk_duration / 1000:.2f} s)"
    )
    print(
        f"Losses per update: {losses_per_update} losses ({chunks_per_update * chunk_duration / 1000:.2f} s)"
    )
    print(
        f"Log interval: {log_interval} chunks ({log_interval * chunk_duration / 1000:.2f} s)"
    )

    # =========================
    # STEP 7: Run Training Loop
    # =========================

    # Setup signal handler for graceful interruption (Ctrl+C)
    pbar = None  # Will be initialized below

    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n" + "=" * 60)
        print("⚠️  INTERRUPT RECEIVED - Cleaning up...")
        print("=" * 60)

        # Close progress bar if it exists
        if pbar is not None:
            pbar.close()

        # Finish wandb run if active
        if use_wandb:
            print("  Finalizing Weights & Biases...")
            wandb.finish()

        print("  ✓ Cleanup complete")
        print("=" * 60)
        print("Exiting...\n")
        sys.exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize gradient accumulation
    optimiser.zero_grad()

    # Accumulators for loss computation (GPU tensors with gradients)
    loss_spikes = []

    # Accumulators for plotting (CPU numpy arrays in deques with maxlen)
    # These use deques with maxlen=plot_size for efficient memory management
    plot_spikes = deque(maxlen=plot_size)
    plot_voltages = deque(maxlen=plot_size)
    plot_currents = deque(maxlen=plot_size)
    plot_currents_FF = deque(maxlen=plot_size)
    plot_conductances = deque(maxlen=plot_size)
    plot_conductances_FF = deque(maxlen=plot_size)
    plot_input_spikes = deque(maxlen=plot_size)

    # Initialize progress bar for training loop
    pbar = tqdm(
        range(start_epoch, num_chunks),
        desc="Training",
        unit="chunk",
        initial=start_epoch,
        total=num_chunks,
    )

    for epoch in pbar:
        # Generate new feedforward spikes for this chunk
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

        # Run network simulation for this chunk (with checkpointing and mixed precision)
        with autocast(
            device_type="cuda", enabled=use_mixed_precision and device == "cuda"
        ):
            chunk_s, chunk_v, chunk_I, chunk_I_FF, chunk_g, chunk_g_FF = checkpoint(
                run_chunk_with_checkpoint,
                model,
                n_steps,
                dt,
                input_spikes,
                initial_v,
                initial_g,
                initial_g_FF,
                use_reentrant=False,
            )

        # Carry state forward to next chunk (detach to stop gradient flow)
        initial_v = chunk_v[:, -1, :].detach()
        initial_g = chunk_g[:, -1, :, :, :].detach()
        initial_g_FF = chunk_g_FF[:, -1, :, :, :].detach()

        # Accumulate spikes for loss computation (keep on GPU with gradients)
        loss_spikes.append(chunk_s)

        # Accumulate data for plotting (detach and move to CPU as numpy arrays)
        # These are stored in deques with maxlen, so old data is automatically discarded
        plot_spikes.append(chunk_s.detach().cpu().numpy())
        plot_voltages.append(chunk_v.detach().cpu().numpy())
        plot_currents.append(chunk_I.detach().cpu().numpy())
        plot_currents_FF.append(chunk_I_FF.detach().cpu().numpy())
        plot_conductances.append(chunk_g.detach().cpu().numpy())
        plot_conductances_FF.append(chunk_g_FF.detach().cpu().numpy())
        plot_input_spikes.append(input_spikes.detach().cpu().numpy())

        # Delete chunk tensors to free GPU memory immediately
        del chunk_v, chunk_I, chunk_I_FF, chunk_g, chunk_g_FF

        # Check if it's time to compute loss
        if (epoch + 1) % chunks_per_loss == 0:
            # Indicate gradient computation is starting
            pbar.set_description("Training [Computing gradients...]")

            # Concatenate accumulated chunks for loss computation
            full_spikes = torch.cat(
                loss_spikes, dim=1
            )  # (batch, chunks_per_loss * n_steps, n_neurons)

            # Compute loss over full trajectory with mixed precision
            with autocast(
                device_type="cuda", enabled=use_mixed_precision and device == "cuda"
            ):
                cv_loss = cv_loss_fn(full_spikes)
                fr_loss = firing_rate_loss_fn(full_spikes)
                total_loss = loss_ratio * fr_loss + (1 - loss_ratio) * cv_loss

                # Scale loss by number of loss computations per update (so gradient magnitude is consistent)
                total_loss = total_loss / (chunks_per_update // chunks_per_loss)

            # Backward pass with gradient scaling (accumulates gradients)
            scaler.scale(total_loss).backward()

            # Reset description after gradient computation
            pbar.set_description("Training")

            # Convert to numpy for logging (detach first)
            cv_loss_np = cv_loss.detach().cpu().item()
            fr_loss_np = fr_loss.detach().cpu().item()
            total_loss_np = total_loss.detach().cpu().item()

            # Delete loss tensors and clear loss_spikes accumulator to free GPU memory
            del cv_loss, fr_loss, total_loss, full_spikes
            loss_spikes.clear()

            # Update progress bar with loss information
            pbar.set_postfix(
                {
                    "CV": f"{cv_loss_np:.4f}",
                    "FR": f"{fr_loss_np:.4f}",
                    "Total": f"{total_loss_np:.4f}",
                }
            )

        # Check if it's time to update weights
        if (epoch + 1) % chunks_per_update == 0:
            # Indicate weight update is starting
            pbar.set_description("Training [Updating weights...]")

            # Unscale gradients before clipping
            scaler.unscale_(optimiser)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights with scaled gradients
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

            # Clear CUDA cache after weight update to release orphaned memory
            if device == "cuda":
                torch.cuda.empty_cache()

            # Brief indication that weights were updated
            pbar.set_description("Training [✓ Weights updated]")

            # Reset back to normal after a moment
            pbar.set_description("Training")

        # Save checkpoint and statistics periodically
        if (epoch + 1) % log_interval == 0:
            # Clear progress bar and print checkpoint header
            pbar.clear()
            print(f"\n{'=' * 60}")
            print(f"Checkpoint at chunk {epoch + 1}/{num_chunks}")
            print("=" * 60)

            # Concatenate all accumulated data for visualization and checkpointing
            vis_spikes = np.concatenate(list(plot_spikes), axis=1)
            vis_voltages = np.concatenate(list(plot_voltages), axis=1)
            vis_currents = np.concatenate(list(plot_currents), axis=1)
            vis_currents_FF = np.concatenate(list(plot_currents_FF), axis=1)
            vis_conductances = np.concatenate(list(plot_conductances), axis=1)
            vis_conductances_FF = np.concatenate(list(plot_conductances_FF), axis=1)
            vis_input_spikes = np.concatenate(list(plot_input_spikes), axis=1)

            # Save checkpoint
            is_best = save_checkpoint(
                output_dir=output_dir,
                epoch=epoch + 1,
                model=model,
                optimiser=optimiser,
                scaler=scaler,
                initial_v=initial_v.cpu().numpy()
                if torch.is_tensor(initial_v)
                else initial_v,
                initial_g=initial_g.cpu().numpy()
                if torch.is_tensor(initial_g)
                else initial_g,
                initial_g_FF=initial_g_FF.cpu().numpy()
                if torch.is_tensor(initial_g_FF)
                else initial_g_FF,
                input_spikes=vis_input_spikes,
                cv_loss=cv_loss_np,
                fr_loss=fr_loss_np,
                total_loss=total_loss_np,
                best_loss=best_loss,
            )
            if is_best:
                best_loss = total_loss_np

            # Compute statistics on accumulated trajectory
            stats = homeostatic_plots.compute_network_statistics(
                spikes=vis_spikes,
                cell_type_indices=cell_type_indices,
                cell_type_names=cell_type_names,
                dt=dt,
            )

            # Log to console
            print(f"  CV Loss: {cv_loss_np:.6f}")
            print(f"  FR Loss: {fr_loss_np:.6f}")
            print(f"  Total Loss: {total_loss_np:.6f}")
            # Print statistics by cell type
            for cell_type_name in cell_type_names:
                mean_fr = stats[f"firing_rate/{cell_type_name}/mean"]
                mean_cv = stats[f"cv/{cell_type_name}/mean"]
                frac_active = stats[f"fraction_active/{cell_type_name}"]
                print(
                    f"  {cell_type_name}: FR={mean_fr:.3f} Hz, CV={mean_cv:.3f}, active={frac_active:.3f}"
                )

            # Generate and log plots
            print("  Generating plots...")
            figures_dir = plots_output_dir / "figures" / f"chunk_{epoch + 1:06d}"
            figures_dir.mkdir(parents=True, exist_ok=True)

            figures = homeostatic_plots.generate_training_plots(
                spikes=vis_spikes,
                voltages=vis_voltages,
                conductances=vis_conductances,
                conductances_FF=vis_conductances_FF,
                currents=vis_currents,
                currents_FF=vis_currents_FF,
                input_spikes=vis_input_spikes,
                cell_type_indices=cell_type_indices,
                input_cell_type_indices=input_source_indices,
                cell_type_names=cell_type_names,
                input_cell_type_names=input_cell_type_names,
                weights=weights,
                feedforward_weights=feedforward_weights,
                connectivity_graph=connectivity_graph,
                num_assemblies=num_assemblies,
                params=params,
                dt=dt,
            )

            # Log to wandb (combine all metrics and plots in a single call)
            if use_wandb:
                # Create wandb Images directly from matplotlib figures
                wandb_plots = {
                    f"plots/{plot_name}": wandb.Image(fig)
                    for plot_name, fig in figures.items()
                }

                # Log everything together at the same step
                wandb.log(
                    {
                        "loss/cv": cv_loss_np,
                        "loss/firing_rate": fr_loss_np,
                        "loss/total": total_loss_np,
                        **stats,
                        **wandb_plots,
                    },
                    step=epoch + 1,
                )

            # Save figures to disk after wandb has processed them
            for plot_name, fig in figures.items():
                fig_path = figures_dir / f"{plot_name}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

            print(f"  ✓ Saved plots to {figures_dir}")
            print("=" * 60)

            # Force garbage collection and clear CUDA cache to ensure memory is freed
            import gc

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Refresh the progress bar after checkpoint output
            pbar.refresh()
            pbar.set_description("Training")

    # Close progress bar
    pbar.close()

    # =====================================
    # STEP 8: Save Final Model and Clean Up
    # =====================================

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Save final checkpoint
    # Get the last input spikes from the plot queue
    final_input_spikes = (
        plot_input_spikes[-1]
        if len(plot_input_spikes) > 0
        else input_spikes.detach().cpu().numpy()
    )
    save_checkpoint(
        output_dir=output_dir,
        epoch=num_chunks,
        model=model,
        optimiser=optimiser,
        scaler=scaler,
        initial_v=initial_v.cpu().numpy() if torch.is_tensor(initial_v) else initial_v,
        initial_g=initial_g.cpu().numpy() if torch.is_tensor(initial_g) else initial_g,
        initial_g_FF=initial_g_FF.cpu().numpy()
        if torch.is_tensor(initial_g_FF)
        else initial_g_FF,
        input_spikes=final_input_spikes,
        cv_loss=cv_loss_np,
        fr_loss=fr_loss_np,
        total_loss=total_loss_np,
        best_loss=best_loss,
    )

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    print(f"✓ Final checkpoint saved to {output_dir / 'checkpoints'}")
    print(f"✓ Best loss achieved: {best_loss:.6f}")
