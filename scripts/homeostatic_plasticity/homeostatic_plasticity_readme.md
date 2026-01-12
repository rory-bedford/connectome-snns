# Homeostatic Plasticity Training

This workflow trains a connectome-constrained conductance-based LIF network using homeostatic plasticity mechanisms to regulate network activity toward target firing rates and spike train statistics. The network generates its own connectivity structure and learns through gradient-based weight updates with multiple loss functions that encourage biologically plausible activity patterns.

**Workflow Position:** Stage 2 - After network_inference (which can be used for grid search). Optimized weights from this stage serve as the teacher network for student training (Stage 3).

## Scripts

### 1. `backprop_hp.py` (Gradient-Based Approach) ⭐ MAIN IMPLEMENTATION
Trains a network using backpropagation through time with multiple loss functions (firing rate, CV, silent penalty, membrane variance, recurrent-feedforward balance). Uses parameters identified from Stage 1 grid search to initialize connectivity, then optimizes weights for target activity patterns.

### 2. `feedback_hp.py` (NOT YET IMPLEMENTED)
Placeholder for a feedback-based approach that would adjust weights based on observed firing rates without backpropagation (similar to biological homeostatic mechanisms). Currently uses the same gradient-based approach as `backprop_hp.py`.

### 3. `resume_backprop_hp.py`
Resumes gradient-based homeostatic plasticity training from a checkpoint.

## Features

### Checkpointing
- **Strategy**: Best + Latest checkpoint strategy
  - `checkpoint_latest.pt`: Always contains the most recent epoch (for resuming after failures)
  - `checkpoint_best.pt`: Contains the model with the best loss achieved so far
- **Interval**: Configurable via `checkpoint_interval` in parameters file (default: 50 epochs)
- **Contents**: Model state, optimizer state, epoch number, losses, random states, initial neuron states

### Weights & Biases Logging
- **Minimal integration** - logs essential metrics without overhead
- **What's logged**:
  - Losses (firing rate, CV, silent penalty, membrane variance)
  - Network statistics (mean/std firing rates, CVs, active fraction per cell type)
  - All training plots (updated every `accumulation_interval`)
- **Config**: All parameters from TOML file are logged to wandb

### Plotting
- Generates comprehensive plots every `log_interval` chunks
- **Initial state**: Before training, runs 10s inference with initial weights and saves plots to `initial_state/figures/`
- Plots include:
  - Network structure (assembly graphs, connectivity matrices)
  - Input analysis (mitral cell spikes, feedforward connectivity)
  - Output analysis (spike rasters, firing rate distributions)
  - Neuronal dynamics (membrane voltages, conductances)
  - Firing statistics (Fano factors, CV histograms, ISI distributions)
- All plots saved to `output_dir/figures/chunk_XXXX/`
- All plots automatically logged to wandb

## Usage

### Basic Usage (via experiment runner)

```bash
# Run with default parameters
python scripts/run_experiment.py parameters/homeostatic-plasticity.toml
```

### Resuming from Checkpoint

To resume from a crashed run, modify the script call to include `resume_from`:

```python
# In your experiment config or by modifying the main() call:
module.main(
    output_dir=tracker.output_dir, 
    params_file=tracker.params_file,
    resume_from=Path("path/to/output_dir/checkpoints/checkpoint_latest.pt")
)
```

Or create a simple runner script:

```python
from pathlib import Path
from scripts.homeostatic_plasticity import main

main(
    output_dir=Path("workspace/my_experiment"),
    params_file=Path("parameters/homeostatic-plasticity.toml"),
    resume_from=Path("workspace/my_experiment/checkpoints/checkpoint_latest.pt"),
    use_wandb=True
)
```

### Disabling wandb

```python
main(
    output_dir=Path("workspace/my_experiment"),
    params_file=Path("parameters/homeostatic-plasticity.toml"),
    use_wandb=False  # Disable wandb logging
)
```

## Loss Functions

The training uses multiple loss components to shape network activity:

1. **Firing Rate Loss**: Encourages neurons to match target firing rates per cell type
2. **CV Loss**: Regulates spike train regularity via coefficient of variation
3. **Silent Neuron Penalty**: Penalizes neurons with zero activity to maintain network-wide participation
4. **Membrane Variance Loss**: Controls subthreshold voltage dynamics to prevent excessive membrane fluctuations

Each loss component is weighted and combined into a total loss for gradient-based optimization.

## Configuration

Key parameters in `parameters/homeostatic-plasticity.toml`:

```toml
[simulation]
num_chunks = 500                  # Total training chunks
chunk_size = 5000                 # Timesteps per chunk
dt = 0.1                         # Timestep (ms)

[training]
batch_size = 32                  # Batch size for training
log_interval = 5                 # Log metrics every N chunks
checkpoint_interval = 50         # Save checkpoint every N chunks

[targets]
firing_rate.excitatory = 0.5     # Target firing rate per cell type (Hz)
firing_rate.inhibitory = 5.0
alpha.excitatory = 0.01          # Silent neuron penalty strength
alpha.inhibitory = 0.01
threshold_ratio.excitatory = 0.8 # Target ratio for subthreshold variance
threshold_ratio.inhibitory = 0.8

[hyperparameters]
learning_rate = 5e-4             # Adam learning rate
loss_weight.firing_rate = 1.0    # Weight for firing rate loss
loss_weight.cv = 0.5             # Weight for CV loss
loss_weight.silent_penalty = 0.1 # Weight for silent neuron penalty
loss_weight.membrane_variance = 0.1  # Weight for membrane variance loss
```

## Output Structure

```
output_dir/
├── checkpoints/
│   ├── checkpoint_latest.pt     # Most recent checkpoint
│   └── checkpoint_best.pt       # Best checkpoint (lowest loss)
├── initial_state/               # Initial state before training
│   ├── recurrent_weights.npy    # Initial recurrent weights
│   ├── feedforward_weights.npy  # Initial feedforward weights
│   └── figures/                 # Plots from initial 10s inference
│       ├── assembly_graph.png
│       ├── firing_rate_distribution.png
│       └── ...
├── figures/
│   ├── chunk_0005/              # Plots from chunk 5
│   │   ├── assembly_graph.png
│   │   ├── firing_rate_distribution.png
│   │   ├── spike_raster.png
│   │   └── ...
│   ├── chunk_0010/              # Plots from chunk 10
│   └── ...
├── weights/                     # Weight snapshots
│   ├── recurrent_epoch_0005.npy
│   ├── feedforward_epoch_0005.npy
│   └── ...
├── training_metrics.csv         # CSV log of all training metrics
├── parameters.toml              # Copy of parameters used
├── metadata.json                # Experiment metadata
└── wandb/                       # Wandb logs (if enabled)
```

## Monitoring Training

### Console Output
```
Starting training from chunk 0...
Chunk size: 5000 timesteps (0.5s)
Total chunks: 500 (50.0s total)
Batch size: 32
Log interval: 5 chunks (0.5s)
Checkpoint interval: 50 chunks (5.0s)
...

Chunk 5/500:
  Loss: 0.234567
  firing_rate_loss: 0.123456
  cv_loss: 0.045678
  silent_penalty_loss: 0.012345
  membrane_variance_loss: 0.053088
  mean_firing_rate/excitatory: 0.45 Hz
  mean_firing_rate/inhibitory: 4.8 Hz
  fraction_active/excitatory: 0.87
  Generating plots...
  ✓ Saved plots to output_dir/figures/chunk_0005
```

### Wandb Dashboard
View real-time metrics and plots at https://wandb.ai/your-username/connectome-snns-homeostatic

## Recovery from Failures

If training crashes:

1. Find your output directory
2. Check that `checkpoints/checkpoint_latest.pt` exists
3. Resume using the checkpoint (see "Resuming from Checkpoint" above)

The resumed run will:
- Continue from the exact epoch where it stopped
- Restore model weights, optimizer state, and random states
- Continue with the same best loss tracking

## Notes

- **Random state**: Both PyTorch and NumPy random states are saved/restored for full reproducibility
- **Initial states**: Membrane potentials and conductances are checkpointed for seamless continuation
- **Best model**: The `checkpoint_best.pt` always contains your best performing model, use this for analysis
- **Disk usage**: Checkpoints are ~10-50 MB each depending on network size. Old epoch checkpoints are overwritten, only `latest` and `best` are kept.
