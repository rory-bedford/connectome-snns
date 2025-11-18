# Fitting Activity Training

This script trains a connectome-constrained conductance-based LIF network to fit target activity patterns in a teacher-student learning framework. The student network's connectome structure is constrained by biological connectivity, while learning to reproduce target firing statistics through gradient-based weight optimization.

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
  - Losses (CV, firing rate, total)
  - Network statistics (mean/std firing rates, CVs, active fraction)
  - All training plots (updated every `accumulation_interval`)
- **Config**: All parameters from TOML file are logged to wandb

### Plotting
- Generates comprehensive plots every `accumulation_interval` epochs
- Plots include:
  - Network structure (assembly graphs, connectivity matrices)
  - Input analysis (mitral cell spikes, feedforward connectivity)
  - Output analysis (spike rasters, firing rate distributions)
  - Neuronal dynamics (membrane voltages)
  - Firing statistics (Fano factors, CV histograms, ISI distributions)
- All plots saved to `output_dir/figures/epoch_XXXX/`
- All plots automatically logged to wandb

## Usage

### Basic Usage (via experiment runner)

```bash
# Run with default parameters
python run_experiment.py parameters/fitting-activity.toml
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
from scripts.fitting_activity import main

main(
    output_dir=Path("workspace/my_experiment"),
    params_file=Path("parameters/fitting-activity.toml"),
    resume_from=Path("workspace/my_experiment/checkpoints/checkpoint_latest.pt"),
    use_wandb=True
)
```

### Disabling wandb

```python
main(
    output_dir=Path("workspace/my_experiment"),
    params_file=Path("parameters/fitting-activity.toml"),
    use_wandb=False  # Disable wandb logging
)
```

## Configuration

Key parameters in `parameters/fitting-activity.toml`:

```toml
[simulation]
epochs = 500                      # Total training epochs
accumulation_interval = 5         # Update weights every N epochs, generate plots
checkpoint_interval = 50          # Save checkpoint every N epochs

[targets]
firing_rates = 0.5               # Target firing rate (Hz) per cell type
cvs = 1.0                        # Target coefficient of variation
alpha = 0.01                     # Silent neuron penalty strength
threshold_ratio = 0.8            # Target ratio for subthreshold variance

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
├── figures/
│   ├── epoch_0005/              # Plots from epoch 5
│   │   ├── assembly_graph.png
│   │   ├── firing_rate_distribution.png
│   │   └── ...
│   ├── epoch_0010/              # Plots from epoch 10
│   └── ...
├── parameters.toml              # Copy of parameters used
├── metadata.json                # Experiment metadata
└── wandb/                       # Wandb logs (if enabled)
```

## Monitoring Training

### Console Output
```
Starting training from epoch 0...
Target firing rate: 0.5 Hz
Target CV: 1.0
...

Epoch 5/500:
  CV Loss: 0.123456
  FR Loss: 0.234567
  Total Loss: 0.179012
  Mean FR: 0.45 Hz
  Mean CV: 0.98
  Active fraction: 0.87
  Generating plots...
  Saved plots to output_dir/figures/epoch_0005
```

### Wandb Dashboard
View real-time metrics and plots at https://wandb.ai/your-username/connectome-snns-fitting-activity

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
