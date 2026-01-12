# Student Network Training (Fitting Activity)

This workflow trains a connectome-constrained conductance-based LIF network (student) to reproduce target spike train activity from a pre-generated teacher network. The student network's connectivity structure is loaded from disk and kept fixed, while synaptic weights are optimized using gradient-based learning through backpropagation with surrogate gradients.

**Workflow Position:** Stages 3-4 - After homeostatic_plasticity training. The optimized weights from Stage 2 are used to generate teacher activity (Stage 3), which the student then learns to match (Stage 4).

## Complete Workflow Pipeline

1. **network_inference** → Generate/explore connectome structures (grid search)
2. **homeostatic_plasticity** → Optimize weights for target activity patterns  
3. **generate_teacher_activity** → Generate target spike trains using optimized weights
4. **train_student** → Train student network to match teacher patterns

## Scripts

### 1. `generate_teacher_activity.py` (Stage 3)
Generates teacher network spike trains from a pre-existing connectome structure (typically from homeostatic plasticity training). Loads network connectivity from disk and simulates activity with odour-modulated input patterns.

### 2. `train_student.py` (Stage 4)
Trains a student network to match teacher spike patterns using gradient descent with multiple loss functions (firing rate matching, van Rossum distance, silent neuron penalty).

### 3. `plot_trained_network_vs_target.py` (Post-Training Analysis)
**Must be run after training is complete** on a trained experiment folder. Compares trained student network performance against the target (teacher) network to evaluate learning success.

**Inputs:**
- `experiment_dir`: Path to trained experiment folder (contains `checkpoints/checkpoint_best.pt`)
- `output_dir`: Where to save comparison plots
- `teacher_params_file`: Path to `generate-teacher-activity.toml` (for odour configurations)

**What it does:**
- Generates fresh odour 1 input spikes (using `HomogeneousPoissonSpikeDataLoader`)
- Runs target network twice with different noise (batch size 2)
- Runs trained network with identical input spikes as first target presentation
- Creates three comparison plots:
  1. **Variability comparison dashboard** (side-by-side):
     - Left: Poisson variability (target odour 1 vs repeat) - shows intrinsic noise
     - Right: Learning variability (target vs trained with same inputs) - shows learning quality
  2. **Spike raster plot**: First 10 neurons with target/trained pairs interleaved

**Usage:**
```bash
python scripts/fitting_activity/plot_trained_network_vs_target.py \
    workspace/student_training_run \
    workspace/student_training_run/comparison_plots \
    parameters/generate-teacher-activity.toml
```

**Interpretation:** If learning variability is similar to Poisson variability, the student has learned as well as possible given stochastic spike generation.

### 4. `plot_odourants.py` (Teacher Network Analysis)
Analyzes how the teacher network responds to different odour inputs vs baseline activity. Useful for understanding odour selectivity before student training.

**Inputs:**
- `experiment_dir`: Path to teacher activity generation folder (contains `inputs/network_structure.npz` and `parameters.toml`)

**What it does:**
- Generates network responses to:
  - Odourant 1 (two independent noise realizations)
  - Baseline (homogeneous Poisson input without odour modulation)
- Creates comparison plots:
  1. **Input firing rate histogram**: Distribution of odourant 1 input rates
  2. **Odourant comparison dashboard** (side-by-side):
     - Left: Odourant 1 vs Baseline - shows odour selectivity
     - Right: Odourant 1 vs Odourant 1 (repeat) - shows response reliability

**Usage:**
```bash
python scripts/fitting_activity/plot_odourants.py \
    workspace/teacher_activity_generation
```

Outputs saved to `experiment_dir/figures/`

### 5. `resume_training.py`
Resumes student training from a checkpoint after interruption.

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
python scripts/run_experiment.py parameters/fitting-activity.toml
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
