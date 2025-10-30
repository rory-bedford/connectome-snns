# Current-Based Dp Network Simulation

This script simulates a current-based LIF network with assembly structure inspired by zebrafish dorsal pallium (Dp), with feedforward inputs from mitral cells.

## Overview

The simulation consists of six main steps:

1. **Network Topology**: Generates assembly-based connectivity with recurrent E-I structure
2. **Weight Assignment**: Assigns signed log-normal distributed synaptic weights to connections
3. **Feedforward Inputs**: Creates Poisson spike trains from mitral cells with sparse projections
4. **Network Simulation**: Runs current-based LIF dynamics
5. **Data Saving**: Stores all network outputs and structure for analysis
6. **Visualization**: Generates comprehensive plots using the companion plotting script

## Key Features

### Current-Based Dynamics
- **Simplified synaptic transmission**: Uses exponentially decaying synaptic currents
- **Signed weights**: Weights carry excitatory (+) or inhibitory (-) sign
- **Dynamic parameters**:
  - `tau_mem`: Membrane time constant
  - `tau_syn`: Synaptic time constant
  - `R`: Membrane resistance
  - `theta`: Spike threshold
  - `U_reset`: Reset potential

### Network Structure
- **Assembly organization**: Neurons grouped into assemblies with within/between connectivity
- **Cell types**: Configurable excitatory and inhibitory populations with distinct properties
- **Biologically-inspired**: Parameters adapted from Meissner-Bernard et al. (2025) [https://doi.org/10.1016/j.celrep.2025.115330](https://doi.org/10.1016/j.celrep.2025.115330)

### Feedforward Inputs
- **Mitral cell inputs**: Poisson spike trains from olfactory bulb
- **Sparse connectivity**: Configurable connection probabilities to different cell types
- **Excitatory projections**: All feedforward inputs are excitatory

## Usage

### Basic Usage (via experiment runner)

```bash
python run_experiment.py parameters/current-based-Dp.toml
```

### Direct Usage

```python
from pathlib import Path
from scripts.current_based_Dp import current_based_Dp

current_based_Dp.main(
    output_dir=Path("workspace/current_Dp_sim"),
    params_file=Path("parameters/current-based-Dp.toml")
)
```

## Configuration

Key parameters in `parameters/current-based-Dp.toml`:

### Simulation Settings
```toml
[simulation]
dt = 0.1                          # Time step (ms)
duration = 10000                  # Simulation duration (ms)
seed = 42                         # Random seed for reproducibility
```

### Recurrent Network
```toml
[connectome.topology]
num_neurons = 1000                # Total neurons in network
num_assemblies = 5                # Number of assemblies
conn_within = [[0.1, 0.05],       # Connection probability within assemblies
               [0.05, 0.1]]       # [E->E, E->I; I->E, I->I]
conn_between = [[0.01, 0.005],    # Connection probability between assemblies
                [0.005, 0.01]]

[connectome.cell_types]
names = ["excitatory", "inhibitory"]
signs = [1, -1]                   # +1 for excitatory, -1 for inhibitory
proportion = [0.8, 0.2]           # 80% E, 20% I

[physiology.excitatory]
tau_mem = 20.0                    # Membrane time constant (ms)
tau_syn = 5.0                     # Synaptic time constant (ms)
R = 1.0                           # Membrane resistance (MΩ)
U_rest = -70.0                    # Resting potential (mV)
theta = -50.0                     # Spike threshold (mV)
U_reset = -70.0                   # Reset potential (mV)

[physiology.inhibitory]
tau_mem = 10.0                    # Faster membrane dynamics
tau_syn = 5.0
R = 1.0
U_rest = -70.0
theta = -50.0
U_reset = -70.0
```

### Feedforward Inputs
```toml
[inputs.topology]
num_neurons = 500                 # Number of mitral cells
conn_inputs = [[0.1, 0.05]]       # Connection probabilities [mitral->E, mitral->I]

[inputs.activity]
firing_rates = [10.0]             # Mitral cell firing rate (Hz)
```

### Weights
```toml
[connectome.weights]
w_mu = [[2.0, 2.0],               # Mean log-normal weights [E->E, E->I; I->E, I->I]
        [2.0, 2.0]]
w_sigma = [[0.5, 0.5],            # Std of log-normal weights
           [0.5, 0.5]]

[inputs.weights]
w_mu = [[2.0, 2.0]]               # Mean feedforward weights [mitral->E, mitral->I]
w_sigma = [[0.5, 0.5]]            # Std feedforward weights
```

## Output Structure

```
output_dir/
├── results/                           # Note: current-based saves directly to output_dir
│   ├── output_spikes.npy              # Network spike trains [batch, time, neurons]
│   ├── output_voltages.npy            # Membrane potentials [batch, time, neurons]
│   ├── output_I.npy                   # Synaptic currents [batch, time, neurons, E/I]
│   ├── output_I_FF.npy                # Feedforward currents [batch, time, neurons]
│   ├── input_spikes.npy               # Mitral cell spike trains
│   ├── connectivity_graph.npy         # Recurrent connectivity matrix
│   ├── weights.npy                    # Recurrent synaptic weights (signed)
│   ├── feedforward_weights.npy        # Feedforward synaptic weights
│   ├── neuron_types.npy               # Neuron signs (+1 E, -1 I)
│   └── cell_type_indices.npy          # Cell type assignment per neuron
├── figures/                            # Note: plots saved directly to output_dir
│   ├── 01_assembly_graph.png
│   ├── 02_weighted_connectivity.png
│   ├── 03_synaptic_input_histogram.png
│   ├── 04_mitral_cell_spikes.png
│   ├── 05_feedforward_connectivity.png
│   ├── 06_dp_network_spikes.png
│   ├── 07_firing_rate_distribution.png
│   ├── 08_membrane_voltages.png
│   └── 09_synaptic_currents.png
├── parameters.toml                     # Copy of parameters used
└── metadata.json                       # Experiment metadata
```

## Generated Plots

### Network Structure (Figures 1-3)
- **Assembly graph**: Connectivity structure showing assembly organization (color-coded by sign)
- **Weighted connectivity**: Synaptic weight distribution across assemblies (red=inhibitory, blue=excitatory)
- **Synaptic input histogram**: Distribution of total synaptic input per neuron

### Input Analysis (Figures 4-5)
- **Mitral cell spikes**: Sample spike rasters from feedforward inputs
- **Feedforward connectivity**: Weight matrix from mitral cells to Dp network

### Output Analysis (Figures 6-7)
- **Dp network spikes**: Sample spike rasters from recurrent network
- **Firing rate distribution**: Histogram of firing rates for E and I populations

### Neuronal Dynamics (Figures 8-9)
- **Membrane voltages**: Detailed voltage traces with threshold crossings
- **Synaptic currents**: Excitatory and inhibitory current components separated

## Differences from Conductance-Based Model

### Simplified Dynamics
- **Single synapse per cell type**: Each neuron type has one synaptic time constant
- **No reversal potentials**: Sign determined by weight, not conductance reversal
- **Faster computation**: Simpler dynamics allow for larger networks

### Weight Representation
- **Signed weights**: Weights can be positive (E) or negative (I)
- **Single time constant**: `tau_syn` determines both rise and decay (exponential)

### Use Cases
- **Rapid prototyping**: Faster simulations for parameter exploration
- **Large-scale networks**: Reduced memory requirements
- **Abstract models**: When detailed conductance dynamics not critical

## Standalone Plotting

The plotting script can be run independently to regenerate figures from saved data:

```bash
python scripts/current_based_Dp/current_based_Dp_plots.py <output_directory>
```

This is useful for:
- Regenerating figures with different parameters
- Creating additional analyses after simulation
- Debugging visualization issues

## Notes

- **GPU acceleration**: Automatically uses CUDA if available
- **Memory efficiency**: Current-based models use ~30% less memory than conductance-based
- **Reproducibility**: Set `seed` in config for deterministic results
- **Parameter validation**: Script validates parameter consistency at startup
- **Sign convention**: Positive weights = excitatory, negative weights = inhibitory
- **Output location**: Unlike conductance-based model, files saved directly to `output_dir/` (not `output_dir/results/`)

## Converting to Conductance-Based

To convert a current-based simulation to conductance-based:

1. Use `parameters/conductance-based-Dp.toml` as template
2. Map current-based parameters:
   - `tau_syn` → `tau_rise` and `tau_decay` for each synapse type
   - Add `E_syn` (reversal potentials): 0 mV for E, -70 mV for I
   - Add `g_bar` (conductance scaling)
   - Remove `R` (resistance)
3. Specify multiple synapse types (e.g., AMPA, NMDA, GABA)
4. Run with `conductance_based_Dp.py` instead

## References

- Meissner-Bernard, C., et al. (2025). Circuit mechanisms underlying embryonic olfactory learning in zebrafish. *Cell Reports*, 44(2), 115330. [https://doi.org/10.1016/j.celrep.2025.115330](https://doi.org/10.1016/j.celrep.2025.115330)
