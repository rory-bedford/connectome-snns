# Conductance-Based Dp Network Simulation

This script simulates a conductance-based LIF network with assembly structure inspired by zebrafish dorsal pallium (Dp), with feedforward inputs from mitral cells.

## Overview

The simulation consists of six main steps:

1. **Network Topology**: Generates assembly-based connectivity with recurrent E-I structure
2. **Weight Assignment**: Assigns log-normal distributed synaptic weights to connections
3. **Feedforward Inputs**: Creates Poisson spike trains from mitral cells with sparse projections
4. **Network Simulation**: Runs conductance-based LIF dynamics
5. **Data Saving**: Stores all network outputs and structure for analysis
6. **Visualization**: Generates comprehensive plots using the companion plotting script

## Key Features

### Conductance-Based Dynamics
- **Realistic synaptic transmission**: Uses conductance-based synapses with reversal potentials
- **Multiple synapse types**: Supports different synapse types per cell type (e.g., AMPA, NMDA, GABA)
- **Dynamic parameters**:
  - `tau_rise`: Rise time constant for synaptic conductance
  - `tau_decay`: Decay time constant for synaptic conductance
  - `E_syn`: Reversal potential for each synapse type
  - `g_bar`: Maximum conductance scaling factor

### Network Structure
- **Assembly organization**: Neurons grouped into assemblies with within/between connectivity
- **Cell types**: Configurable excitatory and inhibitory populations with distinct properties
- **Biologically-inspired**: Parameters adapted from Meissner-Bernard et al. (2025) [https://doi.org/10.1016/j.celrep.2025.115330](https://doi.org/10.1016/j.celrep.2025.115330)

### Feedforward Inputs
- **Mitral cell inputs**: Poisson spike trains from olfactory bulb
- **Sparse connectivity**: Configurable connection probabilities to different cell types
- **Type-specific firing rates**: Different mitral cell types can have distinct activity patterns

## Usage

### Basic Usage (via experiment runner)

```bash
python run_experiment.py parameters/conductance-based-Dp.toml
```

### Direct Usage

```python
from pathlib import Path
from scripts.conductance_based_Dp import conductance_based_Dp

conductance_based_Dp.main(
    output_dir=Path("workspace/conductance_Dp_sim"),
    params_file=Path("parameters/conductance-based-Dp.toml")
)
```

## Configuration

Key parameters in `parameters/conductance-based-Dp.toml`:

### Simulation Settings
```toml
[simulation]
dt = 0.1                          # Time step (ms)
duration = 10000                  # Simulation duration (ms)
seed = 42                         # Random seed for reproducibility
```

### Recurrent Network
```toml
[recurrent.topology]
num_neurons = 1000                # Total neurons in network
num_assemblies = 5                # Number of assemblies
conn_within = [[0.1, 0.05],       # Connection probability within assemblies
               [0.05, 0.1]]       # [E->E, E->I; I->E, I->I]
conn_between = [[0.01, 0.005],    # Connection probability between assemblies
                [0.005, 0.01]]

[recurrent.cell_types]
names = ["excitatory", "inhibitory"]
proportion = [0.8, 0.2]           # 80% E, 20% I

[recurrent.physiology.excitatory]
tau_mem = 20.0                    # Membrane time constant (ms)
theta = -50.0                     # Spike threshold (mV)
U_reset = -70.0                   # Reset potential (mV)
E_L = -70.0                       # Leak reversal potential (mV)
g_L = 10.0                        # Leak conductance (nS)
tau_ref = 2.0                     # Refractory period (ms)

[recurrent.synapses.excitatory]
names = ["AMPA", "NMDA"]          # Synapse types for excitatory cells
tau_rise = [1.0, 2.0]             # Rise times (ms)
tau_decay = [5.0, 50.0]           # Decay times (ms)
E_syn = [0.0, 0.0]                # Reversal potentials (mV)
g_bar = [1.0, 0.5]                # Maximum conductances
```

### Feedforward Inputs
```toml
[feedforward.topology]
num_neurons = 500                 # Number of mitral cells
conn_inputs = [[0.1, 0.05]]       # Connection probabilities [mitral->E, mitral->I]

[feedforward.activity.mitral]
firing_rate = 10.0                # Mitral cell firing rate (Hz)
```

## Output Structure

```
output_dir/
├── results/
│   ├── output_spikes.npy             # Network spike trains [batch, time, neurons]
│   ├── output_voltages.npy           # Membrane potentials [batch, time, neurons]
│   ├── output_currents.npy           # Synaptic currents [batch, time, neurons, types]
│   ├── output_conductances.npy       # Synaptic conductances [batch, time, neurons, types]
│   ├── input_spikes.npy              # Mitral cell spike trains
│   ├── input_conductances.npy        # Feedforward conductances
│   ├── connectivity_graph.npy        # Recurrent connectivity matrix
│   ├── weights.npy                   # Recurrent synaptic weights
│   ├── feedforward_weights.npy       # Feedforward synaptic weights
│   ├── cell_type_indices.npy         # Cell type assignment per neuron
│   └── input_cell_type_indices.npy   # Input cell type assignments
├── figures/
│   ├── 01_assembly_graph.png
│   ├── 02_weighted_connectivity.png
│   ├── 03_input_count_histogram.png
│   ├── 04_synaptic_input_histogram.png
│   ├── 05_mitral_cell_spikes.png
│   ├── 06_feedforward_connectivity.png
│   ├── 07_dp_network_spikes.png
│   ├── 08_firing_rate_distribution.png
│   ├── 09_membrane_voltages.png
│   ├── 10_synaptic_currents.png
│   ├── 11_synaptic_conductances.png
│   ├── 12_fano_factor_vs_window_size.png
│   ├── 13_cv_histogram.png
│   └── 14_isi_histogram.png
├── analysis/
│   ├── firing_rate_statistics.csv    # Mean/std firing rates by cell type
│   ├── cv_statistics.csv             # Coefficient of variation by cell type
│   └── voltage_statistics.csv        # Membrane potential statistics
├── parameters.toml                    # Copy of parameters used
└── metadata.json                      # Experiment metadata
```

## Generated Plots

### Network Structure (Figures 1-4)
- **Assembly graph**: Connectivity structure showing assembly organization
- **Weighted connectivity**: Synaptic weight distribution across assemblies
- **Input count histogram**: Distribution of incoming connections per neuron
- **Synaptic input histogram**: Total synaptic input strength distribution

### Input Analysis (Figures 5-6)
- **Mitral cell spikes**: Sample spike rasters from feedforward inputs
- **Feedforward connectivity**: Weight matrix from mitral cells to Dp network

### Output Analysis (Figures 7-8)
- **Dp network spikes**: Sample spike rasters from recurrent network
- **Firing rate distribution**: Histogram of firing rates for E and I populations

### Neuronal Dynamics (Figures 9-11)
- **Membrane voltages**: Detailed voltage traces with threshold crossings
- **Synaptic currents**: Excitatory and inhibitory current components
- **Synaptic conductances**: Time-resolved conductance traces for all synapse types

### Firing Statistics (Figures 12-14)
- **Fano factor**: Spike count variability across different time windows
- **CV histogram**: Distribution of coefficient of variation values
- **ISI histogram**: Inter-spike interval distributions

## Analysis Outputs

Statistical summaries are saved to CSV files:

### firing_rate_statistics.csv
```csv
cell_type,cell_type_name,mean_firing_rate_hz,std_firing_rate_hz,n_silent_cells
0,excitatory,5.23,2.41,15
1,inhibitory,8.67,3.12,3
```

### cv_statistics.csv
```csv
cell_type,cell_type_name,mean_cv,std_cv
0,excitatory,0.89,0.24
1,inhibitory,0.76,0.19
```

### voltage_statistics.csv
```csv
cell_type,cell_type_name,mean_of_means,std_of_means,mean_of_stds,std_of_stds
0,excitatory,-65.4,3.2,4.5,1.1
1,inhibitory,-62.1,2.8,5.2,1.3
```

## Standalone Plotting

The plotting script can be run independently to regenerate figures from saved data:

```bash
python scripts/conductance_based_Dp/conductance_based_Dp_plots.py <output_directory>
```

This is useful for:
- Regenerating figures with different parameters
- Creating additional analyses after simulation
- Debugging visualization issues

## Notes

- **GPU acceleration**: Automatically uses CUDA if available
- **Memory usage**: Large networks may require significant GPU memory (adjust batch size if needed)
- **Reproducibility**: Set `seed` in config for deterministic results
- **Parameter validation**: Script validates parameter consistency at startup
- **Conductance-based specifics**: Unlike current-based models, weights are always positive (sign determined by reversal potential)
