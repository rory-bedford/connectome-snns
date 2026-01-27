script = "/tungstenfs/scratch/gzenke/bedfrory/connectome-snns/scripts/conductance_based/fitting_activity/train_hidden_units_em_full_loss.py"
output_dir = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/activity-fitting/hidden-units-em/full-loss-grid-search-2"
parameters_file = "/tungstenfs/scratch/gzenke/bedfrory/connectome-snns/parameters/conductance_based/train-hidden-units-em.toml"
log_file = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/Dp_simulator_log.jsonl"
description = "EM-like training of hidden units: E-step infers hidden activity, M-step trains visible units. E-step using visible spikes as feedforward inputs. M-step using loss on hidden and visible units."

[data]

[[data.inputs]]
path = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/activity-fitting/teacher-activity/inputs/network_structure.npz"
strategy = "symlink"

[[data.inputs]]
path = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/activity-fitting/teacher-activity/results/spike_data.zarr"
strategy = "symlink"

[wandb]
enabled = true
project = "hidden-units-em"
entity = ""
tags = ["em", "hidden-units"]
notes = ""
