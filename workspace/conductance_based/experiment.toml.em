script = "/tungstenfs/scratch/gzenke/bedfrory/connectome-snns/scripts/conductance_based/fitting_activity/train_hidden_units_em.py"
output_dir = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/activity-fitting/hidden-units-em/grid-search-3"
parameters_file = "/tungstenfs/scratch/gzenke/bedfrory/connectome-snns/parameters/conductance_based/train-hidden-units-em.toml"
log_file = "/tungstenfs/scratch/gzenke/bedfrory/dp-simulations/Dp_simulator_log.jsonl"
description = "EM algorithm with full recurrent model for E step and verified self consistent M step"
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
