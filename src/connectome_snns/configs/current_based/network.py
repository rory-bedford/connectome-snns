"""Current-based network layer configuration."""

from typing import Dict, List, Optional
from pydantic import BaseModel

from ..network import (
    SimpleCellTypesConfig,
    CellTypesConfig,
    TopologyConfig,
    WeightsConfig,
    ActivityConfig,
)
from .physiology import PhysiologyConfig


class RecurrentLayerConfig(BaseModel):
    """Recurrent layer configuration for current-based models."""

    cell_types: CellTypesConfig | SimpleCellTypesConfig
    physiology: Dict[str, PhysiologyConfig]
    topology: Optional[TopologyConfig] = None
    weights: Optional[WeightsConfig] = None

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_mem', 'tau_syn',
            'R', 'U_rest', 'theta', 'U_reset'.
        """
        cell_params = []
        for i, cell_type_name in enumerate(self.cell_types.names):
            physiology = self.physiology[cell_type_name]
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": i,
                    "tau_mem": physiology.tau_mem,
                    "tau_syn": physiology.tau_syn,
                    "R": physiology.R,
                    "U_rest": physiology.U_rest,
                    "theta": physiology.theta,
                    "U_reset": physiology.U_reset,
                }
            )
        return cell_params

    def get_neuron_params_for_plotting(self) -> Dict[int, Dict[str, float | str]]:
        """Get neuron parameters formatted for visualization functions.

        Returns:
            Dict mapping cell type indices to parameter dicts with keys:
            'threshold', 'rest', 'name', 'sign'.
        """
        return {
            idx: {
                "threshold": self.physiology[cell_name].theta,
                "rest": self.physiology[cell_name].U_reset,
                "name": cell_name,
                "sign": 1 if "excit" in cell_name.lower() else -1,
            }
            for idx, cell_name in enumerate(self.cell_types.names)
        }


class FeedforwardLayerConfig(BaseModel):
    """Feedforward layer configuration for current-based models."""

    cell_types: CellTypesConfig | SimpleCellTypesConfig
    physiology: Dict[str, PhysiologyConfig]
    topology: Optional[TopologyConfig] = None
    weights: Optional[WeightsConfig] = None
    activity: Optional[Dict[str, ActivityConfig]] = None

    def get_cell_params(self) -> List[Dict[str, float | int | str]]:
        """Convert physiology to list of cell parameter dicts.

        Returns:
            List of dicts with keys: 'name', 'cell_id', 'tau_syn'.
        """
        cell_params = []
        for i, cell_type_name in enumerate(self.cell_types.names):
            physiology = self.physiology[cell_type_name]
            cell_params.append(
                {
                    "name": cell_type_name,
                    "cell_id": i,
                    "tau_syn": physiology.tau_syn,
                }
            )
        return cell_params
