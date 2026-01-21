"""Neural operator models for solving PDEs."""

from models.base import OperatorModel
from models.fno2d import FNO2D
from models.fno2d_time import FNO2DTimeCond
from models.graph_fno import GraphFNO

__all__ = ["OperatorModel", "FNO2D", "FNO2DTimeCond", "GraphFNO"]
