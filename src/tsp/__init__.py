"""真实数据：TSPLIB 真实城市坐标 TSP。"""
from .data import OPTIMA, distance_matrix, load_coords
from .problem import TSPProblem

__all__ = ["load_coords", "distance_matrix", "OPTIMA", "TSPProblem"]
