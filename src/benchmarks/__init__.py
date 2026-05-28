"""连续测试函数（仿真数据，Appendix C）+ 目标封装。"""
from .functions import FUNCTION_NAMES, FUNCTIONS, FunctionSpec
from .objective import ContinuousObjective, downsample_curve

__all__ = ["FUNCTIONS", "FUNCTION_NAMES", "FunctionSpec",
           "ContinuousObjective", "downsample_curve"]
