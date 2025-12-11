"""
Popper RL Validation - Source Package
Reinforcement Learning for AI System Validation Testing
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .environment import PopperValidationEnv
from src.target_systems import TargetSystemFactory

__all__ = [
    'PopperValidationEnv',
    'TargetSystemFactory'
]