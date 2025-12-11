"""
RL Agents for Validation Testing
"""

from .dqn_agent import DQNAgent
from .ucb_agent import UCBAgent

__all__ = [
    'DQNAgent',
    'UCBAgent'
]