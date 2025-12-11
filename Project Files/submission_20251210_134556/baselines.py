"""
Baseline testing strategies for comparison - FIXED VERSION
"""

import numpy as np
from typing import Dict


class BaselineStrategy:
    """Base class for baseline strategies"""

    def __init__(self, config: Dict):
        self.config = config
        self.test_categories = [
            'adversarial', 'edge_case', 'boundary', 'distribution_shift',
            'performance', 'logic_error', 'coverage_guided', 'random',
            'metamorphic', 'stress_test'
        ]

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action - to be implemented by subclasses"""
        raise NotImplementedError


class RandomStrategy(BaselineStrategy):
    """Random test selection strategy"""

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select random action"""
        return np.array([
            np.random.randint(0, len(self.test_categories)),
            np.random.uniform(0, 1),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
        ])


class CoverageGuidedStrategy(BaselineStrategy):
    """Coverage-guided test selection strategy"""

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action to maximize coverage"""
        coverage_map = state[:10]
        category_idx = np.argmin(coverage_map)

        low_coverage_dims = np.where(coverage_map < 0.5)[0]
        if len(low_coverage_dims) > 0:
            target = np.random.choice(low_coverage_dims)
            param1 = (target - 5.0) * 2.0
        else:
            param1 = np.random.uniform(-10, 10)

        return np.array([
            float(category_idx), 0.7, param1,
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])


class MetamorphicStrategy(BaselineStrategy):
    """Metamorphic testing strategy"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.previous_inputs = []
        self.previous_outputs = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action based on metamorphic relations"""
        # FIXED: Check if we have previous inputs AND that they're valid
        if len(self.previous_inputs) > 0 and self.previous_inputs[-1] is not None and np.random.random() < 0.5:
            base_input = self.previous_inputs[-1]

            # FIXED: Make sure base_input is an array with elements
            if isinstance(base_input, np.ndarray) and len(base_input) > 0:
                transformation = np.random.choice(['scale', 'shift', 'permute', 'negate'])

                if transformation == 'scale':
                    param1 = base_input[0] * np.random.uniform(0.5, 2.0)
                elif transformation == 'shift':
                    param1 = base_input[0] + np.random.uniform(-2, 2)
                elif transformation == 'negate':
                    param1 = -base_input[0]
                else:  # permute
                    param1 = base_input[0] + np.random.randn() * 0.5
            else:
                # Fallback if base_input is invalid
                param1 = np.random.uniform(-10, 10)
        else:
            param1 = np.random.uniform(-10, 10)

        # Focus on metamorphic category
        category_idx = self.test_categories.index('metamorphic')

        return np.array([
            float(category_idx), 0.6, param1,
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])

    def update(self, test_input: np.ndarray, test_output: Dict):
        """Store test results for metamorphic transformations"""
        # FIXED: Only store valid inputs
        if test_input is not None:
            self.previous_inputs.append(test_input)
            self.previous_outputs.append(test_output)

            # Keep only recent history
            if len(self.previous_inputs) > 20:
                self.previous_inputs.pop(0)
                self.previous_outputs.pop(0)


class AdaptiveRandomStrategy(BaselineStrategy):
    """Adaptive random strategy that learns from feedback"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.category_success = np.ones(len(self.test_categories))
        self.category_counts = np.ones(len(self.test_categories))

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action with probability proportional to success rate"""
        success_probs = self.category_success / self.category_counts
        success_probs = success_probs / np.sum(success_probs)

        category_idx = np.random.choice(len(self.test_categories), p=success_probs)

        return np.array([
            float(category_idx),
            np.random.uniform(0.3, 0.9),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10)
        ])

    def update(self, action: np.ndarray, reward: float):
        """Update success statistics"""
        category_idx = int(action[0])
        self.category_counts[category_idx] += 1

        alpha = 0.1
        if reward > 0:
            self.category_success[category_idx] = (
                (1 - alpha) * self.category_success[category_idx] + alpha * reward
            )


def create_baseline(strategy_name: str, config: Dict) -> BaselineStrategy:
    """Factory function to create baseline strategies"""
    strategies = {
        'random': RandomStrategy,
        'coverage_guided': CoverageGuidedStrategy,
        'metamorphic': MetamorphicStrategy,
        'adaptive_random': AdaptiveRandomStrategy
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown baseline strategy: {strategy_name}")

    return strategies[strategy_name](config)