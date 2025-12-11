"""
Upper Confidence Bound (UCB) Agent for Validation Testing
"""

import numpy as np
from typing import Dict, List
import json


class UCBAgent:
    """UCB Agent for balancing exploration and exploitation in test category selection"""

    def __init__(self, n_arms: int, config: Dict):
        self.n_arms = n_arms
        self.config = config
        self.c = config['ucb']['c']

        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_count = 0

        self.arm_history = [[] for _ in range(n_arms)]
        self.rewards_history = []

        self.test_categories = [
            'adversarial', 'edge_case', 'boundary', 'distribution_shift',
            'performance', 'logic_error', 'coverage_guided', 'random',
            'metamorphic', 'stress_test'
        ]

    def select_action(self, state: np.ndarray = None) -> np.ndarray:
        """Select action using UCB1 algorithm"""
        if self.total_count == 0:
            category_idx = self.total_count % self.n_arms
        else:
            ucb_values = self._calculate_ucb_values()
            category_idx = np.argmax(ucb_values)

        action = self._generate_action_for_category(category_idx, state)
        return action

    def _calculate_ucb_values(self) -> np.ndarray:
        """Calculate UCB values: X̄ᵢ + c * √(2 * ln(n) / nᵢ)"""
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb_values[i] = float('inf')
            else:
                exploitation_term = self.values[i]
                exploration_term = self.c * np.sqrt(2 * np.log(self.total_count) / self.counts[i])
                ucb_values[i] = exploitation_term + exploration_term

        return ucb_values

    def _generate_action_for_category(self, category_idx: int, state: np.ndarray = None) -> np.ndarray:
        """Generate full action vector for selected category"""
        if self.counts[category_idx] > 0:
            success_rate = self.values[category_idx] / 100.0
            intensity = 0.5 + 0.5 * success_rate
        else:
            intensity = 0.5

        if state is not None and len(state) > 10:
            coverage = state[:10]
            low_coverage_dims = np.where(coverage < 0.5)[0]
            if len(low_coverage_dims) > 0:
                target_dim = np.random.choice(low_coverage_dims)
                param1 = (target_dim - 5.0) * 2.0
            else:
                param1 = np.random.uniform(-5, 5)
        else:
            param1 = np.random.uniform(-5, 5)

        param2 = np.random.uniform(-5, 5)
        param3 = np.random.uniform(-5, 5)

        return np.array([float(category_idx), intensity, param1, param2, param3])

    def update(self, action: np.ndarray, reward: float):
        """Update arm statistics based on observed reward"""
        category_idx = int(action[0])

        self.counts[category_idx] += 1
        self.total_count += 1

        n = self.counts[category_idx]
        old_value = self.values[category_idx]
        self.values[category_idx] = old_value + (reward - old_value) / n

        self.arm_history[category_idx].append(reward)
        self.rewards_history.append({
            'step': self.total_count,
            'category': self.test_categories[category_idx],
            'category_idx': category_idx,
            'reward': reward
        })

    def get_stats(self) -> Dict:
        """Get current UCB agent statistics"""
        ucb_values = self._calculate_ucb_values() if self.total_count > 0 else np.zeros(self.n_arms)
        ucb_values = np.where(np.isinf(ucb_values), 1000.0, ucb_values)

        return {
            'total_pulls': self.total_count,
            'arm_counts': self.counts.tolist(),
            'arm_values': self.values.tolist(),
            'ucb_values': ucb_values.tolist(),
            'best_arm': int(np.argmax(self.values)),
            'best_category': self.test_categories[int(np.argmax(self.values))],
            'exploration_rate': np.sum(self.counts == 0) / self.n_arms
        }

    def get_arm_statistics(self) -> List[Dict]:
        """Get detailed statistics for each arm"""
        stats = []
        for i in range(self.n_arms):
            arm_stat = {
                'category': self.test_categories[i],
                'index': i,
                'pulls': int(self.counts[i]),
                'avg_reward': float(self.values[i]),
                'total_reward': float(self.values[i] * self.counts[i]),
                'pull_rate': float(self.counts[i] / max(1, self.total_count))
            }

            if len(self.arm_history[i]) > 0:
                arm_stat['reward_std'] = float(np.std(self.arm_history[i]))
                arm_stat['reward_min'] = float(np.min(self.arm_history[i]))
                arm_stat['reward_max'] = float(np.max(self.arm_history[i]))

            stats.append(arm_stat)

        return stats

    def save(self, filepath: str):
        """Save UCB agent state"""
        state = {
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'total_count': self.total_count,
            'arm_history': [list(h) for h in self.arm_history],
            'rewards_history': self.rewards_history
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: str):
        """Load UCB agent state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.counts = np.array(state['counts'])
        self.values = np.array(state['values'])
        self.total_count = state['total_count']
        self.arm_history = [list(h) for h in state['arm_history']]
        self.rewards_history = state['rewards_history']