"""
Gym Environment for Popper Validation Testing
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple
from .target_systems import TargetSystemFactory, BugSeverity


class PopperValidationEnv(gym.Env):
    """Custom Gym environment for RL-based AI validation testing"""

    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.max_steps = config['environment']['max_steps_per_episode']
        self.test_budget = config['environment']['test_budget']

        target_config = config['target_system']
        self.target_system = TargetSystemFactory.create(
            target_config['type'],
            input_dim=target_config.get('input_dim', 10),
            output_dim=target_config.get('output_dim', 2)
        )

        self.test_categories = [
            'adversarial', 'edge_case', 'boundary', 'distribution_shift',
            'performance', 'logic_error', 'coverage_guided', 'random',
            'metamorphic', 'stress_test'
        ]
        self.n_categories = len(self.test_categories)

        self.action_space = spaces.Box(
            low=np.array([0, 0, -10, -10, -10]),
            high=np.array([self.n_categories - 1, 1, 10, 10, 10]),
            dtype=np.float32
        )

        state_dim = 10 + 10 + 3 + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.reset()
        self.discovered_bugs = []
        self.bug_types_found = set()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.tests_conducted = 0
        self.bugs_found = 0
        self.discovered_bugs = []
        self.bug_types_found = set()

        self.coverage_map = np.zeros(10)
        self.input_space_coverage = np.zeros(10)
        self.bug_history = np.zeros(10)

        self.remaining_budget = self.test_budget
        self.time_elapsed = 0
        self.confidence_scores = np.ones(10) * 0.5

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one test based on action"""
        self.current_step += 1
        self.tests_conducted += 1

        category_idx = int(np.clip(action[0], 0, self.n_categories - 1))
        intensity = np.clip(action[1], 0, 1)
        test_params = action[2:]

        category = self.test_categories[category_idx]
        test_input = self._generate_test_input(category, intensity, test_params)
        test_result = self._execute_test(test_input, category)

        self._update_state(test_result, category_idx)
        reward = self._calculate_reward(test_result)

        done = (
                self.current_step >= self.max_steps or
                self.remaining_budget <= 0 or
                self.bugs_found >= 20
        )

        info = {
            'bugs_found': self.bugs_found,
            'tests_conducted': self.tests_conducted,
            'coverage': np.mean(self.coverage_map),
            'test_result': test_result,
            'category': category
        }

        return self._get_state(), reward, done, info

    def _generate_test_input(self, category: str, intensity: float, params: np.ndarray) -> np.ndarray:
        """Generate test input based on category and parameters"""
        input_dim = self.config['target_system']['input_dim']

        if category == 'adversarial':
            base = np.array([0.5, 0.5] + [0.0] * (input_dim - 2))
            noise = np.random.randn(input_dim) * intensity * 0.1
            test_input = base + noise
        elif category == 'edge_case':
            test_input = np.random.randn(input_dim) * (5.0 + intensity * 5.0)
        elif category == 'boundary':
            target_sum = 10.0 + params[0] * 0.5
            test_input = np.random.randn(input_dim)
            test_input = test_input / np.sum(test_input) * target_sum
        elif category == 'distribution_shift':
            test_input = np.random.randn(input_dim) - (2.0 + intensity)
        elif category == 'logic_error':
            test_input = np.random.randn(input_dim)
            test_input[0] = -5.0 - params[0]
            test_input[1] = 1.0 + params[1]
        elif category == 'coverage_guided':
            low_coverage_dims = np.where(self.coverage_map < 0.5)[0]
            test_input = np.random.randn(input_dim)
            if len(low_coverage_dims) > 0:
                idx = np.random.choice(low_coverage_dims)
                test_input[idx % input_dim] = params[0]
        else:
            test_input = np.random.randn(input_dim) * (1.0 + intensity)

        return test_input

    def _execute_test(self, test_input: np.ndarray, category: str) -> Dict:
        """Execute test on target system"""
        prediction, has_bug, bug_info = self.target_system.predict(test_input)

        input_region = int(np.clip(np.mean(test_input) + 5, 0, 9))
        self.input_space_coverage[input_region] = 1.0

        cost = 1.0

        is_novel = False
        if has_bug and bug_info['type'] not in self.bug_types_found:
            is_novel = True
            self.bug_types_found.add(bug_info['type'])

        return {
            'has_bug': has_bug,
            'bug_info': bug_info,
            'is_novel': is_novel,
            'cost': cost,
            'prediction': prediction,
            'test_input': test_input,
            'category': category,
            'new_coverage': 0.0
        }

    def _update_state(self, test_result: Dict, category_idx: int):
        """Update internal state based on test result"""
        self.coverage_map[category_idx] = min(1.0, self.coverage_map[category_idx] + 0.1)

        self.bug_history = np.roll(self.bug_history, -1)
        self.bug_history[-1] = 1.0 if test_result['has_bug'] else 0.0

        self.remaining_budget -= test_result['cost']
        self.time_elapsed += 1

        if test_result['has_bug']:
            self.bugs_found += 1
            self.discovered_bugs.append(test_result['bug_info'])
            self.confidence_scores[category_idx] = max(0.0, self.confidence_scores[category_idx] - 0.1)
        else:
            self.confidence_scores[category_idx] = min(1.0, self.confidence_scores[category_idx] + 0.05)

    def _calculate_reward(self, test_result: Dict) -> float:
        """Calculate reward based on test result"""
        reward = 0.0
        rewards_config = self.config['rewards']

        if test_result['has_bug']:
            severity = test_result['bug_info']['severity']
            severity_rewards = rewards_config['bug_found']
            reward += severity_rewards.get(severity.value, 0)

            if test_result['is_novel']:
                reward += rewards_config['novelty_bonus']

        coverage_increase = test_result['new_coverage']
        reward += coverage_increase * rewards_config['coverage_bonus']
        reward += test_result['cost'] * rewards_config['cost_penalty']

        state_novelty = 1.0 - np.mean(self.coverage_map)
        reward += state_novelty * rewards_config['exploration_bonus']

        return reward

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = np.concatenate([
            self.coverage_map,
            self.bug_history,
            np.array([
                self.remaining_budget / self.test_budget,
                self.time_elapsed / self.max_steps,
                self.bugs_found / 20.0
            ]),
            self.confidence_scores
        ])
        return state.astype(np.float32)

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n{'=' * 50}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Tests Conducted: {self.tests_conducted}")
            print(f"Bugs Found: {self.bugs_found}")
            print(f"Coverage: {np.mean(self.coverage_map):.2%}")
            print(f"Remaining Budget: {self.remaining_budget:.1f}")
            print(f"{'=' * 50}\n")

    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            'bugs_found': self.bugs_found,
            'tests_conducted': self.tests_conducted,
            'bug_discovery_rate': self.bugs_found / max(1, self.tests_conducted),
            'coverage': np.mean(self.coverage_map),
            'unique_bug_types': len(self.bug_types_found),
            'efficiency': self.bugs_found / max(1, self.time_elapsed)
        }