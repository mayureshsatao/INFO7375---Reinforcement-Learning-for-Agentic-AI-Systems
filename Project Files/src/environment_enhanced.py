"""
Enhanced Gym Environment for Popper Validation Testing
Includes progressive difficulty, custom tools, and advanced metrics
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple
from target_systems import TargetSystemFactory, BugSeverity
from custom_tools import CustomToolManager
from multi_agent import FallbackManager


class EnhancedPopperValidationEnv(gym.Env):
    """
    Enhanced environment with:
    - Progressive difficulty
    - Custom tool integration
    - Advanced metrics
    - Fallback strategies
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.max_steps = config['environment']['max_steps_per_episode']
        self.test_budget = config['environment']['test_budget']
        self.progressive_difficulty = config['environment'].get('progressive_difficulty', False)

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

        state_dim = 10 + 10 + 3 + 10 + 5  # Added 5 for difficulty/tools
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Initialize custom tools
        self.custom_tools = CustomToolManager(self.target_system, config)

        # Initialize fallback manager
        self.fallback_manager = FallbackManager(config)

        # Difficulty tracking
        self.current_difficulty = 1.0
        self.episode_count = 0

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
        self.recent_bugs_window = []  # Track recent bug discoveries

        self.coverage_map = np.zeros(10)
        self.input_space_coverage = np.zeros(10)
        self.bug_history = np.zeros(10)

        self.remaining_budget = self.test_budget
        self.time_elapsed = 0
        self.confidence_scores = np.ones(10) * 0.5

        # Track category usage for diversity
        self.category_usage = np.zeros(10)

        # Progressive difficulty
        self.episode_count += 1
        if self.progressive_difficulty:
            self.current_difficulty = min(5.0, 1.0 + (self.episode_count / 100))

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one test based on action"""
        self.current_step += 1
        self.tests_conducted += 1

        category_idx = int(np.clip(action[0], 0, self.n_categories - 1))
        intensity = np.clip(action[1], 0, 1)
        test_params = action[2:]

        category = self.test_categories[category_idx]

        # Update category usage for diversity tracking
        self.category_usage[category_idx] += 1

        # Check fallback strategies
        should_fallback, fallback_type, fallback_action = self.fallback_manager.check_fallbacks(
            self._get_state(),
            len(self.recent_bugs_window),
            self.current_step
        )

        if should_fallback:
            action = fallback_action
            category_idx = int(action[0])
            category = self.test_categories[category_idx]

        # Generate test input
        test_input = self._generate_test_input(category, intensity, test_params)

        # Use custom tools if applicable
        if category == 'adversarial' and self.custom_tools.adversarial_generator:
            test_input, has_bug, attack_info = self.custom_tools.adversarial_generator.generate_adversarial_test(test_input)
            test_result = self._execute_test(test_input, category)
            test_result['used_custom_tool'] = 'adversarial_generator'
        else:
            test_result = self._execute_test(test_input, category)
            test_result['used_custom_tool'] = None

        # Record in coverage analyzer
        if self.custom_tools.coverage_analyzer:
            self.custom_tools.coverage_analyzer.record_test(test_input)

        # Add to mutation engine population if successful
        if test_result['has_bug'] and self.custom_tools.mutation_engine:
            bug_severity = test_result['bug_info']['severity']
            fitness = self._severity_to_fitness(bug_severity)
            self.custom_tools.mutation_engine.add_to_population(test_input, fitness)

        # Update state
        self._update_state(test_result, category_idx)

        # Calculate reward with enhancements
        reward = self._calculate_enhanced_reward(test_result, fallback_type)

        # Track recent bugs for fallback
        if test_result['has_bug']:
            self.recent_bugs_window.append(self.current_step)
            if len(self.recent_bugs_window) > 10:
                self.recent_bugs_window.pop(0)
            self.fallback_manager.reset_stagnation_counter()

        # Progressive difficulty increases bug threshold
        bug_threshold = int(20 * self.current_difficulty)

        done = (
            self.current_step >= self.max_steps or
            self.remaining_budget <= 0 or
            self.bugs_found >= bug_threshold
        )

        info = {
            'bugs_found': self.bugs_found,
            'tests_conducted': self.tests_conducted,
            'coverage': np.mean(self.coverage_map),
            'test_result': test_result,
            'category': category,
            'difficulty': self.current_difficulty,
            'fallback_active': should_fallback,
            'fallback_type': fallback_type,
            'custom_tool_stats': self.custom_tools.get_all_statistics(),
            'diversity_score': self._calculate_diversity(),
            'fallback_stats': self.fallback_manager.get_statistics()
        }

        return self._get_state(), reward, done, info

    def _generate_test_input(self, category: str, intensity: float, params: np.ndarray) -> np.ndarray:
        """Generate test input with progressive difficulty"""
        input_dim = self.config['target_system']['input_dim']

        # Apply difficulty scaling
        intensity_scaled = intensity * self.current_difficulty

        if category == 'adversarial':
            base = np.array([0.5, 0.5] + [0.0] * (input_dim - 2))
            noise = np.random.randn(input_dim) * intensity_scaled * 0.1
            test_input = base + noise
        elif category == 'edge_case':
            test_input = np.random.randn(input_dim) * (5.0 + intensity_scaled * 5.0)
        elif category == 'boundary':
            target_sum = 10.0 + params[0] * 0.5
            test_input = np.random.randn(input_dim)
            test_input = test_input / np.sum(test_input) * target_sum
        elif category == 'distribution_shift':
            test_input = np.random.randn(input_dim) - (2.0 + intensity_scaled)
        elif category == 'logic_error':
            test_input = np.random.randn(input_dim)
            test_input[0] = -5.0 - params[0]
            test_input[1] = 1.0 + params[1]
        elif category == 'coverage_guided':
            # Use custom tool suggestions if available
            if self.custom_tools.coverage_analyzer:
                uncovered = self.custom_tools.coverage_analyzer.get_uncovered_regions()
                if uncovered:
                    test_input = uncovered[0]
                else:
                    test_input = self._coverage_guided_generation(params, input_dim)
            else:
                test_input = self._coverage_guided_generation(params, input_dim)
        elif category == 'metamorphic':
            # Use mutation engine if available
            if self.custom_tools.mutation_engine and len(self.custom_tools.mutation_engine.population) > 0:
                offspring = self.custom_tools.mutation_engine.generate_offspring(1)
                test_input = offspring[0] if offspring else np.random.randn(input_dim)
            else:
                test_input = np.random.randn(input_dim)
        else:
            test_input = np.random.randn(input_dim) * (1.0 + intensity_scaled)

        return test_input

    def _coverage_guided_generation(self, params: np.ndarray, input_dim: int) -> np.ndarray:
        """Generate coverage-guided test input"""
        low_coverage_dims = np.where(self.coverage_map < 0.5)[0]
        test_input = np.random.randn(input_dim)
        if len(low_coverage_dims) > 0:
            idx = np.random.choice(low_coverage_dims)
            test_input[idx % input_dim] = params[0]
        return test_input

    def _execute_test(self, test_input: np.ndarray, category: str) -> Dict:
        """Execute test on target system"""
        prediction, has_bug, bug_info = self.target_system.predict(test_input)

        input_region = int(np.clip(np.mean(test_input) + 5, 0, 9))
        self.input_space_coverage[input_region] = 1.0

        # Cost increases with difficulty
        cost = 1.0 * self.current_difficulty

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

    def _calculate_enhanced_reward(self, test_result: Dict, fallback_type: str = None) -> float:
        """Calculate reward with diversity and collaboration bonuses"""
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

        # Diversity bonus
        diversity_score = self._calculate_diversity()
        reward += diversity_score * rewards_config.get('diversity_bonus', 3)

        # Fallback penalty (slight penalty for needing fallback)
        if fallback_type:
            reward -= 2.0

        # Custom tool bonus
        if test_result.get('used_custom_tool'):
            reward += 5.0

        return reward

    def _calculate_diversity(self) -> float:
        """Calculate diversity of testing strategy"""
        if np.sum(self.category_usage) == 0:
            return 0.0

        # Entropy-based diversity
        probs = self.category_usage / np.sum(self.category_usage)
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log(probs))

        # Normalize (max entropy for uniform distribution)
        max_entropy = np.log(self.n_categories)
        diversity = entropy / max_entropy if max_entropy > 0 else 0

        return diversity

    def _severity_to_fitness(self, severity) -> float:
        """Convert bug severity to fitness score"""
        fitness_map = {
            BugSeverity.CRITICAL: 100,
            BugSeverity.HIGH: 50,
            BugSeverity.MEDIUM: 20,
            BugSeverity.LOW: 5,
            BugSeverity.NONE: 0
        }
        return fitness_map.get(severity, 0)

    def _get_state(self) -> np.ndarray:
        """Get enhanced state representation"""
        base_state = np.concatenate([
            self.coverage_map,
            self.bug_history,
            np.array([
                self.remaining_budget / self.test_budget,
                self.time_elapsed / self.max_steps,
                self.bugs_found / (20.0 * self.current_difficulty)
            ]),
            self.confidence_scores
        ])

        # Add difficulty and tool usage indicators
        tool_indicators = np.array([
            self.current_difficulty / 5.0,
            1.0 if self.custom_tools.adversarial_generator else 0.0,
            1.0 if self.custom_tools.mutation_engine else 0.0,
            1.0 if self.custom_tools.coverage_analyzer else 0.0,
            self._calculate_diversity()
        ])

        state = np.concatenate([base_state, tool_indicators])
        return state.astype(np.float32)

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Difficulty: {self.current_difficulty:.2f}")
            print(f"Tests Conducted: {self.tests_conducted}")
            print(f"Bugs Found: {self.bugs_found}")
            print(f"Coverage: {np.mean(self.coverage_map):.2%}")
            print(f"Diversity: {self._calculate_diversity():.2%}")
            print(f"Remaining Budget: {self.remaining_budget:.1f}")

            if self.custom_tools.coverage_analyzer:
                metrics = self.custom_tools.coverage_analyzer.get_coverage_metrics()
                print(f"Advanced Coverage: {metrics['overall_coverage']:.2%}")

            print(f"{'='*60}\n")

    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        base_metrics = {
            'bugs_found': self.bugs_found,
            'tests_conducted': self.tests_conducted,
            'bug_discovery_rate': self.bugs_found / max(1, self.tests_conducted),
            'coverage': np.mean(self.coverage_map),
            'unique_bug_types': len(self.bug_types_found),
            'efficiency': self.bugs_found / max(1, self.time_elapsed),
            'diversity_score': self._calculate_diversity(),
            'difficulty': self.current_difficulty
        }

        # Add custom tool metrics
        if self.custom_tools:
            base_metrics['custom_tool_stats'] = self.custom_tools.get_all_statistics()

        # Add fallback metrics
        if self.fallback_manager:
            base_metrics['fallback_stats'] = self.fallback_manager.get_statistics()

        return base_metrics