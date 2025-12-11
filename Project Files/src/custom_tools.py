"""
Custom Tools for Enhanced Popper Validation
Provides advanced testing capabilities beyond baseline methods
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from collections import deque


class AdversarialTestGenerator:
    """
    CUSTOM TOOL: Generates adversarial test cases using gradient-based methods
    guided by RL agent decisions. This tool helps discover edge cases that
    traditional random testing would miss.
    """

    def __init__(self, target_model, config: Dict):
        self.target_model = target_model
        self.config = config
        self.perturbation_budget = config.get('custom_tools', {}).get(
            'adversarial_generator', {}
        ).get('perturbation_budget', 0.1)
        self.gradient_steps = config.get('custom_tools', {}).get(
            'adversarial_generator', {}
        ).get('gradient_steps', 10)

        self.successful_attacks = []
        self.attack_history = []
        self.best_attacks = deque(maxlen=100)

    def generate_adversarial_test(self, base_input: np.ndarray,
                                  target_category: str = None) -> Tuple[np.ndarray, bool, Dict]:
        """
        Generate adversarial test case using iterative perturbation

        Args:
            base_input: Starting test input
            target_category: Optional specific vulnerability to target

        Returns:
            adversarial_input: Modified test input
            attack_success: Whether bug was triggered
            attack_info: Details about the attack
        """
        best_input = base_input.copy()
        best_bug_severity = None

        # Iterative adversarial generation
        for step in range(self.gradient_steps):
            # Create tensor with gradient tracking
            x = torch.FloatTensor(best_input).requires_grad_(True)

            # Forward pass
            output = self.target_model.forward(x.unsqueeze(0))

            # Compute gradient to maximize bug likelihood
            loss = -output.sum()  # Maximize uncertainty/instability

            # Compute gradients
            loss.backward()

            # Apply perturbation based on gradient
            with torch.no_grad():
                if x.grad is not None:
                    # Gradient-based perturbation
                    grad_sign = torch.sign(x.grad)
                    perturbation = self.perturbation_budget * grad_sign.numpy()
                else:
                    # Random perturbation as fallback
                    perturbation = np.random.randn(*best_input.shape) * self.perturbation_budget

                # Apply perturbation
                candidate_input = best_input + perturbation

            # Test current adversarial example
            _, has_bug, bug_info = self.target_model.predict(candidate_input)

            if has_bug:
                if best_bug_severity is None or \
                   self._severity_rank(bug_info['severity']) > self._severity_rank(best_bug_severity):
                    best_input = candidate_input.copy()
                    best_bug_severity = bug_info['severity']
            else:
                # Even if no bug, update best_input occasionally for exploration
                if step % 3 == 0:
                    best_input = candidate_input.copy()

        # Final evaluation
        prediction, has_bug, bug_info = self.target_model.predict(best_input)

        attack_info = {
            'method': 'gradient_based',
            'steps': self.gradient_steps,
            'perturbation_magnitude': np.linalg.norm(best_input - base_input),
            'success': has_bug
        }

        if has_bug:
            self.successful_attacks.append({
                'input': best_input,
                'bug_info': bug_info,
                'attack_info': attack_info
            })
            self.best_attacks.append(best_input)

        self.attack_history.append(attack_info)

        return best_input, has_bug, attack_info

    def _severity_rank(self, severity):
        """Convert severity to numeric rank"""
        from target_systems import BugSeverity
        ranks = {
            BugSeverity.CRITICAL: 4,
            BugSeverity.HIGH: 3,
            BugSeverity.MEDIUM: 2,
            BugSeverity.LOW: 1,
            BugSeverity.NONE: 0
        }
        return ranks.get(severity, 0)

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics on adversarial testing"""
        if not self.attack_history:
            return {'total_attempts': 0, 'success_rate': 0.0}

        successful = sum(1 for a in self.attack_history if a['success'])

        return {
            'total_attempts': len(self.attack_history),
            'successful_attacks': successful,
            'success_rate': successful / len(self.attack_history),
            'avg_perturbation': np.mean([a['perturbation_magnitude']
                                        for a in self.attack_history]),
            'unique_bugs_found': len(self.successful_attacks)
        }


class MutationEngine:
    """
    CUSTOM TOOL: Genetic algorithm-inspired test case mutation
    Evolves successful test cases to find similar bugs
    """

    def __init__(self, config: Dict):
        self.config = config
        self.mutation_rate = config.get('custom_tools', {}).get(
            'mutation_engine', {}
        ).get('mutation_rate', 0.3)
        self.crossover_rate = config.get('custom_tools', {}).get(
            'mutation_engine', {}
        ).get('crossover_rate', 0.5)

        self.population = []  # Successful test cases
        self.fitness_scores = []

    def add_to_population(self, test_case: np.ndarray, fitness: float):
        """Add successful test case to population"""
        self.population.append(test_case.copy())
        self.fitness_scores.append(fitness)

        # Keep population size manageable
        if len(self.population) > 100:
            # Remove lowest fitness
            min_idx = np.argmin(self.fitness_scores)
            self.population.pop(min_idx)
            self.fitness_scores.pop(min_idx)

    def mutate_test_case(self, test_case: np.ndarray) -> np.ndarray:
        """Apply mutation to test case"""
        mutated = test_case.copy()

        # Decide which dimensions to mutate
        mutation_mask = np.random.random(test_case.shape) < self.mutation_rate

        # Apply Gaussian mutation
        mutations = np.random.randn(*test_case.shape) * 0.5
        mutated[mutation_mask] += mutations[mutation_mask]

        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Combine two test cases"""
        child = parent1.copy()

        # Uniform crossover
        crossover_mask = np.random.random(parent1.shape) < self.crossover_rate
        child[crossover_mask] = parent2[crossover_mask]

        return child

    def generate_offspring(self, n: int = 1) -> List[np.ndarray]:
        """Generate new test cases from population"""
        if len(self.population) < 2:
            return []

        offspring = []

        for _ in range(n):
            # Select parents based on fitness (tournament selection)
            indices = np.random.choice(len(self.population), size=2,
                                      p=self._selection_probabilities())
            parent1 = self.population[indices[0]]
            parent2 = self.population[indices[1]]

            # Crossover
            child = self.crossover(parent1, parent2)

            # Mutation
            child = self.mutate_test_case(child)

            offspring.append(child)

        return offspring

    def _selection_probabilities(self) -> np.ndarray:
        """Calculate selection probabilities based on fitness"""
        if not self.fitness_scores:
            return np.array([])

        fitness_array = np.array(self.fitness_scores)
        # Add small epsilon to avoid division by zero
        fitness_array = fitness_array + 1e-10
        # Normalize to probabilities
        probs = fitness_array / fitness_array.sum()
        return probs

    def get_statistics(self) -> Dict:
        """Get mutation engine statistics"""
        return {
            'population_size': len(self.population),
            'avg_fitness': np.mean(self.fitness_scores) if self.fitness_scores else 0,
            'best_fitness': np.max(self.fitness_scores) if self.fitness_scores else 0,
            'genetic_diversity': np.std([tc.mean() for tc in self.population])
                                if self.population else 0
        }


class CoverageAnalyzer:
    """
    CUSTOM TOOL: Advanced coverage tracking and analysis
    Provides fine-grained insights into test coverage
    """

    def __init__(self, config: Dict, input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.granularity = config.get('custom_tools', {}).get(
            'coverage_analyzer', {}
        ).get('granularity', 'fine')

        # Multi-dimensional coverage tracking
        if self.granularity == 'fine':
            self.bins_per_dim = 20
        else:
            self.bins_per_dim = 10

        self.coverage_grid = {}
        self.dimension_coverage = [set() for _ in range(input_dim)]
        self.interaction_coverage = {}  # Track 2D interactions

        self.total_tests = 0
        self.coverage_history = []

    def record_test(self, test_input: np.ndarray):
        """Record test case for coverage analysis"""
        self.total_tests += 1

        # Record per-dimension coverage
        for dim_idx, value in enumerate(test_input):
            bin_idx = self._value_to_bin(value)
            self.dimension_coverage[dim_idx].add(bin_idx)

        # Record grid cell coverage
        grid_key = self._input_to_grid_key(test_input)
        self.coverage_grid[grid_key] = self.coverage_grid.get(grid_key, 0) + 1

        # Record 2D interactions (pairs of dimensions)
        for i in range(min(5, self.input_dim)):  # Limit to avoid explosion
            for j in range(i+1, min(5, self.input_dim)):
                interaction_key = (i, j,
                                 self._value_to_bin(test_input[i]),
                                 self._value_to_bin(test_input[j]))
                self.interaction_coverage[interaction_key] = True

        # Track coverage over time
        if self.total_tests % 10 == 0:
            self.coverage_history.append(self.get_coverage_metrics())

    def _value_to_bin(self, value: float) -> int:
        """Convert continuous value to discrete bin"""
        # Map to [-10, 10] range (based on action space)
        normalized = np.clip(value, -10, 10)
        bin_idx = int((normalized + 10) / 20 * self.bins_per_dim)
        return np.clip(bin_idx, 0, self.bins_per_dim - 1)

    def _input_to_grid_key(self, test_input: np.ndarray) -> tuple:
        """Convert input to grid cell key"""
        return tuple(self._value_to_bin(v) for v in test_input)

    def get_coverage_metrics(self) -> Dict:
        """Get comprehensive coverage metrics"""
        # Per-dimension coverage
        dim_coverage = [len(covered) / self.bins_per_dim
                       for covered in self.dimension_coverage]

        # Overall grid coverage
        total_possible_cells = self.bins_per_dim ** self.input_dim
        grid_coverage = len(self.coverage_grid) / total_possible_cells

        return {
            'overall_coverage': np.mean(dim_coverage) if dim_coverage else 0.0,
            'min_dimension_coverage': np.min(dim_coverage) if dim_coverage else 0.0,
            'max_dimension_coverage': np.max(dim_coverage) if dim_coverage else 0.0,
            'grid_coverage': grid_coverage,
            'unique_cells_covered': len(self.coverage_grid),
            'interaction_coverage': len(self.interaction_coverage),
            'coverage_balance': np.std(dim_coverage) if dim_coverage else 0.0
        }

    def get_uncovered_regions(self) -> List[np.ndarray]:
        """Identify regions with low coverage"""
        uncovered = []

        # Find dimensions with low coverage
        for dim_idx, covered_bins in enumerate(self.dimension_coverage):
            coverage_ratio = len(covered_bins) / self.bins_per_dim

            if coverage_ratio < 0.5:  # Less than 50% covered
                # Find uncovered bins
                all_bins = set(range(self.bins_per_dim))
                uncovered_bins = all_bins - covered_bins

                for bin_idx in list(uncovered_bins)[:5]:  # Suggest top 5
                    # Convert bin back to value
                    value = (bin_idx / self.bins_per_dim) * 20 - 10

                    # Create test input targeting this region
                    target_input = np.zeros(self.input_dim)
                    target_input[dim_idx] = value
                    uncovered.append(target_input)

        return uncovered[:10]  # Return top 10 suggestions

    def get_statistics(self) -> Dict:
        """Get coverage analyzer statistics"""
        metrics = self.get_coverage_metrics()
        metrics['total_tests'] = self.total_tests
        metrics['coverage_growth_rate'] = self._calculate_growth_rate()
        return metrics

    def _calculate_growth_rate(self) -> float:
        """Calculate rate of coverage improvement"""
        if len(self.coverage_history) < 2:
            return 0.0

        recent = self.coverage_history[-10:]
        if len(recent) < 2:
            return 0.0

        growth = recent[-1]['overall_coverage'] - recent[0]['overall_coverage']
        return growth / len(recent)


class CustomToolManager:
    """Manager for all custom tools"""

    def __init__(self, target_system, config: Dict):
        self.config = config
        self.target_system = target_system

        # Initialize tools
        self.adversarial_generator = None
        self.mutation_engine = None
        self.coverage_analyzer = None

        if config.get('custom_tools', {}).get('adversarial_generator', {}).get('enabled', False):
            self.adversarial_generator = AdversarialTestGenerator(target_system, config)

        if config.get('custom_tools', {}).get('mutation_engine', {}).get('enabled', False):
            self.mutation_engine = MutationEngine(config)

        if config.get('custom_tools', {}).get('coverage_analyzer', {}).get('enabled', False):
            input_dim = config['target_system']['input_dim']
            self.coverage_analyzer = CoverageAnalyzer(config, input_dim)

    def get_all_statistics(self) -> Dict:
        """Get statistics from all tools"""
        stats = {}

        if self.adversarial_generator:
            stats['adversarial'] = self.adversarial_generator.get_statistics()

        if self.mutation_engine:
            stats['mutation'] = self.mutation_engine.get_statistics()

        if self.coverage_analyzer:
            stats['coverage'] = self.coverage_analyzer.get_statistics()

        return stats