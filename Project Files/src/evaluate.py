import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
from scipy import stats

from environment import PopperValidationEnv
from agents.dqn_agent import DQNAgent
from agents.ucb_agent import UCBAgent


class Evaluator:
    """Comprehensive evaluation of trained agents"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config['experiment']['output_dir']
        self.env = PopperValidationEnv(self.config)

    def load_and_evaluate_dqn(self, model_path: str, n_episodes: int = 100):
        """Load and evaluate DQN agent"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        agent = DQNAgent(state_dim, action_dim, self.config)
        agent.load(model_path)

        return self._evaluate_agent(agent, "DQN", n_episodes)

    def load_and_evaluate_ucb(self, model_path: str, n_episodes: int = 100):
        """Load and evaluate UCB agent"""
        n_arms = len(self.env.test_categories)
        agent = UCBAgent(n_arms, self.config)
        agent.load(model_path)

        return self._evaluate_agent(agent, "UCB", n_episodes)

    def _evaluate_agent(self, agent, agent_name: str, n_episodes: int):
        """Evaluate agent and collect detailed metrics"""
        results = {
            'episodes': [],
            'bugs_by_severity': {'critical': [], 'high': [], 'medium': [], 'low': []},
            'bugs_by_type': {},
            'coverage_over_time': [],
            'discovery_times': [],
            'test_efficiency': []
        }

        for episode in range(n_episodes):
            state = self.env.reset()
            episode_data = {
                'bugs_found': [],
                'coverage_progression': [],
                'tests_conducted': 0
            }

            done = False
            step = 0
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)

                # Track bugs
                if info.get('test_result', {}).get('has_bug', False):
                    bug_info = info['test_result']['bug_info']
                    episode_data['bugs_found'].append({
                        'step': step,
                        'severity': bug_info['severity'].value,
                        'type': bug_info['type'].value if bug_info['type'] else None
                    })

                # Track coverage
                episode_data['coverage_progression'].append(info['coverage'])
                episode_data['tests_conducted'] += 1

                state = next_state
                step += 1

            results['episodes'].append(episode_data)

            # Aggregate episode results
            for bug in episode_data['bugs_found']:
                severity = bug['severity']
                if severity in results['bugs_by_severity']:
                    results['bugs_by_severity'][severity].append(1)

                bug_type = bug['type']
                if bug_type:
                    if bug_type not in results['bugs_by_type']:
                        results['bugs_by_type'][bug_type] = 0
                    results['bugs_by_type'][bug_type] += 1

        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)

        return results, summary

    def _calculate_summary_stats(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        total_bugs = sum(len(ep['bugs_found']) for ep in results['episodes'])
        total_tests = sum(ep['tests_conducted'] for ep in results['episodes'])

        bugs_per_episode = [len(ep['bugs_found']) for ep in results['episodes']]
        coverage_per_episode = [ep['coverage_progression'][-1] if ep['coverage_progression'] else 0
                                for ep in results['episodes']]

        summary = {
            'total_bugs_found': total_bugs,
            'total_tests_conducted': total_tests,
            'avg_bugs_per_episode': np.mean(bugs_per_episode),
            'std_bugs_per_episode': np.std(bugs_per_episode),
            'bug_discovery_rate': total_bugs / total_tests if total_tests > 0 else 0,
            'avg_coverage': np.mean(coverage_per_episode),
            'std_coverage': np.std(coverage_per_episode),
            'bugs_by_severity': {k: len(v) for k, v in results['bugs_by_severity'].items()},
            'bugs_by_type': results['bugs_by_type']
        }

        return summary

    def compare_methods(self, results_dir: str = None):
        """Load and compare all method results"""
        if results_dir is None:
            results_dir = self.output_dir

        comparison_file = os.path.join(results_dir, "comparison_results.json")

        if not os.path.exists(comparison_file):
            print("Comparison results not found. Run training first.")
            return

        with open(comparison_file, 'r') as f:
            results = json.load(f)

        # Create comparison visualizations
        self._create_comparison_visualizations(results)

        # Statistical tests
        self._perform_statistical_tests(results)

    def _create_comparison_visualizations(self, results: Dict):
        """Create comprehensive comparison visualizations"""
        # Set style
        sns.set_style("whitegrid")

        # 1. Performance comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        methods = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        # Bugs found
        bugs = [results[m]['avg_bugs_found'] for m in methods]
        axes[0, 0].bar(methods, bugs, color=colors)
        axes[0, 0].set_title('Average Bugs Found', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Bugs', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Coverage
        coverage = [results[m]['avg_coverage'] * 100 for m in methods]
        axes[0, 1].bar(methods, coverage, color=colors)
        axes[0, 1].set_title('Average Test Coverage', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Coverage (%)', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Bug discovery rate
        rates = [results[m]['avg_bug_discovery_rate'] for m in methods]
        axes[1, 0].bar(methods, rates, color=colors)
        axes[1, 0].set_title('Bug Discovery Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Bugs per Test', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Efficiency comparison (bugs found / avg reward as proxy)
        efficiency = [results[m]['avg_bugs_found'] / max(1, results[m].get('avg_reward', 1))
                      for m in methods]
        axes[1, 1].bar(methods, efficiency, color=colors)
        axes[1, 1].set_title('Testing Efficiency', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Efficiency Score', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detailed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Radar chart for multi-dimensional comparison
        self._create_radar_chart(results)

    def _create_radar_chart(self, results: Dict):
        """Create radar chart for multi-dimensional comparison"""
        categories = ['Bugs Found', 'Coverage', 'Discovery Rate', 'Consistency']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        methods = list(results.keys())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for method in methods:
            values = [
                results[method]['avg_bugs_found'] / 20.0,  # Normalize to 0-1
                results[method]['avg_coverage'],
                results[method]['avg_bug_discovery_rate'] * 10,  # Scale up
                1.0 - (results[method].get('std_bugs_found', 0) / 20.0)  # Consistency
            ]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _perform_statistical_tests(self, results: Dict):
        """Perform statistical significance tests"""
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)

        methods = list(results.keys())

        # Compare RL methods against best baseline
        rl_methods = [m for m in methods if m in ['dqn', 'ucb']]
        baseline_methods = [m for m in methods if m not in rl_methods]

        if not rl_methods or not baseline_methods:
            print("Need both RL and baseline results for comparison")
            return

        # Find best baseline
        best_baseline = max(baseline_methods,
                            key=lambda m: results[m]['avg_bugs_found'])

        print(f"\nBest Baseline: {best_baseline}")
        print(f"  Bugs Found: {results[best_baseline]['avg_bugs_found']:.2f}")

        for rl_method in rl_methods:
            improvement = (
                    (results[rl_method]['avg_bugs_found'] - results[best_baseline]['avg_bugs_found'])
                    / results[best_baseline]['avg_bugs_found'] * 100
            )

            print(f"\n{rl_method.upper()} vs {best_baseline}:")
            print(f"  Bugs Found: {results[rl_method]['avg_bugs_found']:.2f}")
            print(f"  Improvement: {improvement:+.1f}%")
            print(f"  Coverage: {results[rl_method]['avg_coverage']:.2%}")

        # Save statistical summary
        stats_summary = {
            'best_baseline': best_baseline,
            'rl_improvements': {
                rl: {
                    'bugs_improvement': (results[rl]['avg_bugs_found'] - results[best_baseline]['avg_bugs_found']),
                    'coverage_improvement': (results[rl]['avg_coverage'] - results[best_baseline]['avg_coverage'])
                }
                for rl in rl_methods
            }
        }

        with open(os.path.join(self.output_dir, 'statistical_analysis.json'), 'w') as f:
            json.dump(stats_summary, f, indent=2)

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\nGenerating evaluation report...")

        report_path = os.path.join(self.output_dir, "evaluation_report.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("POPPER RL VALIDATION - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Load comparison results
            comparison_file = os.path.join(self.output_dir, "comparison_results.json")
            if os.path.exists(comparison_file):
                with open(comparison_file, 'r') as cf:
                    results = json.load(cf)

                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 80 + "\n\n")

                for method, metrics in results.items():
                    f.write(f"{method.upper()}\n")
                    f.write(f"  Average Bugs Found: {metrics['avg_bugs_found']:.2f}\n")
                    f.write(f"  Average Coverage: {metrics['avg_coverage']:.2%}\n")
                    f.write(f"  Bug Discovery Rate: {metrics['avg_bug_discovery_rate']:.4f}\n")
                    f.write(f"  Std Bugs Found: {metrics.get('std_bugs_found', 0):.2f}\n")
                    f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated successfully!\n")

        print(f"Report saved to: {report_path}")


def main():
    """Main evaluation function"""
    evaluator = Evaluator()

    # Compare all methods
    evaluator.compare_methods()

    # Generate comprehensive report
    evaluator.generate_report()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()