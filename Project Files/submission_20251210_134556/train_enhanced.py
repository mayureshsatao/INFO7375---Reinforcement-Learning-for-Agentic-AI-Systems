"""
Enhanced Training Script with All Improvements
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
import json

from environment_enhanced import EnhancedPopperValidationEnv
from agents.dqn_agent import DQNAgent
from agents.ucb_agent import UCBAgent
from multi_agent import MultiAgentCoordinator
from baselines import create_baseline
from generate_report import TechnicalReportGenerator


class EnhancedTrainer:
    """Enhanced trainer with all improvements integrated"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        seed = self.config['experiment']['seed']
        np.random.seed(seed)

        self.output_dir = self.config['experiment']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.env = EnhancedPopperValidationEnv(self.config)

        self.metrics_history = {
            'episode_rewards': [],
            'episode_bugs_found': [],
            'episode_coverage': [],
            'episode_lengths': [],
            'episode_diversity': [],
            'custom_tool_usage': [],
            'fallback_activations': []
        }

    def train_dqn_enhanced(self, n_episodes: int = None):
        """Train DQN with all enhancements"""
        if n_episodes is None:
            n_episodes = self.config['environment']['max_episodes']

        print("Training Enhanced DQN Agent...")
        print("Features: Custom Tools, Fallback Strategies, Progressive Difficulty")

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        agent = DQNAgent(state_dim, action_dim, self.config)

        for episode in tqdm(range(n_episodes), desc="Training DQN"):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)

                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()

                episode_reward += reward
                episode_length += 1
                state = next_state

            # Collect enhanced metrics
            metrics = self.env.get_metrics()
            self.metrics_history['episode_rewards'].append(episode_reward)
            self.metrics_history['episode_bugs_found'].append(metrics['bugs_found'])
            self.metrics_history['episode_coverage'].append(metrics['coverage'])
            self.metrics_history['episode_lengths'].append(episode_length)
            self.metrics_history['episode_diversity'].append(metrics.get('diversity_score', 0))

            # Log progress
            if episode % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpisode {episode}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Bugs Found: {metrics['bugs_found']}")
                print(f"  Coverage: {metrics['coverage']:.2%}")
                print(f"  Diversity: {metrics.get('diversity_score', 0):.2f}")
                print(f"  Difficulty: {metrics.get('difficulty', 1.0):.2f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")

                # Custom tool stats
                tool_stats = metrics.get('custom_tool_stats', {})
                if tool_stats:
                    print(f"  Custom Tools: {len(tool_stats)} active")

            if episode % self.config['experiment']['save_interval'] == 0:
                agent.save(os.path.join(self.output_dir, f"dqn_enhanced_checkpoint_{episode}.pt"))

        agent.save(os.path.join(self.output_dir, "dqn_enhanced_final.pt"))
        self._save_enhanced_metrics("dqn_enhanced")

        return agent

    def train_ucb_enhanced(self, n_episodes: int = None):
        """Train UCB with enhancements"""
        if n_episodes is None:
            n_episodes = self.config['environment']['max_episodes']

        print("Training Enhanced UCB Agent...")
        print("Features: Diversity Bonus, Fallback Strategies, Progressive Difficulty")

        n_arms = len(self.env.test_categories)
        agent = UCBAgent(n_arms, self.config)

        for episode in tqdm(range(n_episodes), desc="Training UCB"):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                agent.update(action, reward)

                episode_reward += reward
                episode_length += 1
                state = next_state

            metrics = self.env.get_metrics()
            self.metrics_history['episode_rewards'].append(episode_reward)
            self.metrics_history['episode_bugs_found'].append(metrics['bugs_found'])
            self.metrics_history['episode_coverage'].append(metrics['coverage'])
            self.metrics_history['episode_lengths'].append(episode_length)
            self.metrics_history['episode_diversity'].append(metrics.get('diversity_score', 0))

            if episode % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpisode {episode}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Bugs Found: {metrics['bugs_found']}")
                print(f"  Coverage: {metrics['coverage']:.2%}")
                print(f"  Diversity: {metrics.get('diversity_score', 0):.2f}")
                print(f"  Difficulty: {metrics.get('difficulty', 1.0):.2f}")
                ucb_stats = agent.get_stats()
                print(f"  Best Category: {ucb_stats['best_category']}")
                print(f"  Exploration Rate: {ucb_stats['exploration_rate']:.2%}")

        agent.save(os.path.join(self.output_dir, "ucb_enhanced_final.json"))
        self._save_enhanced_metrics("ucb_enhanced")

        arm_stats = agent.get_arm_statistics()
        with open(os.path.join(self.output_dir, "ucb_enhanced_arm_stats.json"), 'w') as f:
            json.dump(arm_stats, f, indent=2)

        return agent

    def train_multi_agent(self, n_episodes: int = 200):
        """Train multi-agent collaborative system"""
        if not self.config.get('multi_agent', {}).get('enabled', False):
            print("Multi-agent system not enabled in config. Skipping.")
            return None

        print("Training Multi-Agent System...")
        print("Features: Specialist Agents, Knowledge Sharing, Coordination")

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        coordinator = MultiAgentCoordinator(state_dim, action_dim, self.config)

        for episode in tqdm(range(n_episodes), desc="Training Multi-Agent"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                action, active_idx = coordinator.coordinate_action(state)
                next_state, reward, done, info = self.env.step(action)

                coordinator.update_agents(active_idx, action, reward, info, state)

                episode_reward += reward
                state = next_state

            if episode % 10 == 0:
                team_stats = coordinator.get_team_statistics()
                print(f"\nEpisode {episode}")
                print(f"  Team Bugs Found: {team_stats['total_bugs_found']}")
                print(f"  Team Success Rate: {team_stats['team_success_rate']:.4f}")
                print(f"  Collaborations: {team_stats['collaborations']}")
                print(f"  Knowledge Base: {team_stats['knowledge_base_size']} entries")

        # Save team statistics
        team_stats = coordinator.get_team_statistics()
        with open(os.path.join(self.output_dir, "multi_agent_stats.json"), 'w') as f:
            json.dump(team_stats, f, indent=2)

        return coordinator

    def compare_all_enhanced(self):
        """Compare all methods including enhancements"""
        print("=" * 70)
        print("COMPREHENSIVE ENHANCED EVALUATION")
        print("=" * 70)

        results = {}

        # Train enhanced DQN
        print("\n1. Training Enhanced DQN...")
        self.metrics_history = {k: [] for k in self.metrics_history.keys()}
        dqn_agent = self.train_dqn_enhanced(n_episodes=200)
        results['dqn_enhanced'] = self._evaluate_trained_agent(dqn_agent, 'DQN_Enhanced', n_episodes=100)

        # Train enhanced UCB
        print("\n2. Training Enhanced UCB...")
        self.metrics_history = {k: [] for k in self.metrics_history.keys()}
        ucb_agent = self.train_ucb_enhanced(n_episodes=200)
        results['ucb_enhanced'] = self._evaluate_trained_agent(ucb_agent, 'UCB_Enhanced', n_episodes=100)

        # Train multi-agent if enabled
        if self.config.get('multi_agent', {}).get('enabled', False):
            print("\n3. Training Multi-Agent System...")
            multi_agent = self.train_multi_agent(n_episodes=200)
            if multi_agent:
                results['multi_agent'] = self._evaluate_multi_agent(multi_agent, n_episodes=100)

        # Evaluate baselines
        baselines = ['random', 'coverage_guided', 'metamorphic', 'adaptive_random']
        for baseline_name in baselines:
            print(f"\n4. Evaluating {baseline_name}...")
            results[baseline_name] = self.evaluate_baseline(baseline_name, n_episodes=100)

        # Save results
        with open(os.path.join(self.output_dir, "enhanced_comparison_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        # Generate plots
        self._plot_enhanced_comparison(results)

        # Generate technical report
        print("\n5. Generating Technical Report...")
        report_gen = TechnicalReportGenerator(self.output_dir)
        report_gen.generate_full_report()

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)

        return results

    def evaluate_baseline(self, strategy_name: str, n_episodes: int = 100):
        """Evaluate baseline with enhanced environment"""
        baseline = create_baseline(strategy_name, self.config)
        metrics_list = []

        for episode in tqdm(range(n_episodes), desc=f"Evaluating {strategy_name}"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                action = baseline.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                if hasattr(baseline, 'update'):
                    if strategy_name == 'adaptive_random':
                        baseline.update(action, reward)
                    elif strategy_name == 'metamorphic':
                        test_result = info.get('test_result', {})
                        test_input = test_result.get('test_input', None)
                        baseline.update(test_input, test_result)

                episode_reward += reward
                state = next_state

            metrics = self.env.get_metrics()
            metrics['episode_reward'] = episode_reward
            metrics_list.append(metrics)

        return {
            'strategy': strategy_name,
            'avg_reward': np.mean([m['episode_reward'] for m in metrics_list]),
            'avg_bugs_found': np.mean([m['bugs_found'] for m in metrics_list]),
            'avg_coverage': np.mean([m['coverage'] for m in metrics_list]),
            'avg_bug_discovery_rate': np.mean([m['bug_discovery_rate'] for m in metrics_list]),
            'avg_diversity': np.mean([m.get('diversity_score', 0) for m in metrics_list]),
            'std_bugs_found': np.std([m['bugs_found'] for m in metrics_list])
        }

    def _evaluate_trained_agent(self, agent, agent_name: str, n_episodes: int = 100):
        """Evaluate trained agent"""
        metrics_list = []

        for episode in tqdm(range(n_episodes), desc=f"Evaluating {agent_name}"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state)

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state

            metrics = self.env.get_metrics()
            metrics['episode_reward'] = episode_reward
            metrics_list.append(metrics)

        return {
            'strategy': agent_name,
            'avg_reward': np.mean([m['episode_reward'] for m in metrics_list]),
            'avg_bugs_found': np.mean([m['bugs_found'] for m in metrics_list]),
            'avg_coverage': np.mean([m['coverage'] for m in metrics_list]),
            'avg_bug_discovery_rate': np.mean([m['bug_discovery_rate'] for m in metrics_list]),
            'avg_diversity': np.mean([m.get('diversity_score', 0) for m in metrics_list]),
            'std_bugs_found': np.std([m['bugs_found'] for m in metrics_list])
        }

    def _evaluate_multi_agent(self, coordinator, n_episodes: int = 100):
        """Evaluate multi-agent system"""
        metrics_list = []

        for episode in tqdm(range(n_episodes), desc="Evaluating Multi-Agent"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                action, active_idx = coordinator.coordinate_action(state)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

            metrics = self.env.get_metrics()
            metrics['episode_reward'] = episode_reward
            metrics_list.append(metrics)

        team_stats = coordinator.get_team_statistics()

        return {
            'strategy': 'Multi-Agent',
            'avg_reward': np.mean([m['episode_reward'] for m in metrics_list]),
            'avg_bugs_found': np.mean([m['bugs_found'] for m in metrics_list]),
            'avg_coverage': np.mean([m['coverage'] for m in metrics_list]),
            'avg_bug_discovery_rate': np.mean([m['bug_discovery_rate'] for m in metrics_list]),
            'avg_diversity': np.mean([m.get('diversity_score', 0) for m in metrics_list]),
            'std_bugs_found': np.std([m['bugs_found'] for m in metrics_list]),
            'team_stats': team_stats
        }

    def _save_enhanced_metrics(self, agent_name: str):
        """Save enhanced training metrics"""
        with open(os.path.join(self.output_dir, f"{agent_name}_metrics.json"), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self._plot_enhanced_training_curves(agent_name)

    def _plot_enhanced_training_curves(self, agent_name: str):
        """Plot enhanced training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].plot(self.metrics_history['episode_rewards'])
        axes[0, 0].set_title(f'{agent_name.upper()} - Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.metrics_history['episode_bugs_found'])
        axes[0, 1].set_title(f'{agent_name.upper()} - Bugs Found per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Bugs Found')
        axes[0, 1].grid(True)

        axes[0, 2].plot(self.metrics_history['episode_coverage'])
        axes[0, 2].set_title(f'{agent_name.upper()} - Coverage per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Coverage')
        axes[0, 2].grid(True)

        axes[1, 0].plot(self.metrics_history['episode_lengths'])
        axes[1, 0].set_title(f'{agent_name.upper()} - Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.metrics_history['episode_diversity'])
        axes[1, 1].set_title(f'{agent_name.upper()} - Diversity Score')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Diversity')
        axes[1, 1].grid(True)

        # Smoothed bugs found
        if len(self.metrics_history['episode_bugs_found']) > 10:
            smoothed = np.convolve(self.metrics_history['episode_bugs_found'],
                                   np.ones(10) / 10, mode='valid')
            axes[1, 2].plot(smoothed)
            axes[1, 2].set_title(f'{agent_name.upper()} - Smoothed Bugs Found')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Bugs (10-ep MA)')
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{agent_name}_training_curves.png'), dpi=300)
        plt.close()

    def _plot_enhanced_comparison(self, results: Dict):
        """Plot enhanced comparison"""
        methods = list(results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Bugs found
        bugs = [results[m]['avg_bugs_found'] for m in methods]
        axes[0, 0].bar(methods, bugs)
        axes[0, 0].set_title('Average Bugs Found', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Bugs')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, axis='y')

        # Coverage
        coverage = [results[m]['avg_coverage'] * 100 for m in methods]
        axes[0, 1].bar(methods, coverage)
        axes[0, 1].set_title('Average Coverage', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Coverage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, axis='y')

        # Bug discovery rate
        rates = [results[m]['avg_bug_discovery_rate'] for m in methods]
        axes[1, 0].bar(methods, rates)
        axes[1, 0].set_title('Bug Discovery Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Bugs per Test')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, axis='y')

        # Diversity
        diversity = [results[m].get('avg_diversity', 0) for m in methods]
        axes[1, 1].bar(methods, diversity)
        axes[1, 1].set_title('Testing Diversity', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Diversity Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'enhanced_comparison.png'), dpi=300)
        plt.close()


def main():
    """Main enhanced training function"""
    trainer = EnhancedTrainer("config.yaml")
    results = trainer.compare_all_enhanced()

    print("\n" + "=" * 70)
    print("FINAL ENHANCED RESULTS SUMMARY")
    print("=" * 70)
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Avg Bugs Found: {metrics['avg_bugs_found']:.2f}")
        print(f"  Avg Coverage: {metrics['avg_coverage']:.2%}")
        print(f"  Bug Discovery Rate: {metrics['avg_bug_discovery_rate']:.4f}")
        print(f"  Diversity: {metrics.get('avg_diversity', 0):.4f}")


if __name__ == "__main__":
    main()