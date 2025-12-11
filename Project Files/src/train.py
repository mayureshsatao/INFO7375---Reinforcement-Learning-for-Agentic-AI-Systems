"""
Training script for Popper RL Validation agents - COMPLETE FIX
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
import json

from environment import PopperValidationEnv
from agents.dqn_agent import DQNAgent
from agents.ucb_agent import UCBAgent
from baselines import create_baseline


class Trainer:
    """Trainer for RL validation agents"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        seed = self.config['experiment']['seed']
        np.random.seed(seed)

        self.output_dir = self.config['experiment']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.env = PopperValidationEnv(self.config)

        self.metrics_history = {
            'episode_rewards': [],
            'episode_bugs_found': [],
            'episode_coverage': [],
            'episode_lengths': [],
            'losses': []
        }

    def train_dqn(self, n_episodes: int = None):
        """Train DQN agent"""
        if n_episodes is None:
            n_episodes = self.config['environment']['max_episodes']

        print("Training DQN Agent...")

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

            metrics = self.env.get_metrics()
            self.metrics_history['episode_rewards'].append(episode_reward)
            self.metrics_history['episode_bugs_found'].append(metrics['bugs_found'])
            self.metrics_history['episode_coverage'].append(metrics['coverage'])
            self.metrics_history['episode_lengths'].append(episode_length)

            if episode % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpisode {episode}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Bugs Found: {metrics['bugs_found']}")
                print(f"  Coverage: {metrics['coverage']:.2%}")
                print(f"  Epsilon: {agent.epsilon:.3f}")

            if episode % self.config['experiment']['save_interval'] == 0:
                agent.save(os.path.join(self.output_dir, f"dqn_checkpoint_{episode}.pt"))

        agent.save(os.path.join(self.output_dir, "dqn_final.pt"))
        self._save_metrics("dqn")

        return agent

    def train_ucb(self, n_episodes: int = None):
        """Train UCB agent"""
        if n_episodes is None:
            n_episodes = self.config['environment']['max_episodes']

        print("Training UCB Agent...")

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

            if episode % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpisode {episode}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Bugs Found: {metrics['bugs_found']}")
                print(f"  Coverage: {metrics['coverage']:.2%}")
                ucb_stats = agent.get_stats()
                print(f"  Best Category: {ucb_stats['best_category']}")

        agent.save(os.path.join(self.output_dir, "ucb_final.json"))
        self._save_metrics("ucb")

        arm_stats = agent.get_arm_statistics()
        with open(os.path.join(self.output_dir, "ucb_arm_stats.json"), 'w') as f:
            json.dump(arm_stats, f, indent=2)

        return agent

    def evaluate_baseline(self, strategy_name: str, n_episodes: int = 100):
        """Evaluate baseline strategy"""
        print(f"Evaluating {strategy_name} baseline...")

        baseline = create_baseline(strategy_name, self.config)
        metrics_list = []

        for episode in tqdm(range(n_episodes), desc=f"Evaluating {strategy_name}"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                action = baseline.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # FIXED: Update baselines with proper data
                if hasattr(baseline, 'update'):
                    if strategy_name == 'adaptive_random':
                        baseline.update(action, reward)
                    elif strategy_name == 'metamorphic':
                        # Pass the test_input from test_result
                        test_result = info.get('test_result', {})
                        test_input = test_result.get('test_input', None)
                        baseline.update(test_input, test_result)

                episode_reward += reward
                state = next_state

            metrics = self.env.get_metrics()
            metrics['episode_reward'] = episode_reward
            metrics_list.append(metrics)

        avg_metrics = {
            'strategy': strategy_name,
            'avg_reward': np.mean([m['episode_reward'] for m in metrics_list]),
            'avg_bugs_found': np.mean([m['bugs_found'] for m in metrics_list]),
            'avg_coverage': np.mean([m['coverage'] for m in metrics_list]),
            'avg_bug_discovery_rate': np.mean([m['bug_discovery_rate'] for m in metrics_list]),
            'std_bugs_found': np.std([m['bugs_found'] for m in metrics_list])
        }

        with open(os.path.join(self.output_dir, f"baseline_{strategy_name}.json"), 'w') as f:
            json.dump(avg_metrics, f, indent=2)

        return avg_metrics

    def compare_all_methods(self):
        """Train and compare all methods"""
        print("=" * 60)
        print("Comprehensive Evaluation: RL vs Baselines")
        print("=" * 60)

        results = {}

        print("\n1. Training DQN...")
        self.metrics_history = {'episode_rewards': [], 'episode_bugs_found': [],
                               'episode_coverage': [], 'episode_lengths': []}
        dqn_agent = self.train_dqn(n_episodes=200)
        results['dqn'] = self._evaluate_trained_agent(dqn_agent, 'DQN', n_episodes=100)

        print("\n2. Training UCB...")
        self.metrics_history = {'episode_rewards': [], 'episode_bugs_found': [],
                               'episode_coverage': [], 'episode_lengths': []}
        ucb_agent = self.train_ucb(n_episodes=200)
        results['ucb'] = self._evaluate_trained_agent(ucb_agent, 'UCB', n_episodes=100)

        baselines = ['random', 'coverage_guided', 'metamorphic', 'adaptive_random']
        for baseline_name in baselines:
            print(f"\n3. Evaluating {baseline_name}...")
            results[baseline_name] = self.evaluate_baseline(baseline_name, n_episodes=100)

        with open(os.path.join(self.output_dir, "comparison_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_comparison(results)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        return results

    def _evaluate_trained_agent(self, agent, agent_name: str, n_episodes: int = 100):
        """Evaluate a trained agent"""
        metrics_list = []

        for episode in tqdm(range(n_episodes), desc=f"Evaluating {agent_name}"):
            state = self.env.reset()
            episode_reward = 0

            done = False
            while not done:
                # FIXED: Check agent type and call select_action appropriately
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=False)
                else:  # UCBAgent
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
            'std_bugs_found': np.std([m['bugs_found'] for m in metrics_list])
        }

    def _save_metrics(self, agent_name: str):
        """Save training metrics"""
        with open(os.path.join(self.output_dir, f"{agent_name}_metrics.json"), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self._plot_training_curves(agent_name)

    def _plot_training_curves(self, agent_name: str):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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

        axes[1, 0].plot(self.metrics_history['episode_coverage'])
        axes[1, 0].set_title(f'{agent_name.upper()} - Coverage per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Coverage')
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.metrics_history['episode_lengths'])
        axes[1, 1].set_title(f'{agent_name.upper()} - Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{agent_name}_training_curves.png'), dpi=300)
        plt.close()

    def _plot_comparison(self, results: Dict):
        """Plot comparison between all methods"""
        methods = list(results.keys())
        bugs_found = [results[m]['avg_bugs_found'] for m in methods]
        coverage = [results[m]['avg_coverage'] for m in methods]
        bug_rates = [results[m]['avg_bug_discovery_rate'] for m in methods]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(methods, bugs_found)
        axes[0].set_title('Average Bugs Found')
        axes[0].set_ylabel('Bugs')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, axis='y')

        axes[1].bar(methods, coverage)
        axes[1].set_title('Average Coverage')
        axes[1].set_ylabel('Coverage')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, axis='y')

        axes[2].bar(methods, bug_rates)
        axes[2].set_title('Bug Discovery Rate')
        axes[2].set_ylabel('Bugs per Test')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison.png'), dpi=300)
        plt.close()


def main():
    """Main training function"""
    trainer = Trainer("config.yaml")
    results = trainer.compare_all_methods()

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for method, metrics in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Avg Bugs Found: {metrics['avg_bugs_found']:.2f}")
        print(f"  Avg Coverage: {metrics['avg_coverage']:.2%}")
        print(f"  Bug Discovery Rate: {metrics['avg_bug_discovery_rate']:.4f}")


if __name__ == "__main__":
    main()