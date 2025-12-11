"""
Quick demo script to showcase the validation system
"""

import yaml
import numpy as np
from src.environment import PopperValidationEnv
from src.agents.ucb_agent import UCBAgent
from src.baselines import RandomStrategy


def run_demo():
    """Run a quick demonstration of the validation system"""

    print("=" * 70)
    print("POPPER RL VALIDATION - LIVE DEMONSTRATION")
    print("=" * 70)

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Reduce episodes for demo
    n_episodes = 5

    # Create environment
    env = PopperValidationEnv(config)

    print("\n1. RANDOM BASELINE")
    print("-" * 70)

    # Run random baseline
    random_agent = RandomStrategy(config)
    random_bugs = []

    for ep in range(n_episodes):
        state = env.reset()
        episode_bugs = 0
        done = False

        while not done:
            action = random_agent.select_action(state)
            state, reward, done, info = env.step(action)
            if info.get('test_result', {}).get('has_bug'):
                episode_bugs += 1

        random_bugs.append(episode_bugs)
        print(f"  Episode {ep + 1}: {episode_bugs} bugs found")

    print(f"\n  Average: {np.mean(random_bugs):.1f} bugs per episode")

    print("\n2. UCB AGENT (Learning)")
    print("-" * 70)

    # Train UCB agent
    ucb_agent = UCBAgent(len(env.test_categories), config)
    ucb_bugs = []

    for ep in range(n_episodes):
        state = env.reset()
        episode_bugs = 0
        done = False

        while not done:
            action = ucb_agent.select_action(state)
            state, reward, done, info = env.step(action)

            ucb_agent.update(action, reward)

            if info.get('test_result', {}).get('has_bug'):
                episode_bugs += 1

        ucb_bugs.append(episode_bugs)
        stats = ucb_agent.get_stats()
        print(f"  Episode {ep + 1}: {episode_bugs} bugs found | Best category: {stats['best_category']}")

    print(f"\n  Average: {np.mean(ucb_bugs):.1f} bugs per episode")

    print("\n3. COMPARISON")
    print("-" * 70)
    improvement = (np.mean(ucb_bugs) - np.mean(random_bugs)) / np.mean(random_bugs) * 100
    print(f"  Random Baseline:  {np.mean(random_bugs):.1f} bugs")
    print(f"  UCB Agent:        {np.mean(ucb_bugs):.1f} bugs")
    print(f"  Improvement:      {improvement:+.1f}%")

    print("\n4. UCB LEARNING INSIGHTS")
    print("-" * 70)
    arm_stats = ucb_agent.get_arm_statistics()

    # Sort by average reward
    arm_stats_sorted = sorted(arm_stats, key=lambda x: x['avg_reward'], reverse=True)

    print("\n  Most Effective Test Categories:")
    for i, stat in enumerate(arm_stats_sorted[:5], 1):
        print(f"    {i}. {stat['category']:20s} - Avg Reward: {stat['avg_reward']:6.2f} ({stat['pulls']} tests)")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nTo run full training:")
    print("  python src/train.py")
    print("\nTo evaluate results:")
    print("  python src/evaluate.py")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()