# Quick Reference Guide

## Installation (2 minutes)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Basic Commands

### Run Demo (5 min)
```bash
python demo.py
```

### Full Training (30 min)
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

## File Organization

```
📁 Project Root
├── 📄 config.yaml          # Main configuration
├── 📄 requirements.txt     # Dependencies
├── 📄 demo.py             # Quick demo script
│
├── 📁 src/
│   ├── 📄 environment.py      # Gym environment
│   ├── 📄 target_systems.py   # Buggy AI systems
│   ├── 📄 train.py            # Training script
│   ├── 📄 evaluate.py         # Evaluation script
│   ├── 📄 baselines.py        # Baseline strategies
│   └── 📁 agents/
│       ├── 📄 dqn_agent.py    # DQN implementation
│       └── 📄 ucb_agent.py    # UCB implementation
│
└── 📁 experiments/results/  # Output (auto-created)
    ├── 🖼️ comparison.png
    ├── 🖼️ detailed_comparison.png
    ├── 📊 comparison_results.json
    └── 📝 evaluation_report.txt
```

## Key Configuration Parameters

### Edit `config.yaml`

**Training Speed:**
```yaml
environment:
  max_episodes: 200        # Number of training episodes
  max_steps_per_episode: 100  # Steps per episode
```

**RL Parameters:**
```yaml
rl:
  learning_rate: 0.0003    # DQN learning rate
  batch_size: 64           # Training batch size
  gamma: 0.99              # Discount factor
```

**Rewards:**
```yaml
rewards:
  bug_found:
    critical: 100          # Reward for critical bugs
    high: 50              # Reward for high severity
```

## Common Tasks

### Custom Training

```python
from src.train import Trainer

trainer = Trainer("config.yaml")

# Train specific agent
dqn = trainer.train_dqn(n_episodes=100)
ucb = trainer.train_ucb(n_episodes=100)

# Evaluate baseline
results = trainer.evaluate_baseline("random", n_episodes=50)
```

### Load Trained Model

```python
import yaml
from src.agents.dqn_agent import DQNAgent
from src.environment import PopperValidationEnv

with open("config.yaml") as f:
    config = yaml.safe_load(f)

env = PopperValidationEnv(config)
agent = DQNAgent(env.observation_space.shape[0], 
                 env.action_space.shape[0], 
                 config)

agent.load("experiments/results/dqn_final.pt")
```

### Run Single Episode

```python
import yaml
from src.environment import PopperValidationEnv
from src.agents.ucb_agent import UCBAgent

with open("config.yaml") as f:
    config = yaml.safe_load(f)

env = PopperValidationEnv(config)
agent = UCBAgent(10, config)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    agent.update(action, reward)
    total_reward += reward
    
print(f"Bugs found: {info['bugs_found']}")
print(f"Coverage: {info['coverage']:.2%}")
```

## Understanding Results

### Key Metrics

**Bug Discovery Rate**: Bugs found per test
- Good: > 0.10
- Excellent: > 0.15

**Coverage**: Proportion of test space covered
- Good: > 70%
- Excellent: > 85%

**Improvement**: RL vs Best Baseline
- Good: > 30%
- Excellent: > 50%

### Interpreting Visualizations

**training_curves.png**: Shows learning progress
- Episode Rewards: Should trend upward
- Bugs Found: Should increase over time
- Coverage: Should approach 80-90%

**comparison.png**: Bar charts comparing all methods
- Taller bars = better performance
- RL methods should outperform baselines

**radar_comparison.png**: Multi-dimensional view
- Larger area = better overall performance
- Look for balanced profiles

## Troubleshooting

### Quick Fixes

**Import Error:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA Error:**
```bash
export CUDA_VISIBLE_DEVICES=""
```

**Slow Training:**
```yaml
# Edit config.yaml
environment:
  max_episodes: 100  # Reduce
rl:
  batch_size: 32     # Reduce
```

## Performance Expectations

### Training Time
- **CPU**: 20-30 minutes
- **GPU**: 5-10 minutes

### Expected Results
- **DQN**: 12-15 bugs per episode
- **UCB**: 10-13 bugs per episode
- **Random**: 6-8 bugs per episode

### Resource Usage
- **RAM**: 2-4 GB
- **Disk**: 50 MB for results
- **CPU**: 50-100% during training

## For Your Report

### Key Files to Reference

1. **Architecture Diagram**: See README.md
2. **Learning Curves**: `*_training_curves.png`
3. **Performance Comparison**: `comparison.png`
4. **Statistical Results**: `comparison_results.json`
5. **Detailed Analysis**: `evaluation_report.txt`

### Metrics to Report

```json
{
  "avg_bugs_found": 14.2,
  "avg_coverage": 0.81,
  "bug_discovery_rate": 0.142,
  "improvement_vs_baseline": 58.2
}
```

### Mathematical Formulations

**UCB Formula:**
```
UCB(i) = X̄ᵢ + c√(2ln(n)/nᵢ)
```

**Reward Function:**
```
R = R_severity + R_novelty + R_coverage - R_cost + R_exploration
```

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

## Extension Ideas

### Add New Bug Type
```python
# In target_systems.py
'new_bug': {
    'trigger': lambda x: your_condition(x),
    'severity': BugSeverity.HIGH,
    'type': BugType.CUSTOM
}
```

### Add New Baseline
```python
# In baselines.py
class MyBaseline(BaselineStrategy):
    def select_action(self, state):
        # Your strategy
        return action
```

### Custom Reward
```python
# In environment.py, modify _calculate_reward()
def _calculate_reward(self, test_result):
    reward = 0.0
    # Add your custom rewards
    return reward
```

## Cheat Sheet

| Task | Command |
|------|---------|
| Setup | `pip install -r requirements.txt` |
| Demo | `python demo.py` |
| Train All | `python src/train.py` |
| Train DQN Only | `python -c "from src.train import Trainer; Trainer().train_dqn()"` |
| Train UCB Only | `python -c "from src.train import Trainer; Trainer().train_ucb()"` |
| Evaluate | `python src/evaluate.py` |
| View Results | `ls experiments/results/` |
| Check GPU | `python -c "import torch; print(torch.cuda.is_available())"` |

## Support

1. Check error messages
2. Review SETUP_GUIDE.md
3. Try demo.py first
4. Check Python version (3.8+)
5. Verify all dependencies installed

---

**Happy Validating! 🎯**