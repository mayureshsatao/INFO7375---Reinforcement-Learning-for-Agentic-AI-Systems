# Complete Setup Guide

This guide will walk you through setting up and running the Popper RL Validation project from scratch.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA for faster training

## Step-by-Step Setup

### 1. Download and Extract

If you received this as a ZIP file:
```bash
# Extract the ZIP file
unzip popper-rl-validation.zip
cd popper-rl-validation
```

If cloning from Git:
```bash
git clone <repository-url>
cd popper-rl-validation
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation time**: 2-5 minutes depending on internet speed.

**Verify installation:**
```bash
python -c "import torch; import gym; import numpy; print('All dependencies installed successfully!')"
```

### 4. Create Project Structure

The project should have this structure:
```
popper-rl-validation/
├── config.yaml
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md
├── demo.py
├── src/
│   ├── __init__.py
│   ├── target_systems.py
│   ├── environment.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baselines.py
│   └── agents/
│       ├── __init__.py
│       ├── dqn_agent.py
│       └── ucb_agent.py
└── experiments/
    └── results/  (will be created automatically)
```

Create the `__init__.py` files if missing:
```bash
touch src/__init__.py
touch src/agents/__init__.py
```

### 5. Quick Test (5 minutes)

Run the demo to verify everything works:
```bash
python demo.py
```

**Expected output:**
```
======================================================================
POPPER RL VALIDATION - LIVE DEMONSTRATION
======================================================================

1. RANDOM BASELINE
----------------------------------------------------------------------
  Episode 1: 3 bugs found
  Episode 2: 2 bugs found
  ...

2. UCB AGENT (Learning)
----------------------------------------------------------------------
  Episode 1: 4 bugs found | Best category: adversarial
  ...

3. COMPARISON
----------------------------------------------------------------------
  Random Baseline:  2.6 bugs
  UCB Agent:        4.2 bugs
  Improvement:      +61.5%
...
```

### 6. Run Full Training (15-30 minutes)

```bash
python src/train.py
```

This will:
1. Train DQN agent (200 episodes)
2. Train UCB agent (200 episodes)
3. Evaluate all baselines
4. Generate visualizations
5. Save results to `experiments/results/`

**Progress indicators:**
- You'll see progress bars for each phase
- Training metrics printed every 10 episodes
- Final results summary at the end

### 7. View Results

After training completes:

```bash
# Generate evaluation report
python src/evaluate.py

# View results
ls experiments/results/

# Key files to check:
# - comparison.png              (main results chart)
# - detailed_comparison.png     (comprehensive charts)
# - radar_comparison.png        (multi-dimensional comparison)
# - evaluation_report.txt       (text summary)
# - comparison_results.json     (numerical results)
```

## Troubleshooting

### Issue: Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or on Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
```

### Issue: CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python src/train.py
```

### Issue: Slow Training

**Solutions:**

1. **Reduce episodes** (edit `config.yaml`):
```yaml
environment:
  max_episodes: 100  # Instead of 1000
```

2. **Use GPU** (if available):
```python
# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

3. **Reduce batch size** (edit `config.yaml`):
```yaml
rl:
  batch_size: 32  # Instead of 64
```

### Issue: Missing Dependencies

**Problem:**
```
ImportError: cannot import name 'X'
```

**Solution:**
```bash
# Reinstall specific package
pip install --upgrade <package-name>

# Or reinstall all
pip install --force-reinstall -r requirements.txt
```

## Running Individual Components

### Train Only DQN

```python
from src.train import Trainer

trainer = Trainer("config.yaml")
dqn_agent = trainer.train_dqn(n_episodes=100)
```

### Train Only UCB

```python
from src.train import Trainer

trainer = Trainer("config.yaml")
ucb_agent = trainer.train_ucb(n_episodes=100)
```

### Run Single Baseline

```python
from src.train import Trainer

trainer = Trainer("config.yaml")
results = trainer.evaluate_baseline("random", n_episodes=50)
print(results)
```

### Custom Training Loop

```python
import yaml
from src.environment import PopperValidationEnv
from src.agents.ucb_agent import UCBAgent

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Create environment and agent
env = PopperValidationEnv(config)
agent = UCBAgent(10, config)

# Training loop
for episode in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(action, reward)
        total_reward += reward
        state = next_state
    
    print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Bugs = {info['bugs_found']}")
```

## Configuration Options

### Quick Configurations

**Fast Testing (2 minutes):**
```yaml
environment:
  max_episodes: 10
  max_steps_per_episode: 20
```

**Thorough Evaluation (1 hour):**
```yaml
environment:
  max_episodes: 1000
  max_steps_per_episode: 200
rl:
  total_timesteps: 100000
```

**GPU Optimized:**
```yaml
rl:
  batch_size: 128
  buffer_size: 50000
```

**CPU Optimized:**
```yaml
rl:
  batch_size: 32
  buffer_size: 5000
```

## Verification Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Demo runs successfully
- [ ] Training completes without errors
- [ ] Results saved to `experiments/results/`
- [ ] Visualizations generated
- [ ] Can open and view PNG files

## Next Steps

1. **Run full training**: `python src/train.py`
2. **Review results**: Check `experiments/results/` directory
3. **Customize**: Edit `config.yaml` for your experiments
4. **Extend**: Add custom bugs in `src/target_systems.py`
5. **Report**: Use generated data for your technical report

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review this troubleshooting guide
3. Check Python version: `python --version` (should be 3.8+)
4. Check dependencies: `pip list`
5. Try the demo first: `python demo.py`

## Hardware Requirements

**Minimum:**
- CPU: 2+ cores
- RAM: 4GB
- Storage: 1GB
- Time: 30 minutes (CPU training)

**Recommended:**
- CPU: 4+ cores or GPU
- RAM: 8GB
- Storage: 2GB
- Time: 10 minutes (GPU training)

## Expected Outputs

After successful training, you should have:

```
experiments/results/
├── dqn_final.pt                      (~5MB)
├── ucb_final.json                    (~100KB)
├── dqn_training_curves.png           
├── ucb_training_curves.png           
├── comparison.png                    
├── detailed_comparison.png           
├── radar_comparison.png              
├── comparison_results.json           
├── statistical_analysis.json         
└── evaluation_report.txt             
```

**Total size**: ~20-30MB

## Success Criteria

You've successfully set up the project if:

1. ✅ Demo runs without errors
2. ✅ Training completes successfully
3. ✅ All PNG visualizations are generated
4. ✅ JSON results files contain valid data
5. ✅ UCB agent shows >30% improvement over random baseline
6. ✅ DQN agent shows >50% improvement over random baseline

---

**You're all set! Start with the demo and work your way up to full training.** 🚀