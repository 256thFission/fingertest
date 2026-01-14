# Logging and Configuration Guide

This document explains the new Weights & Biases (wandb) logging and configuration features added to the authorship verification system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Management](#configuration-management)
3. [Weights & Biases Logging](#weights--biases-logging)
4. [Training Scripts](#training-scripts)
5. [Environment Variables](#environment-variables)

## Quick Start

### Installing wandb

```bash
pip install wandb
```

### Login to wandb

```bash
wandb login
```

### Running with wandb

All training scripts now support wandb by default:

```bash
# Baseline training with wandb
python train_baseline.py --epochs 3 --batch-size 64

# Triplet training with wandb
python train_triplet.py --epochs 2 --lr 1e-5

# Autonomous loop with wandb
python run_loop.py --iterations 5

# Evaluation with wandb
python evaluate.py --model models/baseline
```

### Disabling wandb

If you want to disable wandb logging:

```bash
# Using command line flag
python train_baseline.py --no-wandb

# Using environment variable
export WANDB_DISABLED=1
python train_baseline.py
```

## Configuration Management

### Using config.py

A new centralized configuration module (`config.py`) has been added for easy parameter management:

```python
from config import get_config, get_baseline_config, get_wandb_config

# Get full config
config = get_config()

# Get specific config sections
baseline_config = get_baseline_config()
wandb_config = get_wandb_config()

# Access parameters
print(f"Epochs: {baseline_config.num_epochs}")
print(f"Batch size: {baseline_config.batch_size}")
print(f"Wandb project: {wandb_config.project}")
```

### Modifying config.ini

Edit `config.ini` to change default parameters:

```ini
# Baseline Training
[baseline]
batch_size = 64
num_epochs = 3        # Change from 1 to 3
learning_rate = 2e-5
warmup_ratio = 0.1
fp16 = true

# Triplet Training
[triplet]
batch_size = 32
num_epochs = 2        # Change from 1 to 2
learning_rate = 1e-5
margin = 0.5
fp16 = true

# Autonomous Loop
[loop]
num_iterations = 5    # Change from 3 to 5

# Weights & Biases Logging
[wandb]
enabled = true
project = authorship-verification
entity = your-username-or-team
log_model = true
log_interval = 100
```

### Available Configuration Sections

- **data**: Data processing parameters (min_tokens, max_tokens, etc.)
- **model**: Model architecture (base_model, max_seq_length, etc.)
- **baseline**: Baseline training parameters
- **triplet**: Triplet training parameters
- **mining**: Hard negative mining parameters
- **loop**: Autonomous loop parameters
- **evaluation**: Evaluation parameters
- **visualization**: Visualization settings
- **hardware**: Hardware configuration
- **wandb**: Weights & Biases settings

## Weights & Biases Logging

### What Gets Logged

#### Baseline Training (train_baseline.py)
- Training configuration (batch size, learning rate, epochs, etc.)
- Dataset statistics (number of samples, authors, etc.)
- Validation metrics (cosine similarity)
- Model artifacts (optional)

#### Triplet Training (train_triplet.py)
- Training configuration
- Number of triplets
- Model artifacts (optional)

#### Autonomous Loop (run_loop.py)
- Iteration-level metrics:
  - Number of triplets mined per iteration
  - EER (Equal Error Rate) per iteration
  - ROC-AUC per iteration
- Summary table of all iterations
- Final model artifact

#### Evaluation (evaluate.py)
- Evaluation metrics:
  - EER (Equal Error Rate)
  - ROC-AUC
  - Accuracy at EER threshold
- Visualizations:
  - ROC curve
  - FAR/FRR curves
  - Score distribution
  - UMAP visualization

### Wandb Project Organization

By default, all runs are logged to the `authorship-verification` project. Each script adds specific tags:

- **train_baseline.py**: `["baseline", "phase2"]`
- **train_triplet.py**: `["triplet", "phase3b"]`
- **run_loop.py**: `["loop", "phase3"]`
- **evaluate.py**: `["evaluation", "phase4"]`

### Customizing Wandb Settings

#### Via Command Line

```bash
# Custom project name
python train_baseline.py --wandb-project my-project

# Custom entity (username/team)
python train_baseline.py --wandb-entity my-team

# Custom run name
python train_baseline.py --wandb-name experiment-001
```

#### Via config.ini

```ini
[wandb]
enabled = true
project = my-custom-project
entity = my-team
log_model = true
log_interval = 100
```

#### Via Environment Variables

```bash
export WANDB_PROJECT=my-project
export WANDB_ENTITY=my-team
python train_baseline.py
```

## Training Scripts

### train_baseline.py - Baseline Model Training

**New Parameters:**

```bash
python train_baseline.py \
  --epochs 3 \                    # Number of epochs (default: 1)
  --batch-size 64 \               # Batch size (default: auto)
  --lr 2e-5 \                     # Learning rate (default: 2e-5)
  --warmup-steps 1000 \           # Warmup steps (default: 1000)
  --fp16 \                        # Use FP16 (default: true)
  --wandb \                       # Enable wandb (default: true)
  --wandb-project my-project \    # Wandb project name
  --wandb-entity my-team          # Wandb entity
```

**Disable wandb:**

```bash
python train_baseline.py --no-wandb
```

### train_triplet.py - Triplet Fine-tuning

**New Parameters:**

```bash
python train_triplet.py \
  --epochs 2 \                    # Number of epochs (default: 1)
  --batch-size 32 \               # Batch size (default: 32)
  --lr 1e-5 \                     # Learning rate (default: 1e-5)
  --margin 0.5 \                  # Triplet margin (default: 0.5)
  --fp16 \                        # Use FP16 (default: true)
  --wandb \                       # Enable wandb (default: true)
  --wandb-project my-project      # Wandb project name
```

### run_loop.py - Autonomous Loop

**New Parameters:**

```bash
python run_loop.py \
  --iterations 5 \                # Number of loop iterations (default: 3)
  --sample-size 50000 \           # Mining sample size (default: 50000)
  --mining-k 10 \                 # K-nearest neighbors (default: 10)
  --batch-size 32 \               # Training batch size (default: 32)
  --lr 1e-5 \                     # Learning rate (default: 1e-5)
  --margin 0.5 \                  # Triplet margin (default: 0.5)
  --wandb \                       # Enable wandb (default: true)
  --wandb-project my-project      # Wandb project name
```

### evaluate.py - Model Evaluation

**New Parameters:**

```bash
python evaluate.py \
  --model models/baseline \       # Model path
  --num-positive 2000 \           # Positive test pairs (default: 2000)
  --num-negative 2000 \           # Negative test pairs (default: 2000)
  --wandb \                       # Enable wandb (default: true)
  --wandb-project my-project      # Wandb project name
```

## Environment Variables

The following environment variables can be used to control wandb:

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_DISABLED` | Disable wandb globally | `false` |
| `WANDB_PROJECT` | Override wandb project name | `authorship-verification` |
| `WANDB_ENTITY` | Override wandb entity | `None` |
| `WANDB_API_KEY` | Wandb API key | (set via `wandb login`) |

**Example:**

```bash
# Disable wandb for all scripts
export WANDB_DISABLED=1

# Set custom project and entity
export WANDB_PROJECT=my-project
export WANDB_ENTITY=my-team

# Run training
python train_baseline.py
```

## Example Workflows

### Training with Custom Parameters

```bash
# Train baseline model for 5 epochs with custom batch size
python train_baseline.py \
  --epochs 5 \
  --batch-size 128 \
  --lr 3e-5 \
  --warmup-steps 2000 \
  --wandb-name baseline-5epochs-bs128

# Fine-tune with triplets for 3 epochs
python train_triplet.py \
  --epochs 3 \
  --lr 5e-6 \
  --margin 0.3 \
  --wandb-name triplet-3epochs

# Run autonomous loop with 10 iterations
python run_loop.py \
  --iterations 10 \
  --sample-size 100000 \
  --wandb-name loop-10iter
```

### Using config.ini for Easy Experimentation

1. Edit `config.ini`:

```ini
[baseline]
num_epochs = 5
batch_size = 128
learning_rate = 3e-5

[triplet]
num_epochs = 3
learning_rate = 5e-6

[loop]
num_iterations = 10
```

2. Run scripts (they'll use config.ini defaults):

```bash
python train_baseline.py
python train_triplet.py
python run_loop.py
```

### Comparing Experiments in Wandb

```bash
# Experiment 1: Standard settings
python train_baseline.py --wandb-name exp1-standard

# Experiment 2: Higher learning rate
python train_baseline.py --lr 5e-5 --wandb-name exp2-high-lr

# Experiment 3: More epochs
python train_baseline.py --epochs 5 --wandb-name exp3-5epochs

# Compare in wandb dashboard: https://wandb.ai/<entity>/<project>
```

## Troubleshooting

### Wandb not logging

1. Check if wandb is installed: `pip install wandb`
2. Login to wandb: `wandb login`
3. Check if wandb is enabled: `--wandb` flag or `enabled = true` in config.ini
4. Check environment variable: `echo $WANDB_DISABLED`

### Model artifacts too large

Disable model logging in config.ini:

```ini
[wandb]
log_model = false
```

Or skip model artifacts on specific runs:

```python
# In your script
wandb_config = {"log_model": False}
```

### Custom metric logging

You can extend the wandb logging in any script:

```python
if self.wandb_run is not None:
    self.wandb_run.log({
        "custom_metric": value,
        "step": step,
    })
```

## Additional Resources

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Project README](README.md)
