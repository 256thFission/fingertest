# Setup and Usage Guide

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  

# Install dependencies
uv pip install -r requirements.txt

# Login to wandb (optional but recommended)
wandb login
```

### 2. Create New Experiment

```bash
# Auto-generates experiment ID (next available number)
python scripts/new_experiment.py --name my_experiment_name

# With more details
python scripts/new_experiment.py \
  --name distance_filtering \
  --description "Test distance-filtered hard negative mining" \
  --hypothesis "Filtering negatives will improve EER by 2-4pp" \
  --parent 002
```

This creates:
- `configs/experiments/00X_my_experiment_name.yaml`
- `experiments/00X_my_experiment_name.md`
- Updates experiment log table

### 3. Edit Configuration

```bash
# Edit the generated config
vim configs/experiments/003_my_experiment_name.yaml
```

Example config:
```yaml
base: "../base.yaml"

experiment:
  id: "003"
  name: "my_experiment_name"
  description: "What this experiment does"
  hypothesis: "What you expect to happen"
  expected_results:
    eer: "<12%"
    roc_auc: ">0.95"
  parent_experiment: "002"
  status: "planning"

# Override any base settings
baseline_training:
  batch_size: 32  # Override default 64
  num_epochs: 2   # Override default 1
```

### 4. Validate Configuration

```bash
# Check config before running
python scripts/validate_config.py configs/experiments/003_my_experiment_name.yaml
```

### 5. Run Training

```bash
# Run with config
python train_baseline.py --config configs/experiments/003_my_experiment_name.yaml

# With overrides for debugging
python train_baseline.py \
  --config configs/experiments/003_my_experiment_name.yaml \
  --overrides baseline_training.batch_size=16 wandb.enabled=false
```

### 6. Check Results

```bash
# View experiment status
python scripts/experiment_status.py

# Check experiment doc (auto-updated)
cat experiments/003_my_experiment_name.md

# View wandb (URL in experiment doc)
```

---

## Configuration System

### Config Hierarchy

```
configs/base.yaml                    # Default values for everything
    ↓ (inherits)
configs/experiments/003_name.yaml    # Experiment-specific overrides
    ↓ (can override at runtime)
--overrides batch_size=16            # CLI overrides for debugging
```

### Config Sections

**experiment:** Metadata
- id, name, description, hypothesis
- expected_results, parent_experiment
- status (planning|running|complete|failed)

**data:** Data paths and versioning
- train_path, val_path, test_path
- version (for tracking data changes)

**model:** Architecture
- base_model (roberta-base, roberta-large, etc.)
- max_seq_length, embedding_dim

**baseline_training:** Baseline training params
- batch_size, num_epochs, learning_rate
- loss: scale, temperature
- output_dir

**evaluation:** Evaluation settings
- use_whitening, num_positive_pairs
- target_eer, target_roc_auc

**wandb:** Wandb integration
- enabled, project, entity
- group, tags

**reproducibility:** Reproducibility settings
- random_seed, deterministic
- log_system_info

---

## Presets

Use presets for common scenarios:

**Quick Testing:**
```yaml
base: "../presets/quick_test.yaml"
# Small batch, few samples, no wandb
```

**Debug:**
```yaml
base: "../presets/debug.yaml"
# Minimal config for debugging
```

**Full Training:**
```yaml
base: "../presets/full_training.yaml"
# Production settings: more epochs, larger samples
```

---

## Experiment Workflow

### Standard Flow

1. **Plan:** Create experiment, document hypothesis
2. **Configure:** Edit YAML config
3. **Validate:** Check config validity
4. **Run:** Execute training
5. **Review:** Check auto-updated docs and wandb
6. **Iterate:** Create child experiment with improvements

### Example Session

```bash
# 1. Create experiment 004
python scripts/new_experiment.py --name larger_model --parent 003

# 2. Edit config to use roberta-large
vim configs/experiments/004_larger_model.yaml
# Change: model.base_model = "roberta-large"

# 3. Validate
python scripts/validate_config.py configs/experiments/004_larger_model.yaml

# 4. Run
python train_baseline.py --config configs/experiments/004_larger_model.yaml

# 5. Results automatically logged to:
#    - experiments/004_larger_model.md
#    - wandb (link in doc)
#    - experiments/README.md (table updated)
```

---

## Scripts Reference

### new_experiment.py

Create new experiment with auto-incremented ID.

```bash
python scripts/new_experiment.py --name NAME [OPTIONS]

Options:
  --name NAME           Experiment name (required, snake_case)
  --description DESC    Short description
  --hypothesis HYPO     Experiment hypothesis
  --parent ID           Parent experiment ID
  --preset PRESET       Base preset (base|quick_test|debug|full_training)
```

### validate_config.py

Validate config before running.

```bash
python scripts/validate_config.py CONFIG_PATH

Checks:
- Config loads successfully
- Required fields present
- Data paths exist
- Git status (if required)
```

### experiment_status.py

View all experiments and their status.

```bash
python scripts/experiment_status.py [--detail]

Shows:
- Experiment ID and name
- Status (planning/running/complete/failed)
- Latest metrics (EER, ROC-AUC)
- Wandb link status
```

---

## Troubleshooting

### Config Not Found

```
Error: Config not found: configs/experiments/003_name.yaml
```

**Fix:** Create experiment first with `new_experiment.py`

### Data Path Not Found

```
Error: data.train_path does not exist: data/processed/train.parquet
```

**Fix:** Run preprocessing first:
```bash
python preprocess_parquet.py --skip-channel-mapping
```

### Git Dirty Error

```
Error: Git has uncommitted changes
```

**Fix:** Either:
- Commit changes: `git add . && git commit -m "message"`
- Disable check: Edit config `git.require_clean: false`

### Import Error

```
ModuleNotFoundError: No module named 'src.config_v2'
```

**Fix:** Ensure you're in repo root:
```bash
cd /path/to/fingertest
python train_baseline.py --config ...
```

### Wandb Not Logging

**Check:**
1. `wandb.enabled: true` in config
2. `wandb login` completed
3. Not overridden: `--overrides wandb.enabled=false`

---

## Best Practices

### Experiment Naming

Use descriptive snake_case names:
-  `lower_temperature_training`
-  `distance_filtering_0.7_0.95`
-  `exp1`, `test`, `new`

### Hypothesis Documentation

Write specific, testable hypotheses:
-  "Lowering temperature to 0.01 will improve EER by 3-5pp"
-  "Make model better"

### Config Organization

- Use `base.yaml` for common defaults
- Use experiment configs for variations
- Use presets for common scenarios
- Use `--overrides` only for debugging

### Git Workflow

```bash
# Before experiment
git add configs/experiments/00X_name.yaml
git commit -m "Add experiment 00X: description"

# After experiment
git add experiments/00X_name.md
git commit -m "Experiment 00X results: EER=XX.X%"
```

---

## Advanced Usage

### Config Overrides

Override any config value at runtime:

```bash
python train_baseline.py \
  --config configs/experiments/003_name.yaml \
  --overrides \
    baseline_training.batch_size=32 \
    baseline_training.num_epochs=2 \
    wandb.enabled=false
```

### Custom Wandb Tags

```yaml
wandb:
  tags:
    - "ablation-study"
    - "temperature-sweep"
    - "v2.0"
```

### Multiple GPUs

```yaml
hardware:
  num_gpus: 2
  use_distributed: true
```

(Note: Requires code changes for distributed training)

---

## File Locations

```
configs/
  base.yaml              # Base defaults
  experiments/
    001_*.yaml           # Experiment configs
  presets/
    quick_test.yaml      # Presets

experiments/
  README.md              # Experiment log (auto-updated)
  001_*.md               # Experiment docs (auto-updated)

scripts/
  new_experiment.py      # Create experiment
  validate_config.py     # Validate config
  experiment_status.py   # View status

src/
  config_v2.py           # Config system
  experiment_tracker.py  # Doc automation
  wandb_utils.py         # Wandb integration
  git_utils.py           # Git metadata
  reproducibility.py     # Seeds & system info
```
