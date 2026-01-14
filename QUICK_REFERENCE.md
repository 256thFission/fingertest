# Quick Reference - New Experiment System

## 🚀 Common Commands

### Create New Experiment
```bash
# Auto-increment ID
python scripts/new_experiment.py --name my_experiment

# With details
python scripts/new_experiment.py \
  --name distance_filtering \
  --description "Test distance-filtered mining" \
  --hypothesis "Filtering improves EER by 2-4pp" \
  --parent 002 \
  --preset base
```

### Validate Config
```bash
python scripts/validate_config.py configs/experiments/XXX_name.yaml
```

### View All Experiments
```bash
python scripts/experiment_status.py
```

### Run Training (after migration)
```bash
# Standard
python train_baseline.py --config configs/experiments/XXX_name.yaml

# With overrides for debugging
python train_baseline.py \
  --config configs/experiments/XXX_name.yaml \
  --overrides baseline_training.batch_size=16 wandb.enabled=false
```

## 📁 File Locations

### Your experiment
- Config: `configs/experiments/XXX_yourname.yaml`
- Doc: `experiments/XXX_yourname.md`
- Model: `models/baseline_expXXX/` (or as configured)
- Results: `outputs/evaluation_expXXX/`

### Presets
- Fast testing: `configs/presets/quick_test.yaml`
- Debugging: `configs/presets/debug.yaml`
- Production: `configs/presets/full_training.yaml`

## 🔧 Config Overrides

Format: `key=value` (nested keys use dots)

```bash
# Single override
--overrides baseline_training.batch_size=16

# Multiple overrides
--overrides \
  baseline_training.batch_size=16 \
  baseline_training.num_epochs=2 \
  wandb.enabled=false
```

Common overrides:
- `baseline_training.batch_size=N`
- `baseline_training.num_epochs=N`
- `wandb.enabled=false`
- `evaluation.use_whitening=true`
- `reproducibility.random_seed=N`

## 📊 Checking Status

### View experiment status
```bash
python scripts/experiment_status.py
```

### Check experiment doc
```bash
cat experiments/XXX_yourname.md
```

### View results
```bash
cat outputs/evaluation_expXXX/metrics.json
```

### Open wandb
Wandb URL is in experiment doc after training completes.

## 🎯 Typical Workflow

```bash
# 1. Create
python scripts/new_experiment.py --name my_test

# 2. Edit config
vim configs/experiments/XXX_my_test.yaml

# 3. Validate
python scripts/validate_config.py configs/experiments/XXX_my_test.yaml

# 4. Run
python train_baseline.py --config configs/experiments/XXX_my_test.yaml

# 5. Check results
cat experiments/XXX_my_test.md
python scripts/experiment_status.py
```

## 🐛 Debugging

### Quick test with small data
```bash
python scripts/new_experiment.py --name debug_test --preset debug
python train_baseline.py \
  --config configs/experiments/XXX_debug_test.yaml \
  --overrides wandb.enabled=false
```

### Validation errors
```bash
# See detailed errors
python scripts/validate_config.py configs/experiments/XXX_name.yaml

# Common fixes:
# - Data not found: run preprocess first
# - Git dirty: commit changes or set git.require_clean=false in config
# - Invalid YAML: check syntax
```

### Config not loading
```bash
# Check file exists
ls configs/experiments/

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('configs/experiments/XXX_name.yaml'))"
```

## 📝 Config Structure

```yaml
base: "../base.yaml"  # Or "../presets/debug.yaml"

experiment:
  id: "XXX"  # Auto-generated
  name: "my_experiment"
  description: "What this does"
  hypothesis: "What you expect"
  expected_results:
    eer: "<15%"
    roc_auc: ">0.95"
  parent_experiment: "002"  # Optional
  status: "planning"

# Override any base config values here
baseline_training:
  batch_size: 32  # Override from base (64)
  num_epochs: 2   # Override from base (1)

# Or specify different training
loop:
  num_iterations: 5
```

## 🎓 Learning

1. **Start here:** `NEW_SYSTEM_GUIDE.md`
2. **Migration:** `MIGRATION_EXAMPLE.md`
3. **Status:** `IMPLEMENTATION_STATUS.md`
4. **Full plan:** `FULL_MIGRATION_PLAN.md`

## ⚡ Power User Tips

### Quick experiment with preset
```bash
# Create with quick_test preset (small batch, no wandb)
python scripts/new_experiment.py --name quick_test --preset quick_test
python train_baseline.py --config configs/experiments/XXX_quick_test.yaml
```

### Chain experiments
```bash
# Experiment 1
python scripts/new_experiment.py --name exp1
# ... train ...

# Experiment 2 (builds on 1)
python scripts/new_experiment.py --name exp2 --parent XXX
# Config inherits from parent, you just override what changes
```

### Check git before running
```bash
git status
git diff  # Make sure you want to run with current changes

# Or require clean git in config:
# git:
#   require_clean: true
```

### Compare experiments
```bash
# View all
python scripts/experiment_status.py

# Compare two
diff experiments/001_*.md experiments/002_*.md
```

## 🆘 Help

If stuck:
1. Check `NEW_SYSTEM_GUIDE.md`
2. Validate config: `python scripts/validate_config.py <config>`
3. Check experiment status: `python scripts/experiment_status.py`
4. Use debug preset for testing
5. Check wandb for run details

