# New Experiment System - Complete Guide

## 🎉 What's Been Implemented

### Core Infrastructure ✅
- `src/config_v2.py` - YAML config loader with inheritance
- `src/experiment_tracker.py` - Automatic doc generation & updates
- `src/wandb_utils.py` - Enhanced wandb with full metadata
- `src/git_utils.py` - Git reproducibility tracking
- `src/reproducibility.py` - Seed & system info tracking

### Configs ✅
- `configs/base.yaml` - Default config for all experiments
- `configs/presets/quick_test.yaml` - Fast testing
- `configs/presets/debug.yaml` - Debugging
- `configs/presets/full_training.yaml` - Production
- `configs/experiments/001_*.yaml` - Example experiments

### Helper Scripts ✅
- `scripts/new_experiment.py` - Create experiment (auto-increment ID)
- `scripts/validate_config.py` - Validate config before running
- `scripts/experiment_status.py` - View all experiments

### Dependencies ✅
- Added `pyyaml>=6.0` to requirements.txt

---

## 🚀 Using the New System

### Step 1: Create a New Experiment

```bash
# Simple: Auto-increments ID, creates config + doc
python scripts/new_experiment.py --name my_experiment

# With details:
python scripts/new_experiment.py \
  --name distance_filtering \
  --description "Test distance-filtered hard negative mining" \
  --hypothesis "Filtering negatives in 0.7-0.95 range improves EER by 2-4pp" \
  --parent 002
```

**Output:**
- `configs/experiments/004_my_experiment.yaml` - Config file
- `experiments/004_my_experiment.md` - Experiment doc
- Updates `experiments/README.md` log table

### Step 2: Edit the Config

```bash
vim configs/experiments/004_my_experiment.yaml
```

Customize parameters:
```yaml
base: "../base.yaml"

experiment:
  id: "004"
  name: "my_experiment"
  # ... edit hypothesis, expected results

# Override any base config values:
baseline_training:
  batch_size: 32  # Changed from 64
  num_epochs: 2   # Changed from 1

# Or add specific training type:
loop:
  num_iterations: 5
```

### Step 3: Validate Config

```bash
python scripts/validate_config.py configs/experiments/004_my_experiment.yaml
```

Checks:
- ✅ Config loads correctly
- ✅ All required fields present
- ✅ Data files exist
- ✅ Git status (if required)

### Step 4: Run Training

**Once scripts are migrated:**

```bash
# Train with config
python train_baseline.py --config configs/experiments/004_my_experiment.yaml

# Or with quick overrides for debugging:
python train_baseline.py \
  --config configs/experiments/004_my_experiment.yaml \
  --overrides baseline_training.batch_size=16 wandb.enabled=false
```

**What happens automatically:**
1. ✅ Creates/updates experiment doc
2. ✅ Sets experiment status to "running"
3. ✅ Initializes wandb with full metadata (git hash, system info, etc.)
4. ✅ Trains model
5. ✅ Evaluates model (if configured)
6. ✅ Updates experiment doc with results
7. ✅ Updates experiment log table
8. ✅ Sets status to "complete"
9. ✅ Links wandb ↔ doc bidirectionally

### Step 5: View Experiment Status

```bash
python scripts/experiment_status.py
```

**Output:**
```
Experiment Status
================================================================================
ID    Name                           Status          EER        ROC-AUC
--------------------------------------------------------------------------------
001   dimensional_collapse_fix       ✅ complete     17.35%     0.9010
002   lower_temperature              🔄 running      -          -
003   distance_filtering             📝 planning     -          -
004   my_experiment                  📝 planning     -          -
================================================================================
```

---

## 🔧 Completing the Migration

### Training Scripts to Migrate

You need to update these 5 scripts to use the new config system:

1. **train_baseline.py** (~500 lines) - Most important
2. **evaluate.py** (~680 lines)
3. **train_triplet.py** (~250 lines)
4. **miner.py** (~400 lines)
5. **run_loop.py** (~400 lines)

### Migration Pattern

See `MIGRATION_EXAMPLE.md` for the complete pattern. Key changes:

**1. Update imports (add):**
```python
from src.config_v2 import ExperimentConfig, apply_overrides
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata
from src.git_utils import get_git_info, require_clean_git
from src.reproducibility import set_random_seeds, get_system_info
```

**2. Replace main() function:**
- Remove ~20 argparse arguments
- Replace with single `--config` argument
- Load `ExperimentConfig` from YAML
- Initialize experiment tracker
- Initialize wandb with full metadata
- Run training/evaluation
- Auto-update docs

**3. Update Trainer classes:**
```python
# OLD
class BaselineTrainer:
    def __init__(self, model_name, output_dir, wandb_config):
        ...
    def train(self, train_path, val_path, batch_size, ...):  # 10+ params
        ...

# NEW
class BaselineTrainer:
    def __init__(self, config: ExperimentConfig, wandb_run=None):
        self.config = config
        self.wandb_run = wandb_run
    def train(self):  # All params from self.config
        ...
```

### Quick Migration Script

If you want to automate, create `scripts/migrate_script.py`:

```python
#!/usr/bin/env python3
"""
Quick migration helper - updates a training script to use new config system.
This does the mechanical parts; you'll still need to update the trainer class.
"""

import sys
import re
from pathlib import Path

def migrate_script(script_path: str):
    """Migrate a training script."""
    path = Path(script_path)
    content = path.read_text()

    # 1. Add new imports after existing imports
    new_imports = '''
from src.config_v2 import ExperimentConfig, apply_overrides
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata
from src.git_utils import get_git_info, require_clean_git
from src.reproducibility import set_random_seeds, get_system_info
'''

    # Find last import and insert after
    last_import = max([m.end() for m in re.finditer(r'^import .*$|^from .* import .*$', content, re.MULTILINE)])
    content = content[:last_import] + new_imports + content[last_import:]

    # 2. Replace main() function
    # This is script-specific, so just mark it
    content = re.sub(
        r'def main\(\):',
        '# TODO: Replace this main() function with new pattern (see MIGRATION_EXAMPLE.md)\ndef main():',
        content
    )

    # Save
    backup = path.with_suffix('.py.backup')
    path.rename(backup)
    path.write_text(content)

    print(f"✅ Migrated {path}")
    print(f"📦 Backup saved: {backup}")
    print(f"⚠️  Manual step: Update main() function and Trainer class")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/migrate_script.py train_baseline.py")
        sys.exit(1)
    migrate_script(sys.argv[1])
```

---

## 📋 Cleanup Checklist

After migration is complete:

### Delete Old Files
```bash
# Old config system (no longer needed)
rm config.py
rm config.ini

# Old status files (replaced by new system)
rm PROJECT_STATUS.txt
rm QUICKREF.txt

# Migration planning docs (keep or move to docs/)
# FULL_MIGRATION_PLAN.md
# UNIFIED_CONFIG_PLAN.md
# WANDB_INTEGRATION_AUDIT.md
```

### Update README.md

Replace old usage examples with:

```markdown
## Quick Start

### 1. Create Experiment
\`\`\`bash
python scripts/new_experiment.py --name my_experiment
\`\`\`

### 2. Edit Config
\`\`\`bash
vim configs/experiments/XXX_my_experiment.yaml
\`\`\`

### 3. Run Training
\`\`\`bash
python train_baseline.py --config configs/experiments/XXX_my_experiment.yaml
\`\`\`

Results automatically documented in `experiments/XXX_my_experiment.md`!
```

---

## 🎯 Key Benefits

### Before
```bash
# Create experiment (manual)
cp experiments/EXPERIMENT_TEMPLATE.md experiments/003_name.md
vim experiments/003_name.md  # Fill in manually

# Train (many args)
python train_baseline.py \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir models/baseline \
  --batch-size 64 \
  --epochs 1 \
  --lr 2e-5 \
  --warmup-steps 1000 \
  --fp16 \
  --wandb \
  --wandb-project authorship-verification \
  --wandb-entity your-entity

# Evaluate (separate, more args)
python evaluate.py \
  --model models/baseline \
  --test-data data/processed/test.parquet \
  --train-data data/processed/train.parquet \
  --whitening \
  --output outputs/evaluation

# Update docs (manual)
vim experiments/003_name.md  # Copy results manually
vim experiments/README.md    # Update table manually
# Remember to add wandb URL manually
```

### After
```bash
# Create experiment (automatic)
python scripts/new_experiment.py --name my_experiment

# Edit config once
vim configs/experiments/XXX_my_experiment.yaml

# Train + evaluate + document (all automatic)
python train_baseline.py --config configs/experiments/XXX_my_experiment.yaml

# Everything updates automatically:
# ✅ Experiment doc created
# ✅ Wandb initialized with full metadata
# ✅ Training runs
# ✅ Evaluation runs
# ✅ Results added to doc
# ✅ Wandb link added to doc
# ✅ Experiment log table updated
# ✅ Status tracked

# View status
python scripts/experiment_status.py
```

**Result:** 20+ CLI args → 1 config file. Manual doc updates → automatic. ~10 steps → 2 steps.

---

## 🔍 Troubleshooting

### Config not found
```bash
# Check path
ls configs/experiments/

# Validate
python scripts/validate_config.py configs/experiments/XXX_name.yaml
```

### Validation errors
```bash
# See detailed errors
python scripts/validate_config.py configs/experiments/XXX_name.yaml

# Common fixes:
# - Data files don't exist: run preprocess first
# - Git not clean: commit changes or set git.require_clean=false
# - Invalid YAML: check syntax
```

### Override not working
```bash
# Format: key=value (no spaces)
python train_baseline.py \
  --config path/to/config.yaml \
  --overrides baseline_training.batch_size=16 wandb.enabled=false

# Check what was applied (look for warning in output):
# "⚠️  Applied 2 config overrides"
```

---

## 📊 What Gets Logged to Wandb

The new system logs comprehensive metadata:

- **Experiment:** ID, name, description, hypothesis, expected results, parent
- **Git:** commit hash, branch, dirty flag, remote URL
- **System:** Python version, PyTorch version, CUDA, GPU name
- **Data:** paths, version, sample counts
- **Model:** architecture, all hyperparameters
- **Training:** All config values (flattened)
- **Artifacts:** Config YAML, experiment doc, evaluation plots

You can reproduce any experiment from wandb alone!

---

## 🎓 Example Workflow

```bash
# 1. Create experiment
$ python scripts/new_experiment.py --name test_larger_batch
✅ Created config: configs/experiments/005_test_larger_batch.yaml
✅ Created doc: experiments/005_test_larger_batch.md
✅ Updated experiment log
✅ Experiment 005 created!

# 2. Customize config
$ vim configs/experiments/005_test_larger_batch.yaml
# Change batch_size to 128

# 3. Validate
$ python scripts/validate_config.py configs/experiments/005_test_larger_batch.yaml
✅ Config loaded successfully
✅ Config is valid
✅ Clean git state
✅ Config ready to use!

# 4. Train
$ python train_baseline.py --config configs/experiments/005_test_larger_batch.yaml
Creating experiment doc...
Setting random seed: 42
Initializing wandb...
Wandb URL: https://wandb.ai/...
Training...
Evaluating...
Updating experiment doc with results...
✅ Experiment 005 complete!
📊 Results: experiments/005_test_larger_batch.md

# 5. Check status
$ python scripts/experiment_status.py
ID    Name                    Status          EER        ROC-AUC
005   test_larger_batch       ✅ complete     16.23%     0.9145

# 6. View results
$ cat experiments/005_test_larger_batch.md
# Shows complete results, wandb link, visualizations

# 7. Continue iterating
$ python scripts/new_experiment.py --name followup --parent 005
```

---

## ✅ Next Steps

1. **Complete migration** of training scripts (see MIGRATION_EXAMPLE.md)
2. **Test the system** with one experiment end-to-end
3. **Delete old config files** (config.py, config.ini)
4. **Update README.md** with new usage examples
5. **Start using it!** Create new experiments with the new system

All infrastructure is in place. The system is ready to use once training scripts are migrated!
