# Unified Config & Experiment Tracking Plan

## Current State Analysis

### Wandb Integration Issues

**Current Problems:**
1. ❌ Wandb not linked to experiment documentation
2. ❌ No experiment ID tracking in wandb runs
3. ❌ Missing metadata: git hash, data version, experiment hypothesis
4. ❌ Manual experiment doc updates separate from wandb
5. ❌ Can't easily reproduce runs from wandb config alone
6. ❌ No automatic experiment doc generation from runs

**What's Working:**
- ✅ Basic wandb logging (metrics, hyperparameters)
- ✅ Tags and project organization
- ✅ Artifact logging capability

### Config System Issues

**Current Problems:**
1. ❌ Split between INI file and CLI args (precedence unclear)
2. ❌ Scripts don't consistently use config.py
3. ❌ No experiment-specific configs (all share config.ini)
4. ❌ No config versioning or history
5. ❌ INI format limited (no nested structures, lists, etc.)
6. ❌ Hard to track "which config produced which results"

**What's Working:**
- ✅ Centralized config.py with dataclasses
- ✅ Environment variable overrides for wandb

## Proposed Solution

### Part 1: YAML-Based Config System

**Philosophy:**
- **Single source of truth:** YAML files in `configs/` directory
- **Experiment-specific:** Each experiment gets its own config file
- **Reproducible:** Config saved with every run
- **Hierarchical:** Support for config inheritance and overrides
- **Data-driven:** No CLI args except config file path

**Structure:**
```
configs/
├── base.yaml                    # Base config (shared defaults)
├── experiments/
│   ├── 001_baseline.yaml       # Experiment-specific configs
│   ├── 002_lower_temp.yaml
│   └── 003_distance_filter.yaml
└── presets/
    ├── quick_test.yaml         # Quick testing preset
    ├── full_training.yaml      # Production training
    └── debug.yaml              # Debug mode
```

**Example YAML:**
```yaml
# configs/experiments/002_lower_temp.yaml
experiment:
  id: "002"
  name: "lower_temperature_training"
  description: "Train with scale=100.0 (temp=0.01) for sharper boundaries"
  hypothesis: "Lower temperature will improve EER by 3-5 pp"
  baseline_experiment: "001"  # Parent experiment
  status: "running"  # planning|running|complete|failed

data:
  train_path: "data/processed/train.parquet"
  val_path: "data/processed/val.parquet"
  test_path: "data/processed/test.parquet"
  version: "v1.0"  # Data version tracking

model:
  base_model: "roberta-base"
  max_seq_length: 512
  embedding_dim: 768

baseline_training:
  batch_size: 64
  num_epochs: 1
  learning_rate: 2.0e-5
  warmup_steps: 1000
  fp16: true
  loss:
    type: "MultipleNegativesRankingLoss"
    scale: 100.0  # temperature = 1/scale = 0.01
    temperature: 0.01
  output_dir: "models/baseline_exp002"

evaluation:
  use_whitening: true
  num_positive_pairs: 2000
  num_negative_pairs: 2000
  target_eer: 0.15  # 15% target for this experiment
  target_roc_auc: 0.95

wandb:
  enabled: true
  project: "authorship-verification"
  entity: null
  tags: ["exp-002", "lower-temp", "baseline"]
  group: "temperature-experiments"
  notes: "Testing scale=100.0 to fix dimensional collapse"

git:
  auto_commit: false  # Auto-commit results
  require_clean: false  # Require clean working directory

reproducibility:
  random_seed: 42
  deterministic: true
```

### Part 2: Enhanced Wandb Integration

**Key Additions:**
1. **Experiment Metadata Logging:**
   ```python
   wandb.config.update({
       "experiment_id": "002",
       "experiment_name": "lower_temperature_training",
       "hypothesis": "Lower temperature will improve EER by 3-5 pp",
       "git_hash": get_git_hash(),
       "git_dirty": has_uncommitted_changes(),
       "data_version": "v1.0",
       "config_file": "configs/experiments/002_lower_temp.yaml",
       "parent_experiment": "001",
   })
   ```

2. **Automatic Experiment Doc Generation:**
   - Generate experiment markdown from wandb run
   - Auto-populate results section after training
   - Link wandb run URL in experiment doc
   - Update experiments/README.md log table

3. **Config Artifact Logging:**
   ```python
   # Save exact config used for this run
   wandb.save("config_used.yaml")
   wandb.log_artifact(config_path, type="config")
   ```

4. **Bidirectional Linking:**
   - Wandb run → experiment doc (via notes/tags)
   - Experiment doc → wandb run (via URL in markdown)

### Part 3: Implementation Plan

#### Phase 1: Config System Refactor (1-2 hours)

**1.1 Create YAML Config Loader**
- New file: `config_v2.py` with YAML support
- Support config inheritance (base → experiment)
- Validate configs against schema
- Keep backward compatibility with config.ini

```python
from dataclasses import dataclass
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML with inheritance."""
        # Load experiment config
        with open(yaml_path) as f:
            exp_config = yaml.safe_load(f)

        # Load and merge base config if exists
        if "base" in exp_config or Path("configs/base.yaml").exists():
            with open("configs/base.yaml") as f:
                base_config = yaml.safe_load(f)
            exp_config = merge_configs(base_config, exp_config)

        return cls(**exp_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for wandb logging."""
        ...
```

**1.2 Create Base Config**
```bash
# Create base config with all defaults
configs/base.yaml
```

**1.3 Migrate Existing Config**
```bash
# Convert config.ini → configs/base.yaml
python scripts/migrate_config.py
```

#### Phase 2: Training Script Integration (2-3 hours)

**2.1 Update Training Scripts**
Modify `train_baseline.py`, `train_triplet.py`, `run_loop.py`:

```python
# OLD: Multiple CLI args
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", ...)
    parser.add_argument("--batch-size", ...)
    # ... 20 more args
    args = parser.parse_args()

# NEW: Single config arg
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                       help="Path to experiment config YAML")
    parser.add_argument("--overrides", nargs="*",
                       help="Config overrides: key=value")
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides (for quick testing)
    if args.overrides:
        for override in args.overrides:
            key, value = override.split("=")
            set_nested_config(config, key, value)

    # Train with config
    trainer = BaselineTrainer(config)
    trainer.train()
```

**2.2 Enhanced Wandb Initialization**
```python
def init_wandb(config: ExperimentConfig):
    """Initialize wandb with full experiment metadata."""

    # Get git info
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    git_dirty = subprocess.call(["git", "diff-index", "--quiet", "HEAD"]) != 0

    # Initialize wandb
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=f"exp-{config.experiment.id}-{config.experiment.name}",
        tags=config.wandb.tags + [f"exp-{config.experiment.id}"],
        group=config.wandb.group,
        notes=config.experiment.hypothesis,
        config={
            # Experiment metadata
            "experiment_id": config.experiment.id,
            "experiment_name": config.experiment.name,
            "hypothesis": config.experiment.hypothesis,
            "description": config.experiment.description,
            "parent_experiment": config.experiment.baseline_experiment,

            # Reproducibility
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "data_version": config.data.version,
            "config_file": str(config._yaml_path),
            "random_seed": config.reproducibility.random_seed,

            # All training params (flattened)
            **config.to_flat_dict(),
        }
    )

    # Log config as artifact
    artifact = wandb.Artifact(
        f"config-exp-{config.experiment.id}",
        type="config",
        description=f"Config for experiment {config.experiment.id}"
    )
    artifact.add_file(config._yaml_path)
    run.log_artifact(artifact)

    return run
```

#### Phase 3: Experiment Tracking Integration (2 hours)

**3.1 Automatic Experiment Doc Updates**
```python
class ExperimentTracker:
    """Manages experiment documentation and wandb integration."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_dir = Path("experiments")
        self.exp_doc = self.exp_dir / f"{config.experiment.id}_{config.experiment.name}.md"

    def create_experiment_doc(self):
        """Create experiment doc from template if not exists."""
        if not self.exp_doc.exists():
            template = Path("experiments/EXPERIMENT_TEMPLATE.md").read_text()

            # Fill in known fields from config
            doc = template.format(
                exp_id=self.config.experiment.id,
                title=self.config.experiment.name,
                date=datetime.now().strftime("%Y-%m-%d"),
                hypothesis=self.config.experiment.hypothesis,
                description=self.config.experiment.description,
                # ... more fields
            )

            self.exp_doc.write_text(doc)

    def update_results(self, wandb_run_url: str, metrics: Dict[str, float]):
        """Update experiment doc with results after training."""
        # Read current doc
        doc = self.exp_doc.read_text()

        # Update results section
        results_table = f"""
### Metrics

| Metric | Value | Change | Target |
|--------|-------|--------|--------|
| EER | {metrics['eer']:.2%} | {metrics.get('eer_change', 'N/A')} | <15% |
| ROC-AUC | {metrics['roc_auc']:.4f} | {metrics.get('auc_change', 'N/A')} | >0.95 |

**Wandb Run:** {wandb_run_url}
"""

        # Insert/replace results
        doc = update_markdown_section(doc, "## Results", results_table)
        self.exp_doc.write_text(doc)

        # Update experiment log
        self.update_experiment_log(metrics)

    def update_experiment_log(self, metrics: Dict[str, float]):
        """Update experiments/README.md log table."""
        log_file = self.exp_dir / "README.md"
        log = log_file.read_text()

        # Find and update row for this experiment
        new_row = f"| {self.config.experiment.id} | {datetime.now().strftime('%Y-%m-%d')} | {self.config.experiment.name} | ✅ Complete | {metrics['eer']:.2%} | {metrics['roc_auc']:.4f} | See [doc]({self.exp_doc.name}) |"

        log = update_table_row(log, self.config.experiment.id, new_row)
        log_file.write_text(log)
```

**3.2 Integration in Training Scripts**
```python
def train_with_tracking():
    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Create experiment tracker
    tracker = ExperimentTracker(config)
    tracker.create_experiment_doc()

    # Initialize wandb
    run = init_wandb(config)

    # Train model
    trainer = BaselineTrainer(config)
    trainer.train()

    # Evaluate
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate()

    # Update experiment doc with results
    tracker.update_results(run.url, metrics)

    # Finish wandb
    run.finish()
```

#### Phase 4: Helper Scripts (1 hour)

**4.1 Config Generator**
```bash
# Generate new experiment config from template
python scripts/new_experiment.py \
  --id 003 \
  --name "distance_filtering" \
  --description "Test distance-filtered hard negative mining" \
  --parent 002

# Creates: configs/experiments/003_distance_filtering.yaml
# Creates: experiments/003_distance_filtering.md
# Updates: experiments/README.md
```

**4.2 Config Validation**
```bash
# Validate config before running
python scripts/validate_config.py configs/experiments/002_lower_temp.yaml

# Output:
# ✅ Config valid
# ✅ All paths exist
# ✅ Git is clean (or warning if dirty)
# ⚠️  Warning: baseline_experiment "001" not found
```

**4.3 Experiment Status Checker**
```bash
# Check status of all experiments
python scripts/experiment_status.py

# Output:
# ID  | Name                  | Status    | EER     | Wandb
# 001 | dimensional_collapse  | Complete  | 17.35%  | ✅ Linked
# 002 | lower_temp            | Running   | -       | ✅ Active
# 003 | distance_filtering    | Planned   | -       | -
```

### Part 4: Migration Strategy

**For Existing Work:**
1. Create config for completed experiments retroactively
2. Link existing wandb runs to experiment docs
3. Update experiment docs with wandb URLs

**For New Work:**
1. All new experiments MUST use YAML configs
2. Auto-generate experiment docs from configs
3. Bidirectional linking enforced

### Part 5: Benefits

**Reproducibility:**
- ✅ Exact config saved with every run
- ✅ Git hash tracked automatically
- ✅ Data version tracked
- ✅ Can reproduce any experiment: `python train_baseline.py --config configs/experiments/002_lower_temp.yaml`

**Experiment Tracking:**
- ✅ Wandb runs automatically linked to experiment docs
- ✅ Experiment docs auto-updated with results
- ✅ Single source of truth for experiment status
- ✅ Easy to compare experiments (all metadata in wandb)

**Ease of Use:**
- ✅ No more 20+ CLI args to remember
- ✅ Config inheritance (don't repeat yourself)
- ✅ Quick testing with presets
- ✅ Easy to share configs (just send YAML file)

**Research Quality:**
- ✅ Hypothesis documented before running
- ✅ All changes tracked (git + wandb)
- ✅ Results automatically logged
- ✅ Easy to generate papers/reports from experiment history

## Timeline

- **Phase 1 (Config System):** 1-2 hours
- **Phase 2 (Training Integration):** 2-3 hours
- **Phase 3 (Experiment Tracking):** 2 hours
- **Phase 4 (Helper Scripts):** 1 hour

**Total:** 6-8 hours of focused work

## Example Workflow (After Implementation)

```bash
# 1. Create new experiment
python scripts/new_experiment.py --id 004 --name "large_model" --parent 002

# 2. Edit config
vim configs/experiments/004_large_model.yaml
# Change: model.base_model = "roberta-large"

# 3. Run experiment
python train_baseline.py --config configs/experiments/004_large_model.yaml

# 4. Everything happens automatically:
#    - Wandb initialized with full metadata
#    - Experiment doc created
#    - Training runs
#    - Results logged to wandb
#    - Experiment doc updated with results
#    - Experiments/README.md updated

# 5. Review results
cat experiments/004_large_model.md
```

## Questions to Answer

1. **Config inheritance depth?**
   - Proposal: 2 levels (base → experiment)

2. **CLI overrides?**
   - Proposal: Allow for debugging, but warn if used

3. **Backward compatibility?**
   - Proposal: Keep config.ini working for 1-2 releases

4. **Config validation strictness?**
   - Proposal: Strict validation with `--skip-validation` flag

5. **Experiment status workflow?**
   - Proposal: Auto-update status (planning → running → complete)

## Next Steps

1. Review and approve plan
2. Start with Phase 1 (config system)
3. Test on single script (train_baseline.py)
4. Roll out to all scripts
5. Migrate existing experiments
