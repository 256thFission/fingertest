# Training Script Migration Pattern

## Key Changes

### 1. New Imports (top of file)

```python
import sys
from src.config_v2 import ExperimentConfig, apply_overrides
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata, log_evaluation_artifacts
from src.git_utils import get_git_info, require_clean_git
from src.reproducibility import set_random_seeds, get_system_info, log_environment
```

### 2. Replace main() function

**OLD** (~80 lines):
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    # ... 15 more arguments
    args = parser.parse_args()

    trainer = BaselineTrainer(...)
    trainer.train(
        train_path=args.train_data,
        val_path=args.val_data,
        # ... pass all args
    )
```

**NEW** (~40 lines):
```python
def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--overrides", nargs="*", help="Config overrides: key=value")
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)
        logger.warning(f"Applied {len(args.overrides)} config overrides")

    # Validate
    errors = config.validate()
    if errors:
        logger.error("Config validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Check git
    if config.git.require_clean:
        require_clean_git(strict=True)

    # Set seeds
    set_random_seeds(config.reproducibility.random_seed, config.reproducibility.deterministic)

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)
    tracker.create_experiment_doc()
    tracker.update_status("running")
    tracker.log_start()

    # Get metadata
    git_info = get_git_info()
    system_info = get_system_info()
    if config.reproducibility.log_system_info:
        log_environment()

    # Initialize wandb
    wandb_run = None
    if config.wandb.enabled:
        wandb_run = init_wandb_with_metadata(config, git_info, system_info)

    # Train
    trainer = BaselineTrainer(config, wandb_run)
    trainer.train()

    # Evaluate (if evaluation is part of this script)
    from evaluate import Evaluator
    evaluator = Evaluator(config, wandb_run)
    metrics = evaluator.evaluate()

    # Update experiment doc
    if wandb_run:
        tracker.log_results(metrics, wandb_run.url)
        wandb_run.finish()
    else:
        tracker.log_results(metrics)

    tracker.update_status("complete")

    logger.info(f"Experiment {config.experiment.id} complete!")
    logger.info(f"Results: {tracker.exp_doc_path}")
```

### 3. Update Trainer Class

**OLD**:
```python
class BaselineTrainer:
    def __init__(self, model_name: str, output_dir: str, wandb_config: dict):
        self.model_name = model_name
        self.output_dir = output_dir
        self.wandb_config = wandb_config

    def train(self, train_path: str, val_path: str, batch_size: int, num_epochs: int, ...):
        # Many parameters
        ...
```

**NEW**:
```python
class BaselineTrainer:
    def __init__(self, config: ExperimentConfig, wandb_run=None):
        self.config = config
        self.wandb_run = wandb_run
        # Extract commonly used values
        self.model_name = config.model.base_model
        self.output_dir = config.baseline_training.output_dir

    def train(self):
        # All params from self.config
        batch_size = self.config.baseline_training.batch_size
        num_epochs = self.config.baseline_training.num_epochs
        # ...
```

## Files to Migrate

1. ✅ train_baseline.py - Pattern above
2. evaluate.py - Similar pattern, returns metrics dict
3. train_triplet.py - Same pattern
4. miner.py - Same pattern
5. run_loop.py - Orchestrates others

## Auto-Evaluation Integration

For train_baseline.py, optionally run evaluation at the end:

```python
# At end of main()
if config.evaluation:  # If evaluation config exists
    logger.info("Running automatic evaluation...")
    from evaluate import Evaluator
    evaluator = Evaluator(config, wandb_run)
    metrics = evaluator.evaluate()
    tracker.log_results(metrics, wandb_run.url if wandb_run else None)
```

## Usage Examples

**Before**:
```bash
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
  --wandb-project authorship-verification
```

**After**:
```bash
python train_baseline.py --config configs/experiments/002_lower_temperature.yaml

# Or with overrides for debugging:
python train_baseline.py \
  --config configs/experiments/002_lower_temperature.yaml \
  --overrides baseline_training.batch_size=16 wandb.enabled=false
```

This is SO much cleaner!
