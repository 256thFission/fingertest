# Full Migration Plan: Unified Config & Experiment Tracking System

## Executive Summary

**Goal:** Migrate entire codebase to YAML-based config system with automated experiment tracking, full reproducibility, and seamless wandb integration.

**Scope:**
- 5 core Python scripts (~2245 lines)
- New infrastructure (~1500 lines)
- Migration of existing experiments
- Documentation updates

**Timeline:** 8-12 hours of focused work

**Outcome:**
- Single config file per experiment
- Automatic experiment documentation
- Full reproducibility
- Bidirectional wandb ↔ docs linking
- No more manual doc updates

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure (Before/After)](#file-structure)
3. [Detailed Implementation Plan](#implementation-plan)
4. [Migration Strategy](#migration-strategy)
5. [Testing Strategy](#testing-strategy)
6. [Rollout Plan](#rollout-plan)
7. [Clarifying Questions](#clarifying-questions)

---

## Architecture Overview

### Current Architecture

```
User → CLI (20+ args) → Training Script → Wandb (basic metrics)
                                       ↓
                                  Manual Updates
                                       ↓
                            Experiment Docs (markdown)
```

**Problems:**
- Manual experiment doc creation
- No automatic linking
- Config scattered (CLI + config.ini)
- Can't reproduce from wandb alone
- Missing metadata (git, data version)

### Target Architecture

```
User → YAML Config → Config Loader → Training Script → Experiment Tracker
                                           ↓                    ↓
                                    Wandb (full metadata)    Auto-update docs
                                           ↓                    ↓
                                    [Bidirectional linking]
```

**Benefits:**
- Single source of truth (YAML)
- Auto experiment doc generation
- Full reproducibility metadata
- Automatic doc updates
- Clean separation of concerns

### Core Components

1. **Config System** (`config_v2.py`)
   - YAML loader with inheritance
   - Schema validation
   - Config merging and overrides
   - Environment variable support

2. **Experiment Tracker** (`experiment_tracker.py`)
   - Experiment doc generation
   - Result tracking
   - Experiment log updates
   - Status management

3. **Wandb Integration** (`wandb_utils.py`)
   - Enhanced initialization with metadata
   - Artifact logging
   - Config versioning
   - Reproducibility tracking

4. **Git Utils** (`git_utils.py`)
   - Get commit hash
   - Check for uncommitted changes
   - Get branch info
   - Validate clean state

5. **Helper Scripts** (`scripts/`)
   - Create new experiment
   - Validate config
   - Compare experiments
   - Migrate legacy configs

---

## File Structure

### Before Migration

```
fingertest/
├── config.ini                    # INI format config
├── config.py                     # Current config loader
├── train_baseline.py             # ~20 CLI args
├── train_triplet.py              # ~15 CLI args
├── run_loop.py                   # ~18 CLI args
├── evaluate.py                   # ~12 CLI args
├── miner.py                      # ~10 CLI args
├── experiments/
│   ├── README.md
│   ├── EXPERIMENT_TEMPLATE.md
│   └── 001_*.md                  # Manual updates
└── requirements.txt
```

### After Migration

```
fingertest/
├── configs/
│   ├── base.yaml                          # Base defaults (NEW)
│   ├── schema.yaml                        # Config schema (NEW)
│   ├── experiments/                       # (NEW)
│   │   ├── 001_dimensional_collapse.yaml
│   │   ├── 002_lower_temp.yaml
│   │   └── 003_distance_filter.yaml
│   └── presets/                           # (NEW)
│       ├── quick_test.yaml                # Fast testing
│       ├── full_training.yaml             # Production
│       └── debug.yaml                     # Debug mode
│
├── src/                                   # (NEW - organized code)
│   ├── __init__.py
│   ├── config_v2.py                       # YAML config loader
│   ├── experiment_tracker.py              # Experiment management
│   ├── wandb_utils.py                     # Enhanced wandb integration
│   ├── git_utils.py                       # Git metadata
│   └── reproducibility.py                 # Reproducibility utils
│
├── scripts/                               # (NEW - helper scripts)
│   ├── new_experiment.py                  # Create experiment
│   ├── validate_config.py                 # Validate YAML
│   ├── experiment_status.py               # Check status
│   ├── compare_experiments.py             # Compare results
│   ├── migrate_config.py                  # INI → YAML
│   └── backfill_experiments.py            # Migrate old experiments
│
├── train_baseline.py                      # MODIFIED - uses config_v2
├── train_triplet.py                       # MODIFIED - uses config_v2
├── run_loop.py                            # MODIFIED - uses config_v2
├── evaluate.py                            # MODIFIED - uses config_v2
├── miner.py                               # MODIFIED - uses config_v2
│
├── experiments/
│   ├── README.md                          # Auto-updated
│   ├── EXPERIMENT_TEMPLATE.md             # Enhanced template
│   └── 001_*.md                           # Auto-updated
│
├── config.ini                             # DEPRECATED (keep for reference)
├── config.py                              # DEPRECATED (keep for reference)
└── requirements.txt                       # UPDATED (add pyyaml, jsonschema)
```

---

## Detailed Implementation Plan

### Phase 1: Core Infrastructure (3-4 hours)

#### 1.1 Config System (`src/config_v2.py`)

**File:** `src/config_v2.py` (~400 lines)

**Features:**
- Load YAML with inheritance
- Validate against schema
- Merge configs (base + experiment)
- Support for overrides
- Convert to dict for wandb
- Backward compat with config.py

**Key Classes:**
```python
@dataclass
class ExperimentMetadata:
    id: str
    name: str
    description: str
    hypothesis: str
    expected_results: dict
    parent_experiment: Optional[str]
    status: Literal["planning", "running", "complete", "failed"]

@dataclass
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    version: str
    # ... other fields

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment: ExperimentMetadata
    data: DataConfig
    model: ModelConfig
    baseline_training: Optional[BaselineTrainingConfig]
    triplet_training: Optional[TripletTrainingConfig]
    mining: Optional[MiningConfig]
    evaluation: EvaluationConfig
    wandb: WandbConfig
    git: GitConfig
    reproducibility: ReproducibilityConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load and validate config from YAML."""
        ...

    def validate(self) -> List[str]:
        """Validate config, return list of errors."""
        ...

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten for wandb logging."""
        ...
```

**Config Inheritance:**
```python
def load_config_with_inheritance(yaml_path: str) -> dict:
    """
    Load YAML config with inheritance.

    Supports:
    - base: "configs/base.yaml"
    - parent: "configs/experiments/001.yaml"
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Load base if specified
    if "base" in config:
        base = load_config_with_inheritance(config["base"])
        config = merge_dicts(base, config)

    # Load parent experiment if specified
    if "experiment" in config and "parent" in config["experiment"]:
        parent = load_config_with_inheritance(
            f"configs/experiments/{config['experiment']['parent']}.yaml"
        )
        config = merge_dicts(parent, config)

    return config
```

#### 1.2 Experiment Tracker (`src/experiment_tracker.py`)

**File:** `src/experiment_tracker.py` (~300 lines)

**Features:**
- Create experiment docs from template
- Update results section
- Update experiment log table
- Track experiment status
- Link to wandb runs

**Key Classes:**
```python
class ExperimentTracker:
    """Manages experiment documentation and status."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.exp_dir = Path("experiments")
        self.exp_doc_path = self._get_doc_path()
        self.log_path = self.exp_dir / "README.md"

    def create_experiment_doc(self, template_path: Optional[str] = None):
        """Create experiment doc from template."""
        ...

    def update_status(self, status: str):
        """Update experiment status in doc and log."""
        ...

    def log_start(self):
        """Log experiment start time and details."""
        ...

    def log_results(self, metrics: Dict[str, Any], wandb_url: str):
        """Update doc with results and wandb link."""
        ...

    def update_experiment_log(self):
        """Update experiments/README.md table."""
        ...

    def get_status(self) -> str:
        """Get current experiment status."""
        ...
```

**Markdown Operations:**
```python
def update_markdown_section(
    content: str,
    section_header: str,
    new_content: str
) -> str:
    """Replace a markdown section with new content."""
    ...

def update_markdown_table_row(
    content: str,
    exp_id: str,
    new_row: str
) -> str:
    """Update or insert a row in a markdown table."""
    ...

def generate_results_section(
    metrics: Dict[str, Any],
    wandb_url: str,
    output_dir: Path
) -> str:
    """Generate markdown for results section."""
    return f"""
### Metrics

| Metric | Value | Change | Target |
|--------|-------|--------|--------|
| EER | {metrics['eer']:.2%} | {metrics.get('eer_delta', 'N/A')} | {metrics['target_eer']:.2%} |
| ROC-AUC | {metrics['roc_auc']:.4f} | {metrics.get('auc_delta', 'N/A')} | {metrics['target_auc']:.4f} |

**Wandb Run:** [{wandb_url}]({wandb_url})

### Visualizations

- [ROC Curve]({output_dir}/roc_curve.png)
- [UMAP Visualization]({output_dir}/umap_visualization.png)
"""
```

#### 1.3 Wandb Integration (`src/wandb_utils.py`)

**File:** `src/wandb_utils.py` (~250 lines)

**Features:**
- Enhanced wandb init with full metadata
- Automatic artifact logging
- Config versioning
- Reproducibility tracking

**Key Functions:**
```python
def init_wandb_with_metadata(
    config: ExperimentConfig,
    git_info: dict,
    system_info: dict
) -> wandb.Run:
    """Initialize wandb with full experiment metadata."""

    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=f"exp-{config.experiment.id}-{config.experiment.name}",
        tags=[
            f"exp-{config.experiment.id}",
            *config.wandb.tags,
        ],
        group=config.wandb.get("group"),
        notes=config.experiment.hypothesis,
        config={
            # Experiment metadata
            "experiment": asdict(config.experiment),

            # Reproducibility
            "git": git_info,
            "system": system_info,
            "data_version": config.data.version,
            "random_seed": config.reproducibility.random_seed,

            # All params (flattened)
            **config.to_flat_dict(),
        }
    )

    # Log config as artifact
    log_config_artifact(run, config)

    return run

def log_config_artifact(run: wandb.Run, config: ExperimentConfig):
    """Log config file as wandb artifact."""
    artifact = wandb.Artifact(
        f"config-exp-{config.experiment.id}",
        type="config",
        description=f"Configuration for experiment {config.experiment.id}",
        metadata={
            "experiment_id": config.experiment.id,
            "experiment_name": config.experiment.name,
        }
    )

    # Add config YAML
    artifact.add_file(config._yaml_path)

    # Add experiment doc if exists
    doc_path = Path(f"experiments/{config.experiment.id}_{config.experiment.name}.md")
    if doc_path.exists():
        artifact.add_file(str(doc_path))

    run.log_artifact(artifact)

def log_evaluation_artifacts(
    run: wandb.Run,
    output_dir: Path,
    metrics: dict
):
    """Log evaluation results as artifacts."""
    artifact = wandb.Artifact(
        f"evaluation-exp-{metrics['experiment_id']}",
        type="evaluation",
        description="Evaluation results and visualizations"
    )

    # Add plots
    for plot in ["roc_curve.png", "far_frr_curves.png", "umap_visualization.png"]:
        plot_path = output_dir / plot
        if plot_path.exists():
            artifact.add_file(str(plot_path))

    # Add metrics JSON
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        artifact.add_file(str(metrics_path))

    run.log_artifact(artifact)
```

#### 1.4 Git Utils (`src/git_utils.py`)

**File:** `src/git_utils.py` (~100 lines)

```python
import subprocess
from pathlib import Path
from typing import Optional, Dict

def get_git_info() -> Dict[str, any]:
    """Get comprehensive git information."""
    try:
        return {
            "hash": get_git_hash(),
            "hash_short": get_git_hash()[:7],
            "branch": get_git_branch(),
            "remote": get_git_remote(),
            "dirty": has_uncommitted_changes(),
            "untracked": has_untracked_files(),
            "tags": get_git_tags(),
        }
    except Exception as e:
        return {"error": str(e)}

def get_git_hash() -> str:
    """Get current git commit hash."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()

def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes."""
    result = subprocess.call(
        ["git", "diff-index", "--quiet", "HEAD"]
    )
    return result != 0

def require_clean_git(strict: bool = True):
    """Require clean git state before experiment."""
    if has_uncommitted_changes():
        if strict:
            raise RuntimeError(
                "Git has uncommitted changes. "
                "Commit or stash before running experiment."
            )
        else:
            print("⚠️  Warning: Git has uncommitted changes")
```

#### 1.5 Reproducibility Utils (`src/reproducibility.py`)

**File:** `src/reproducibility.py` (~150 lines)

```python
import random
import numpy as np
import torch
import platform
from typing import Dict

def set_random_seeds(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for CUBLAS
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_system_info() -> Dict[str, any]:
    """Get comprehensive system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

def log_environment():
    """Log complete environment information."""
    info = get_system_info()
    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 80)
```

---

### Phase 2: Config Schema & Base Configs (1-2 hours)

#### 2.1 Create Config Schema (`configs/schema.yaml`)

**File:** `configs/schema.yaml` (~200 lines)

JSON Schema for config validation:

```yaml
$schema: http://json-schema.org/draft-07/schema#
title: Experiment Configuration
type: object
required:
  - experiment
  - data
  - model
  - evaluation
  - wandb

properties:
  experiment:
    type: object
    required: [id, name, description, hypothesis, status]
    properties:
      id:
        type: string
        pattern: "^[0-9]{3}$"
        description: "3-digit experiment ID (e.g., '001')"
      name:
        type: string
        pattern: "^[a-z0-9_]+$"
        description: "Snake_case experiment name"
      description:
        type: string
        minLength: 10
      hypothesis:
        type: string
        minLength: 20
      expected_results:
        type: object
      parent_experiment:
        type: [string, "null"]
      status:
        enum: [planning, running, complete, failed]

  data:
    type: object
    required: [train_path, val_path, test_path, version]
    properties:
      train_path:
        type: string
      val_path:
        type: string
      test_path:
        type: string
      version:
        type: string
        pattern: "^v[0-9]+\\.[0-9]+$"

  # ... more schema definitions
```

#### 2.2 Create Base Config (`configs/base.yaml`)

**File:** `configs/base.yaml` (~150 lines)

Default values for all experiments:

```yaml
# Base configuration for all experiments
# Individual experiments inherit and override these values

data:
  train_path: "data/processed/train.parquet"
  val_path: "data/processed/val.parquet"
  test_path: "data/processed/test.parquet"
  version: "v1.0"
  min_tokens: 20
  max_tokens: 512

model:
  base_model: "roberta-base"
  max_seq_length: 512
  embedding_dim: 768
  pooling: "mean"

baseline_training:
  batch_size: 64
  num_epochs: 1
  learning_rate: 2.0e-5
  warmup_steps: 1000
  fp16: true
  checkpoint_save_steps: 5000
  loss:
    type: "MultipleNegativesRankingLoss"
    scale: 100.0
    temperature: 0.01

triplet_training:
  batch_size: 32
  num_epochs: 1
  learning_rate: 1.0e-5
  margin: 0.5
  fp16: true
  warmup_ratio: 0.1

mining:
  sample_size: 50000
  k_neighbors: 10
  batch_size: 128
  prioritize_same_channel: true
  min_similarity: 0.7
  max_similarity: 0.95

loop:
  num_iterations: 3

evaluation:
  use_whitening: true
  num_positive_pairs: 2000
  num_negative_pairs: 2000
  target_eer: 0.15
  target_roc_auc: 0.95

visualization:
  umap_authors: 50
  samples_per_author: 10

wandb:
  enabled: true
  project: "authorship-verification"
  entity: null  # Set to your wandb username/team
  log_model: true
  log_interval: 100

git:
  require_clean: false
  auto_commit_results: false

reproducibility:
  random_seed: 42
  deterministic: true
  log_system_info: true

hardware:
  vram_gb: 24
  use_gpu: true
  num_workers: 4
```

#### 2.3 Create Preset Configs

**File:** `configs/presets/quick_test.yaml` (~50 lines)

```yaml
base: "../base.yaml"

# Quick testing preset
# Reduces all sizes for fast iteration

baseline_training:
  batch_size: 16
  num_epochs: 1
  checkpoint_save_steps: 100

mining:
  sample_size: 1000
  k_neighbors: 5

evaluation:
  num_positive_pairs: 100
  num_negative_pairs: 100

wandb:
  enabled: false  # Disable for quick tests
```

**File:** `configs/presets/debug.yaml`

```yaml
base: "../base.yaml"

# Debug preset
# Minimal config for debugging

baseline_training:
  batch_size: 4
  num_epochs: 1

mining:
  sample_size: 100

evaluation:
  num_positive_pairs: 20
  num_negative_pairs: 20

reproducibility:
  deterministic: true
  log_system_info: true

wandb:
  enabled: false
```

---

### Phase 3: Script Migration (3-4 hours)

#### 3.1 Update Training Scripts

Each script gets similar changes. Example for `train_baseline.py`:

**Changes:**

1. **Imports:**
```python
# OLD
import argparse
from config import get_baseline_config

# NEW
from src.config_v2 import ExperimentConfig
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata, log_evaluation_artifacts
from src.git_utils import get_git_info, require_clean_git
from src.reproducibility import set_random_seeds, get_system_info
```

2. **Main function:**
```python
# OLD
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    # ... 15 more args
    args = parser.parse_args()

    trainer = BaselineTrainer(args.model, args.train_data)
    trainer.train(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        # ... pass all args
    )

# NEW
def main():
    parser = argparse.ArgumentParser(
        description="Train baseline model from YAML config"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        help="Config overrides (key=value)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config, don't train"
    )
    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Apply overrides
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    # Validate
    errors = config.validate()
    if errors:
        print("❌ Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    if args.validate_only:
        print("✅ Config is valid")
        return

    # Check git state
    if config.git.require_clean:
        require_clean_git(strict=True)

    # Set random seeds
    set_random_seeds(
        config.reproducibility.random_seed,
        config.reproducibility.deterministic
    )

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)
    tracker.create_experiment_doc()
    tracker.update_status("running")
    tracker.log_start()

    # Get metadata
    git_info = get_git_info()
    system_info = get_system_info()

    # Initialize wandb
    wandb_run = None
    if config.wandb.enabled:
        wandb_run = init_wandb_with_metadata(config, git_info, system_info)

    # Train
    trainer = BaselineTrainer(config, wandb_run)
    trainer.train()

    # Evaluate
    evaluator = Evaluator(config, wandb_run)
    metrics = evaluator.evaluate()

    # Update experiment doc with results
    if wandb_run:
        tracker.log_results(metrics, wandb_run.url)
        log_evaluation_artifacts(wandb_run, Path(config.evaluation.output_dir), metrics)
        wandb_run.finish()

    tracker.update_status("complete")

    print(f"\n✅ Experiment {config.experiment.id} complete!")
    print(f"📊 Results: experiments/{config.experiment.id}_{config.experiment.name}.md")
    if wandb_run:
        print(f"🔗 Wandb: {wandb_run.url}")
```

3. **Trainer class:**
```python
# OLD
class BaselineTrainer:
    def __init__(self, model_name: str, train_data: str):
        self.model_name = model_name
        self.train_data = train_data
        self.wandb_config = {}  # Minimal

    def train(
        self,
        train_path: str,
        val_path: str,
        output_dir: str,
        batch_size: int,
        num_epochs: int,
        # ... 10 more params
    ):
        # Training logic

# NEW
class BaselineTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        wandb_run: Optional[wandb.Run] = None
    ):
        self.config = config
        self.wandb_run = wandb_run

        # All params from config
        self.train_path = config.data.train_path
        self.batch_size = config.baseline_training.batch_size
        # ... all params from config

    def train(self):
        """Train with config settings."""
        # Training logic - all params from self.config

        # Log to wandb automatically
        if self.wandb_run:
            self.wandb_run.log({"loss": loss})
```

**Similar changes for:**
- `train_triplet.py`
- `run_loop.py`
- `evaluate.py`
- `miner.py`

---

### Phase 4: Helper Scripts (2-3 hours)

#### 4.1 Create New Experiment (`scripts/new_experiment.py`)

**File:** `scripts/new_experiment.py` (~200 lines)

```python
#!/usr/bin/env python3
"""
Create a new experiment with config and documentation.

Usage:
    python scripts/new_experiment.py --id 003 --name distance_filtering
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

def main():
    parser = argparse.ArgumentParser(
        description="Create new experiment config and documentation"
    )
    parser.add_argument("--id", required=True, help="3-digit experiment ID")
    parser.add_argument("--name", required=True, help="Experiment name (snake_case)")
    parser.add_argument("--description", help="Experiment description")
    parser.add_argument("--hypothesis", help="Experiment hypothesis")
    parser.add_argument("--parent", help="Parent experiment ID")
    parser.add_argument("--preset", help="Preset to base on (quick_test, debug, full_training)")
    args = parser.parse_args()

    # Validate ID
    if not args.id.isdigit() or len(args.id) != 3:
        print("❌ ID must be 3 digits (e.g., '003')")
        sys.exit(1)

    # Check if experiment exists
    config_path = Path(f"configs/experiments/{args.id}_{args.name}.yaml")
    if config_path.exists():
        print(f"❌ Experiment {args.id} already exists")
        sys.exit(1)

    # Create config
    config = create_config(args)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"✅ Created config: {config_path}")

    # Create experiment doc
    doc_path = Path(f"experiments/{args.id}_{args.name}.md")
    doc_content = create_experiment_doc(args, config_path)
    with open(doc_path, "w") as f:
        f.write(doc_content)

    print(f"✅ Created doc: {doc_path}")

    # Update experiment log
    update_experiment_log(args)

    print(f"\n✅ Experiment {args.id} ready!")
    print(f"\nNext steps:")
    print(f"1. Edit config: vim {config_path}")
    print(f"2. Edit doc hypothesis: vim {doc_path}")
    print(f"3. Run: python train_baseline.py --config {config_path}")

def create_config(args) -> dict:
    """Create experiment config dict."""
    config = {
        "base": "../base.yaml",
        "experiment": {
            "id": args.id,
            "name": args.name,
            "description": args.description or f"Experiment {args.id}: {args.name}",
            "hypothesis": args.hypothesis or "TODO: Fill in hypothesis",
            "expected_results": {
                "eer": "TODO",
                "roc_auc": "TODO",
            },
            "parent_experiment": args.parent,
            "status": "planning",
        },
        # Start with base, user can override
    }

    return config
```

#### 4.2 Validate Config (`scripts/validate_config.py`)

**File:** `scripts/validate_config.py` (~150 lines)

```python
#!/usr/bin/env python3
"""
Validate experiment config against schema.

Usage:
    python scripts/validate_config.py configs/experiments/002_lower_temp.yaml
"""

import sys
from pathlib import Path
from src.config_v2 import ExperimentConfig
from src.git_utils import get_git_info

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Validating: {config_path}")
    print("=" * 80)

    # Load config
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Validate
    errors = config.validate()
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("✅ Config is valid")
    print()

    # Check paths
    print("Checking file paths...")
    path_checks = [
        ("Train data", config.data.train_path),
        ("Val data", config.data.val_path),
        ("Test data", config.data.test_path),
    ]

    for name, path in path_checks:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name}: {path} (not found)")

    # Check git
    print("\nChecking git status...")
    git_info = get_git_info()
    print(f"  Branch: {git_info['branch']}")
    print(f"  Commit: {git_info['hash_short']}")

    if git_info['dirty']:
        print(f"  ⚠️  Uncommitted changes")
        if config.git.require_clean:
            print(f"  ❌ Config requires clean git (git.require_clean: true)")
            sys.exit(1)
    else:
        print(f"  ✅ Clean git state")

    # Summary
    print("\n" + "=" * 80)
    print(f"✅ Config ready to use!")
    print(f"\nRun:")
    print(f"  python train_baseline.py --config {config_path}")
```

#### 4.3 Experiment Status (`scripts/experiment_status.py`)

**File:** `scripts/experiment_status.py` (~150 lines)

```python
#!/usr/bin/env python3
"""
Show status of all experiments.

Usage:
    python scripts/experiment_status.py
    python scripts/experiment_status.py --detail
"""

import argparse
from pathlib import Path
from src.config_v2 import ExperimentConfig
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    # Find all experiment configs
    configs_dir = Path("configs/experiments")
    config_files = sorted(configs_dir.glob("*.yaml"))

    experiments = []
    for config_file in config_files:
        try:
            config = ExperimentConfig.from_yaml(str(config_file))

            # Get metrics if available
            metrics = get_experiment_metrics(config)

            experiments.append({
                "ID": config.experiment.id,
                "Name": config.experiment.name,
                "Status": format_status(config.experiment.status),
                "EER": metrics.get("eer", "-"),
                "ROC-AUC": metrics.get("roc_auc", "-"),
                "Wandb": "✅" if has_wandb_link(config) else "-",
            })
        except Exception as e:
            print(f"⚠️  Failed to load {config_file}: {e}")

    # Print table
    print(tabulate(experiments, headers="keys", tablefmt="github"))

    # Detailed info if requested
    if args.detail:
        print("\n" + "=" * 80)
        print("Detailed Information")
        print("=" * 80)
        for exp in experiments:
            print(f"\nExperiment {exp['ID']}:")
            # ... print detailed info
```

#### 4.4 Compare Experiments (`scripts/compare_experiments.py`)

**File:** `scripts/compare_experiments.py` (~200 lines)

```python
#!/usr/bin/env python3
"""
Compare two or more experiments.

Usage:
    python scripts/compare_experiments.py 001 002
    python scripts/compare_experiments.py 001 002 003 --metrics eer roc_auc
"""

import argparse
from pathlib import Path
from src.config_v2 import ExperimentConfig
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_ids", nargs="+", help="Experiment IDs to compare")
    parser.add_argument("--metrics", nargs="*", help="Metrics to compare")
    args = parser.parse_args()

    # Load experiments
    experiments = []
    for exp_id in args.exp_ids:
        config_file = find_experiment_config(exp_id)
        if not config_file:
            print(f"❌ Experiment {exp_id} not found")
            continue

        config = ExperimentConfig.from_yaml(str(config_file))
        metrics = get_experiment_metrics(config)

        experiments.append({
            "config": config,
            "metrics": metrics,
        })

    # Compare configs
    print("Configuration Differences")
    print("=" * 80)
    compare_configs(experiments)

    # Compare metrics
    print("\nMetrics Comparison")
    print("=" * 80)
    compare_metrics(experiments, args.metrics)
```

#### 4.5 Migrate Legacy Experiments (`scripts/backfill_experiments.py`)

**File:** `scripts/backfill_experiments.py` (~250 lines)

```python
#!/usr/bin/env python3
"""
Backfill existing experiments with configs and proper structure.

Usage:
    python scripts/backfill_experiments.py
"""

from pathlib import Path
import yaml

def main():
    print("Migrating existing experiments...")
    print("=" * 80)

    # Find existing experiment docs
    exp_dir = Path("experiments")
    exp_docs = sorted(exp_dir.glob("0*_*.md"))

    for doc in exp_docs:
        print(f"\nProcessing: {doc.name}")

        # Parse experiment ID and name
        exp_id, exp_name = parse_doc_filename(doc.name)

        # Create config if doesn't exist
        config_path = Path(f"configs/experiments/{exp_id}_{exp_name}.yaml")
        if not config_path.exists():
            config = create_config_from_doc(doc)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            print(f"  ✅ Created config: {config_path}")
        else:
            print(f"  ⏭️  Config exists: {config_path}")

        # Update doc with wandb link if missing
        doc_content = doc.read_text()
        if "Wandb Run:" not in doc_content:
            # Try to find wandb run
            wandb_url = find_wandb_run(exp_id)
            if wandb_url:
                doc_content = add_wandb_link(doc_content, wandb_url)
                doc.write_text(doc_content)
                print(f"  ✅ Added wandb link")

    print("\n" + "=" * 80)
    print("✅ Migration complete!")
```

---

### Phase 5: Testing (1-2 hours)

#### 5.1 Unit Tests

**File:** `tests/test_config_v2.py`

```python
import pytest
from src.config_v2 import ExperimentConfig

def test_load_base_config():
    """Test loading base config."""
    config = ExperimentConfig.from_yaml("configs/base.yaml")
    assert config.model.base_model == "roberta-base"

def test_config_inheritance():
    """Test config inheritance."""
    config = ExperimentConfig.from_yaml("configs/experiments/002_lower_temp.yaml")
    # Should inherit from base
    assert config.model.base_model == "roberta-base"
    # Should override
    assert config.baseline_training.loss.scale == 100.0

def test_config_validation():
    """Test config validation."""
    # Invalid experiment ID
    config = create_invalid_config(id="1")  # Should be 3 digits
    errors = config.validate()
    assert len(errors) > 0
```

**File:** `tests/test_experiment_tracker.py`

```python
import pytest
from src.experiment_tracker import ExperimentTracker
from src.config_v2 import ExperimentConfig

def test_create_experiment_doc(tmp_path):
    """Test experiment doc creation."""
    config = ExperimentConfig.from_yaml("configs/experiments/002_lower_temp.yaml")
    tracker = ExperimentTracker(config)

    # Create doc
    tracker.create_experiment_doc()

    # Check it exists
    assert tracker.exp_doc_path.exists()

    # Check content
    content = tracker.exp_doc_path.read_text()
    assert config.experiment.hypothesis in content
```

#### 5.2 Integration Tests

**File:** `tests/test_end_to_end.py`

```python
def test_full_experiment_flow(tmp_path):
    """Test complete experiment flow."""
    # 1. Create experiment
    subprocess.run([
        "python", "scripts/new_experiment.py",
        "--id", "999",
        "--name", "test_experiment",
    ], check=True)

    # 2. Validate config
    subprocess.run([
        "python", "scripts/validate_config.py",
        "configs/experiments/999_test_experiment.yaml"
    ], check=True)

    # 3. Run training (with test preset)
    subprocess.run([
        "python", "train_baseline.py",
        "--config", "configs/presets/quick_test.yaml",
        "--overrides", "experiment.id=999"
    ], check=True)

    # 4. Check experiment doc was updated
    doc = Path("experiments/999_test_experiment.md")
    assert doc.exists()
    content = doc.read_text()
    assert "## Results" in content
```

---

## Migration Strategy

### Step 1: Backward Compatibility (Day 1)

1. **Keep old files:** Don't delete `config.py` and `config.ini` yet
2. **Add new imports:** Scripts import both old and new config
3. **Fallback logic:** Try new config, fall back to old if not found

```python
# In train_baseline.py
try:
    from src.config_v2 import ExperimentConfig
    USE_NEW_CONFIG = True
except ImportError:
    from config import get_baseline_config
    USE_NEW_CONFIG = False

def main():
    if USE_NEW_CONFIG and "--config" in sys.argv:
        # Use new system
        ...
    else:
        # Use old system
        ...
```

### Step 2: Parallel Operation (Days 2-3)

1. **Test new system:** Run experiments with new configs
2. **Compare results:** Ensure identical behavior
3. **Fix bugs:** Address any issues

### Step 3: Full Migration (Days 4-5)

1. **Backfill experiments:** Migrate existing experiments
2. **Update documentation:** Update README with new workflow
3. **Remove old code:** Delete `config.py` and `config.ini`

### Step 4: Cleanup (Day 6)

1. **Final testing:** Run full test suite
2. **Documentation:** Ensure all docs updated
3. **Commit:** Create migration commit

---

## Rollout Plan

### Week 1: Infrastructure

**Days 1-2:**
- [ ] Implement Phase 1 (Core Infrastructure)
- [ ] Create unit tests
- [ ] Test locally

**Days 3-4:**
- [ ] Implement Phase 2 (Configs & Schema)
- [ ] Create example experiment configs
- [ ] Test config loading

**Days 5-7:**
- [ ] Implement Phase 3 (Script Migration)
- [ ] Test one script end-to-end
- [ ] Fix any issues

### Week 2: Helper Scripts & Testing

**Days 8-10:**
- [ ] Implement Phase 4 (Helper Scripts)
- [ ] Create integration tests
- [ ] Test full workflow

**Days 11-12:**
- [ ] Backfill existing experiments
- [ ] Update documentation
- [ ] Final testing

**Days 13-14:**
- [ ] Buffer for unexpected issues
- [ ] Polish and cleanup
- [ ] Final review

---

## Testing Strategy

### Unit Tests

Test individual components:
- Config loading
- Validation
- Experiment tracker
- Wandb utils
- Git utils

### Integration Tests

Test complete workflows:
- Create experiment → validate → train → evaluate
- Config inheritance
- Artifact logging
- Doc updates

### Manual Testing

Test real experiments:
1. Create new experiment
2. Run training
3. Verify all automation works
4. Check wandb integration
5. Verify doc updates

### Regression Testing

Ensure backward compatibility:
- Old configs still work (during transition)
- Results match previous runs
- No performance regression

---

## Clarifying Questions

Before I start implementation, please answer these questions:

### 1. Wandb Configuration

**Q:** What is your wandb entity (username/team)?
- This will be set in `configs/base.yaml`
- Example: `entity: "your-username"`

**Q:** Do you want wandb enabled by default?
- Currently planned: `enabled: true`
- Can be overridden per experiment

### 2. Git Requirements

**Q:** Should experiments require clean git state before running?
- Option A: Yes, always require clean git (strict reproducibility)
- Option B: No, just warn if dirty (flexible for iteration)
- Currently planned: Option B (`require_clean: false`)

**Q:** Should results be auto-committed?
- After experiment completes, auto-commit:
  - Updated experiment doc
  - Updated experiment log
  - Config used
- Currently planned: No (`auto_commit_results: false`)

### 3. Experiment Numbering

**Q:** How should experiment IDs be assigned?
- Option A: Manual (user specifies when creating)
- Option B: Auto-increment (find next available ID)
- Currently planned: Option A

### 4. Config Overrides

**Q:** Should CLI overrides be allowed?
- Useful for debugging: `--config exp.yaml --overrides batch_size=16`
- But could break reproducibility
- Currently planned: Yes, with warning logged

### 5. Backward Compatibility

**Q:** How long should old config.ini be supported?
- Option A: Keep forever (safest)
- Option B: 1-2 releases then deprecate
- Option C: Immediate removal after migration
- Currently planned: Option B

### 6. Experiment Documentation

**Q:** What template style do you prefer?
- Current template is detailed (good for research)
- Do you want to keep it or simplify?

**Q:** Should hypothesis/expected results be required?
- Forces good practice
- But might slow iteration
- Currently planned: Required

### 7. Evaluation Integration

**Q:** Should evaluation always run after training?
- Option A: Always evaluate (convenient)
- Option B: Separate step (more control)
- Currently planned: Option A

### 8. Data Versioning

**Q:** Do you want to track data versions?
- Adds complexity but improves reproducibility
- Could integrate with DVC later
- Currently planned: Simple version string (`v1.0`)

### 9. Testing Approach

**Q:** How thorough should testing be?
- Option A: Minimal (smoke tests only) - faster
- Option B: Comprehensive (unit + integration) - safer
- Currently planned: Option B

### 10. Migration Priority

**Q:** Which scripts to migrate first?
- Option A: All at once (faster overall, but riskier)
- Option B: One at a time (safer, slower)
- Currently planned: One at a time
  1. `train_baseline.py` (most important)
  2. `evaluate.py` (goes with baseline)
  3. `miner.py` + `train_triplet.py` + `run_loop.py`

---

## Success Criteria

Migration is complete when:

1. ✅ All scripts use YAML configs
2. ✅ Experiments auto-generate docs
3. ✅ Wandb has full metadata
4. ✅ Can reproduce any run from config alone
5. ✅ Helper scripts work correctly
6. ✅ Tests pass
7. ✅ Documentation updated
8. ✅ Old config.py removed

---

## Timeline Summary

| Phase | Duration | Effort |
|-------|----------|--------|
| Phase 1: Core Infrastructure | 3-4 hours | High |
| Phase 2: Configs & Schema | 1-2 hours | Medium |
| Phase 3: Script Migration | 3-4 hours | High |
| Phase 4: Helper Scripts | 2-3 hours | Medium |
| Phase 5: Testing | 1-2 hours | Medium |
| **Total** | **10-15 hours** | **High** |

**Realistic timeline:** 2-3 focused work sessions over 1-2 weeks.

---

## Next Steps

Once you've answered the clarifying questions:

1. I'll start with Phase 1 (Core Infrastructure)
2. We'll test it on `train_baseline.py`
3. Iterate based on feedback
4. Roll out to remaining scripts
5. Create helper scripts
6. Backfill existing experiments
7. Final testing and cleanup

**Ready to proceed?** Answer the questions above and I'll begin implementation!
