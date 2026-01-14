# Wandb Integration Audit

## Executive Summary

**Current Status:** ⚠️ **Partial Integration**

Wandb is logging basic metrics but **NOT** connected to the experiment tracking system. Missing critical metadata for reproducibility and experiment management.

## What's Working ✅

### 1. Basic Logging
- ✅ Training metrics logged
- ✅ Hyperparameters logged
- ✅ Tags for organization ("baseline", "phase2", etc.)
- ✅ Project/entity configuration
- ✅ Environment variable overrides

### 2. Current Logged Data

**train_baseline.py:**
```python
config={
    "model_name": embedding_dim,
    "batch_size": 64,
    "num_epochs": 1,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "fp16": True,
    "loss": "MultipleNegativesRankingLoss",
    "loss_scale": 100.0,
    "loss_temperature": 0.01,
    "phase": "baseline_training",
}
```

**Tags:** `["baseline", "phase2"]`

## What's Missing ❌

### 1. Experiment Tracking Integration

**Problem:** Wandb runs are NOT linked to experiment documentation.

**Missing:**
- ❌ Experiment ID (e.g., "001", "002")
- ❌ Experiment name (e.g., "dimensional_collapse_fix")
- ❌ Hypothesis (what are we testing?)
- ❌ Expected outcomes
- ❌ Reference to experiment markdown doc

**Impact:**
- Can't easily find which wandb run corresponds to which experiment
- Can't track experiment lineage (parent/child experiments)
- Manual work needed to cross-reference runs and docs

### 2. Reproducibility Metadata

**Problem:** Can't fully reproduce runs from wandb alone.

**Missing:**
- ❌ Git commit hash (which code version?)
- ❌ Git dirty flag (were there uncommitted changes?)
- ❌ Data version (which dataset split?)
- ❌ Config file path (which config was used?)
- ❌ Random seed
- ❌ Python/CUDA/library versions

**Impact:**
- Can't reproduce results exactly
- Hard to debug "why did this work last time?"
- No audit trail for publications

### 3. Data Provenance

**Problem:** Don't know which data was used.

**Missing:**
- ❌ Training data path
- ❌ Training data statistics (# samples, # authors)
- ❌ Data preprocessing version
- ❌ Data hashes/checksums

**Impact:**
- Can't verify if results are due to model changes or data changes
- Hard to track down data quality issues

### 4. Model Lineage

**Problem:** Don't track model relationships.

**Missing:**
- ❌ Parent model (what did we fine-tune from?)
- ❌ Model architecture details
- ❌ Which checkpoint was used for initialization

**Impact:**
- Can't track iterative improvements
- Hard to understand autonomous loop progression

### 5. Experiment Documentation

**Problem:** Experiment docs NOT auto-updated from wandb.

**Missing:**
- ❌ Automatic results section population
- ❌ Wandb run URL in experiment markdown
- ❌ Experiment log table auto-update
- ❌ Comparison to previous experiments

**Impact:**
- Manual work to update docs after each run
- Easy to forget to document results
- Docs become stale/inaccurate

### 6. Artifacts

**Problem:** Not logging important artifacts.

**Missing:**
- ❌ Config file as artifact
- ❌ Experiment doc as artifact
- ❌ Best model checkpoint
- ❌ Evaluation plots (ROC, UMAP, etc.)
- ❌ Training curves

**Impact:**
- Can't download exact config that produced results
- Have to search filesystem for plots
- No centralized artifact storage

## Comparison: Current vs Ideal

### Current Workflow

```
1. User edits config.ini
2. User runs: python train_baseline.py --train-data ... --batch-size ... (20+ args)
3. Wandb logs basic metrics
4. Training completes
5. User manually evaluates
6. User manually creates experiment doc (experiments/00X_name.md)
7. User manually fills in results
8. User manually updates experiments/README.md log table
9. User manually adds wandb URL to doc (if they remember)
```

**Problems:**
- 9 manual steps
- Easy to forget steps 6-9
- No automatic linking
- Prone to inconsistencies

### Ideal Workflow

```
1. User creates experiment config: configs/experiments/003_name.yaml
2. User runs: python train_baseline.py --config configs/experiments/003_name.yaml
3. System automatically:
   - Creates experiment doc from template
   - Initializes wandb with full metadata
   - Links wandb run to experiment doc
   - Trains model
   - Evaluates model
   - Updates experiment doc with results
   - Updates experiments/README.md log table
   - Logs all artifacts to wandb
```

**Benefits:**
- 2 manual steps (create config, run script)
- Everything else automatic
- No forgotten steps
- Perfect consistency

## Example: What Should Be Logged

### Full Wandb Config

```python
wandb.init(
    project="authorship-verification",
    entity="your-entity",
    name="exp-002-lower_temperature_training",
    tags=["exp-002", "lower-temp", "baseline", "phase2"],
    group="temperature-experiments",
    notes="Testing scale=100.0 (temp=0.01) to improve decision boundaries",
    config={
        # === Experiment Metadata ===
        "experiment_id": "002",
        "experiment_name": "lower_temperature_training",
        "experiment_doc": "experiments/002_lower_temperature_training.md",
        "hypothesis": "Lower temperature will improve EER by 3-5 percentage points",
        "expected_eer": "14-17%",
        "expected_roc_auc": ">0.90",
        "parent_experiment": "001",
        "status": "running",

        # === Reproducibility ===
        "git_hash": "abc123def456",
        "git_branch": "main",
        "git_dirty": False,
        "git_remote": "https://github.com/user/fingertest",
        "python_version": "3.11.5",
        "cuda_version": "12.1",
        "pytorch_version": "2.1.0",
        "random_seed": 42,

        # === Data ===
        "data_train_path": "data/processed/train.parquet",
        "data_val_path": "data/processed/val.parquet",
        "data_test_path": "data/processed/test.parquet",
        "data_version": "v1.0",
        "num_train_samples": 574614,
        "num_val_samples": 63846,
        "num_test_samples": 3460,
        "num_train_authors": 8234,
        "num_test_authors": 692,

        # === Model ===
        "model_base": "roberta-base",
        "model_parent": "models/baseline/checkpoint-5000",
        "embedding_dim": 768,
        "max_seq_length": 512,
        "pooling": "mean",

        # === Training ===
        "batch_size": 64,
        "num_epochs": 1,
        "learning_rate": 2e-5,
        "warmup_steps": 1000,
        "fp16": True,
        "loss_type": "MultipleNegativesRankingLoss",
        "loss_scale": 100.0,
        "loss_temperature": 0.01,

        # === Evaluation ===
        "eval_use_whitening": True,
        "eval_num_positive_pairs": 2000,
        "eval_num_negative_pairs": 2000,

        # === Hardware ===
        "gpu_type": "RTX 3090",
        "vram_gb": 24,
        "num_gpus": 1,

        # === Config ===
        "config_file": "configs/experiments/002_lower_temp.yaml",
    }
)

# Log artifacts
wandb.save("configs/experiments/002_lower_temp.yaml")  # Exact config
wandb.save("experiments/002_lower_temperature_training.md")  # Experiment doc

# After evaluation
wandb.log({
    "final_eer": 0.1489,
    "final_roc_auc": 0.9123,
    "final_threshold": 0.2045,
})

# Log plots as artifacts
artifact = wandb.Artifact("evaluation-plots", type="plots")
artifact.add_file("outputs/evaluation/roc_curve.png")
artifact.add_file("outputs/evaluation/umap_visualization.png")
run.log_artifact(artifact)
```

## Recommendations

### Immediate (Quick Wins - 30 mins)

1. **Add experiment ID to wandb runs:**
   ```python
   # In train_baseline.py
   config["experiment_id"] = os.getenv("EXPERIMENT_ID", "unknown")
   tags.append(f"exp-{os.getenv('EXPERIMENT_ID', 'unknown')}")
   ```

   Usage:
   ```bash
   EXPERIMENT_ID=002 python train_baseline.py ...
   ```

2. **Add git hash logging:**
   ```python
   import subprocess
   config["git_hash"] = subprocess.check_output(
       ["git", "rev-parse", "HEAD"]
   ).decode().strip()
   ```

3. **Log config file:**
   ```python
   if config_file_path:
       wandb.save(config_file_path)
   ```

### Short-term (Full Integration - 6-8 hours)

Implement the full plan in `UNIFIED_CONFIG_PLAN.md`:
- YAML-based config system
- Automatic experiment doc generation
- Full metadata logging
- Bidirectional linking

### Long-term (Advanced Features)

1. **Experiment comparison dashboard** (wandb reports)
2. **Automatic ablation studies** (vary config systematically)
3. **Experiment recommendations** (ML-powered suggestions)
4. **Data versioning** (DVC integration)

## Action Items

- [ ] Review `UNIFIED_CONFIG_PLAN.md`
- [ ] Decide: Quick wins now, or full integration?
- [ ] If full integration: Start with Phase 1 (config system)
- [ ] Test on one script before rolling out
- [ ] Document new workflow for future experiments

## Current Gap Analysis

| Feature | Current | Needed | Priority | Effort |
|---------|---------|--------|----------|--------|
| Experiment ID tracking | ❌ | ✅ | HIGH | Low |
| Git hash logging | ❌ | ✅ | HIGH | Low |
| Data provenance | ❌ | ✅ | HIGH | Medium |
| Config artifacts | ❌ | ✅ | HIGH | Low |
| Auto doc updates | ❌ | ✅ | MEDIUM | High |
| Model lineage | ❌ | ✅ | MEDIUM | Medium |
| Full reproducibility | ❌ | ✅ | HIGH | Medium |
| Artifact logging | ❌ | ✅ | LOW | Low |

**Priority scoring:**
- HIGH: Blocking reproducibility or experiment tracking
- MEDIUM: Nice to have, improves workflow
- LOW: Advanced features

**Effort scoring:**
- Low: <1 hour
- Medium: 1-3 hours
- High: 3+ hours
