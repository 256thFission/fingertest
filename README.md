# Autonomous Authorship Verification System

A research project implementing metric learning for Discord message authorship verification using active hard-negative mining and dimensional collapse mitigation.

## Current Status

**Latest Results (2026-01-14):**
- **EER:** 17.35% (50% improvement from 34.8% baseline)
- **ROC-AUC:** 0.9010
- **Method:** Whitening (mean-subtraction post-processing)
- **Target:** EER <15%, ROC-AUC >0.95
- **Gap:** 2.35 percentage points

See [experiments/](experiments/) for detailed experiment tracking and results.

## Current Status (2026-01-15)

**System:**  **YAML-Based Config System - FULLY MIGRATED!**

**What Works:**
-  Create experiments with auto-incremented IDs
-  YAML configuration with inheritance
-  All training scripts migrated (baseline, triplet, miner, loop, evaluate)
-  Automatic experiment tracking and documentation
-  Enhanced wandb integration with full metadata
-  Complete reproducibility (git hash, system info, data version)

**All Scripts Migrated:**
-  `train_baseline.py` - YAML config system
-  `evaluate.py` - YAML config system
-  `train_triplet.py` - YAML config system
-  `miner.py` - YAML config system
-  `run_loop.py` - YAML config system

**See:** [`docs/SETUP_AND_USAGE.md`](docs/SETUP_AND_USAGE.md) for complete usage guide

---

## Quick Start

### 1. Setup

```bash
# Clone and install dependencies
git clone <repo-url>
cd fingertest
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Login to wandb (optional but recommended)
wandb login
```

### 2. Prepare Data

You have two options depending on your data format:

**Option A: Existing Parquet files** (fastest)
```bash
python preprocess_parquet.py --skip-channel-mapping
```

**Option B: Raw Discord JSON**
```bash
python preprocess.py \
  --raw-dir data/raw \
  --output-dir data/processed \
  --min-blocks 5
```

**Output:** `data/processed/{train,val,test}.parquet`

### 3. Train Baseline Model

```bash
python train_baseline.py \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir models/baseline \
  --epochs 1 \
  --batch-size 64 \
  --fp16
```

**Note:** Current baseline includes dimensional collapse fix (scale=100.0, temperature=0.01).

### 4. Evaluate with Whitening

```bash
python evaluate.py \
  --model models/baseline \
  --test-data data/processed/test.parquet \
  --train-data data/processed/train.parquet \
  --whitening \
  --output outputs/evaluation
```

**Whitening** applies mean-subtraction to remove common embedding components, spreading the representation across the full hypersphere.

### 5. Run Autonomous Mining Loop (Optional)

```bash
python run_loop.py \
  --base-model models/baseline \
  --output models/loop \
  --iterations 3 \
  --min-similarity 0.7 \
  --max-similarity 0.95
```

This runs iterative hard-negative mining with distance filtering to progressively improve the model.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Data Pipeline                                     │
│  Raw Data → Cleaning → Session Aggregation → Splits        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Baseline Training                                 │
│  RoBERTa + Mean Pooling + MNRL (scale=100.0)              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Hard-Negative Mining Loop (Optional)             │
│  FAISS Mining → Distance Filtering → TripletLoss Training  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Evaluation with Whitening                        │
│  Mean-Subtraction → EER, ROC-AUC, UMAP Visualization      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Data Pipeline
- **Input:** Discord JSON or Parquet files (20.8M messages, 10 servers)
- **Processing:** Session aggregation (5-min window), 20-512 tokens per block
- **Splits:** 90% train, 10% val (same users), 1000 held-out test authors (zero-shot)
- **Format:** Parquet with columns: `author_id`, `channel_id`, `text`, `num_messages`

### Model Architecture
```
RoBERTa-base (125M params)
    ↓
Mean Pooling
    ↓
768-dim L2-normalized embeddings
```

### Loss Functions

**Baseline (Phase 2):**
- `MultipleNegativesRankingLoss` with scale=100.0 (temperature=0.01)
- In-batch negatives: (batch_size - 1) negatives per anchor
- Sharper softmax punishes close negatives harder

**Refinement (Phase 3):**
- `TripletMarginLoss` with margin=0.5
- Uses FAISS-mined hard negatives with distance filtering

### Evaluation Metrics
- **EER (Equal Error Rate):** Point where FAR = FRR (threshold-independent)
- **ROC-AUC:** Area under ROC curve
- **Visualizations:** ROC curves, FAR/FRR curves, score distributions, UMAP projections

### Whitening (Post-Processing)
```python
# Compute mean from training set
mean_vector = np.mean(train_embeddings, axis=0)

# During inference
embedding_centered = embedding_raw - mean_vector
embedding_final = embedding_centered / ||embedding_centered||
```

Removes "common component" from embeddings, fixing dimensional collapse (anisotropy).

## Key Research Findings

### Problem: Dimensional Collapse
- **Initial EER:** 34.8% (threshold: 0.9999)
- **Cause:** All embeddings clustered in tiny cone on hypersphere
- **Root causes:** Dominant [CLS] token signal, weak negative samples, insufficient loss sharpness

### Solution 1: Whitening 
- **Implementation:** Post-processing mean-subtraction
- **Impact:** EER 34.8% → 17.35% (50% reduction)
- **Threshold:** 0.9999 → 0.2122 (proper spread)
- **No retraining required**

### Solution 2: Lower Temperature  (Implemented, needs training)
- **Change:** scale=100.0 (temp=0.01) vs default scale=20.0 (temp=0.05)
- **Expected:** 3-5 pp EER improvement
- **Status:** Code updated, needs model retraining

### Solution 3: Distance Filtering  (Implemented, needs training)
- **Change:** Keep hard negatives where 0.7 < similarity < 0.95
- **Filters:** Too similar (>0.95, likely duplicates), too dissimilar (<0.7, too easy)
- **Expected:** 2-4 pp EER improvement
- **Status:** Code updated, needs loop execution

## Project Structure

```
fingertest/
├── data/
│   ├── raw/                      # Discord JSON dumps (optional)
│   ├── {server_id}.parquet/      # Pre-processed Parquet files
│   └── processed/                # Train/val/test splits
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── models/
│   ├── baseline/                 # Initial MNRL model
│   └── loop/                     # Autonomous loop outputs
├── checkpoints/
│   ├── model_1/                  # Saved checkpoints
│   └── model_2/
├── outputs/
│   └── evaluation*/              # Evaluation results + plots
├── experiments/                  # Experiment tracking (NEW)
│   ├── README.md                 # Experiment log
│   ├── EXPERIMENT_TEMPLATE.md    # Template for new experiments
│   └── 001_*.md                  # Completed experiments
├── preprocess.py                 # JSON → Parquet preprocessor
├── preprocess_parquet.py         # Parquet → processed splits
├── train_baseline.py             # Phase 2: Baseline training
├── miner.py                      # Phase 3A: Hard-negative mining
├── train_triplet.py              # Phase 3B: Triplet training
├── run_loop.py                   # Phase 3: Autonomous loop
├── evaluate.py                   # Phase 4: Evaluation
├── config.py                     # Configuration management
├── config.ini                    # Hyperparameters
└── requirements.txt              # Dependencies
```

## Hardware Requirements

- **GPU:** NVIDIA GPU with 12GB+ VRAM (tested on RTX 3090)
- **CUDA:** 12.x for faiss-gpu-cu12
- **RAM:** 16GB+ recommended
- **Disk:** 20GB+ free space

## Configuration

Edit `config.ini` for experiment parameters:

```ini
[baseline]
batch_size = 64
num_epochs = 1
learning_rate = 2e-5

[triplet]
batch_size = 32
margin = 0.5

[mining]
sample_size = 50000
k = 10
min_similarity = 0.7     # Distance filtering
max_similarity = 0.95    # Distance filtering

[wandb]
enabled = true
project = authorship-verification
```

## Running Experiments

### Standard Workflow

1. **Make changes** to code/config
2. **Copy template:** `cp experiments/EXPERIMENT_TEMPLATE.md experiments/00X_name.md`
3. **Document hypothesis** and methodology
4. **Run experiment** and collect metrics
5. **Update experiment doc** with results
6. **Update experiments/README.md** log table
7. **Commit** with meaningful message

### Useful Flags

```bash
# Disable wandb logging
python train_baseline.py --no-wandb

# Custom wandb run name
python train_baseline.py --wandb-name my-experiment

# Adjust batch size (if OOM)
python train_baseline.py --batch-size 32

# Enable/disable whitening
python evaluate.py --whitening      # Enable (default)
python evaluate.py --no-whitening   # Disable
```

## Key Design Decisions

### 1. Streaming Data Pipeline
Handles 10GB+ datasets without loading into RAM using generators and chunked processing.

### 2. Session Aggregation
Groups consecutive messages from same (user, channel) if Δt < 5 minutes, creating 20-512 token context blocks. Better than single messages for style learning.

### 3. Zero-Shot Test Set
Bottom 1000 authors completely held out. Model never sees these users during training, ensuring true verification capability.

### 4. Hard-Negative Mining with Distance Filtering
- Uses FAISS for fast GPU-accelerated similarity search
- Finds "confusing" examples (different authors, similar style)
- Filters false negatives (too similar) and easy negatives (too dissimilar)
- Goldilocks zone: 0.7 < similarity < 0.95

### 5. Whitening for Dimensional Collapse
Post-processing step that removes common components, spreading embeddings across full hypersphere without retraining.

### 6. Metric Learning over Classification
Learns embedding space via contrastive/triplet losses instead of author classification. Generalizes to unseen authors.

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size`
- Enable `--fp16` (should be default)
- Reduce `--sample-size` in mining

### Poor EER (>20%)
- Ensure whitening is enabled: `--whitening`
- Check if model was trained with scale=100.0
- Increase `--iterations` in loop
- Verify data quality (enough authors with sufficient samples)

### FAISS GPU Error
```bash
# Check GPU availability
python -c "import faiss; print(f'GPUs: {faiss.get_num_gpus()}')"

# If needed, fallback to CPU (slower)
python miner.py --no-gpu
```

### Wandb Issues
```bash
# Login
wandb login

# Disable if needed
export WANDB_DISABLED=1
# or
python train_baseline.py --no-wandb
```

## Next Steps

**With New Config System:**
1. Create new experiment: `python scripts/new_experiment.py --name my_experiment`
2. Edit config: `configs/experiments/00X_my_experiment.yaml`
3. Run training: `python train_baseline.py --config configs/experiments/00X_my_experiment.yaml`
4. Results auto-logged to experiments/ and wandb

**Research Priorities:**
1. Train new baseline with lower temperature (scale=100.0)
2. Run autonomous loop with distance filtering (0.7-0.95 similarity)
3. Evaluate with whitening to reach <15% EER target

See [experiments/README.md](experiments/README.md) for experiment tracking.

## Documentation

**New System:**
-  [`docs/SETUP_AND_USAGE.md`](docs/SETUP_AND_USAGE.md) - Complete usage guide
-  [`docs/MIGRATION_STATUS.md`](docs/MIGRATION_STATUS.md) - Migration progress

**Experiments:**
-  [`experiments/README.md`](experiments/README.md) - Experiment log
-  [`experiments/EXPERIMENT_TEMPLATE.md`](experiments/EXPERIMENT_TEMPLATE.md) - Template

## References

- **Sentence-Transformers:** https://www.sbert.net/
- **FAISS:** https://github.com/facebookresearch/faiss
- **RoBERTa:** https://arxiv.org/abs/1907.11692
- **MultipleNegativesRankingLoss:** https://arxiv.org/abs/1705.00652

## License

MIT
