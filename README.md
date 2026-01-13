# Autonomous Authorship Verification System

A robust, end-to-end authorship verification system that learns to map text styles to a vector space using metric learning with active hard-negative mining.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: Data Pipeline                      │
│  Raw Discord JSON → Cleaning → Session Aggregation → Splits    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: Baseline Training                      │
│        RoBERTa + Mean Pooling + MNRL (In-batch Negatives)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           PHASE 3: Autonomous Hard-Negative Loop                │
│  Train → FAISS Mining → Find Hard Negatives → Retrain (3x)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 4: Forensic Evaluation                       │
│         EER, ROC-AUC, UMAP Visualization (Zero-Shot)           │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **RAM:** 32GB+ recommended
- **Disk:** 20GB+ free space

## Quick Start

### 1. Prepare Data

Place your Discord JSON dumps in `data/raw/`:
```bash
data/raw/
├── server1/
│   ├── channel1.json
│   └── channel2.json
├── server2/
│   └── general.json
...
```

**Expected JSON format:**
```json
[
  {
    "author": {
      "id": "123456789",
      "username": "user123",
      "bot": false,
      "discriminator": "1234"
    },
    "content": "Message text here",
    "timestamp": "2024-01-01T12:00:00.000Z",
    "channel_id": "987654321"
  },
  ...
]
```

### 2. Run Full Pipeline

```bash
./run_full_pipeline.sh
```

This will:
1. Install dependencies
2. Preprocess data
3. Train baseline model
4. Run autonomous mining loop (3 iterations)
5. Evaluate on zero-shot test set

### 3. Check Results

Results will be in:
- `outputs/final_evaluation/metrics.json` - Key metrics (EER, ROC-AUC)
- `outputs/final_evaluation/*.png` - Visualizations
- `models/loop/final_model/` - Trained model

## Manual Execution

### Phase 1: Data Preprocessing

```bash
python preprocess.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --min-blocks 5
```

**What it does:**
- Streams 10GB+ JSON files without OOM
- Removes bots and system messages
- Normalizes URLs/mentions, keeps emojis
- Aggregates messages into context blocks (20-512 tokens)
- Creates stratified splits:
  - Train: 90% of data from users with 5+ blocks
  - Val: 10% of data from train users
  - Test: 1000 held-out authors (zero-shot)

### Phase 2: Baseline Training

```bash
python train_baseline.py \
    --train-data data/processed/train.parquet \
    --val-data data/processed/val.parquet \
    --output-dir models/baseline \
    --batch-size 64 \
    --epochs 1 \
    --lr 2e-5 \
    --fp16
```

**Architecture:**
- Base: `roberta-base` (125M params)
- Pooling: Mean pooling
- Loss: MultipleNegativesRankingLoss (in-batch negatives)
- Output: 768-dim normalized embeddings

### Phase 3: Hard-Negative Mining

**Step A: Mine hard negatives**
```bash
python miner.py \
    --model models/baseline \
    --data data/processed/train.parquet \
    --output data/processed/hard_negatives.parquet \
    --sample-size 50000 \
    --k 10 \
    --prioritize-same-channel
```

**What it does:**
- Encodes 50k random training samples
- Builds FAISS index (GPU-accelerated)
- For each sample, finds top-10 nearest neighbors
- Identifies different-author neighbors as "hard negatives"
- Prioritizes same-channel negatives (kills topic bias)

**Step B: Fine-tune on triplets**
```bash
python train_triplet.py \
    --model models/baseline \
    --triplets data/processed/hard_negatives.parquet \
    --output models/triplet_refined \
    --batch-size 32 \
    --epochs 1 \
    --lr 1e-5 \
    --margin 0.5 \
    --fp16
```

**Loss:** TripletMarginLoss with margin=0.5

### Autonomous Loop

```bash
python run_loop.py \
    --base-model models/baseline \
    --data-dir data/processed \
    --output models/loop \
    --iterations 3 \
    --sample-size 50000 \
    --mining-k 10 \
    --batch-size 32 \
    --lr 1e-5 \
    --margin 0.5 \
    --fp16
```

This automatically runs:
```
Iteration 1: Mine → Train → Evaluate
Iteration 2: Mine → Train → Evaluate
Iteration 3: Mine → Train → Evaluate
```

### Phase 4: Evaluation

```bash
python evaluate.py \
    --model models/loop/final_model \
    --test-data data/processed/test.parquet \
    --output outputs/final_evaluation \
    --num-positive 2000 \
    --num-negative 2000
```

**Metrics computed:**
- **EER (Equal Error Rate):** Target < 0.05
- **ROC-AUC:** Area under ROC curve
- **Accuracy at EER threshold**

**Visualizations:**
- `roc_curve.png` - ROC curve
- `far_frr_curves.png` - FAR/FRR with EER point
- `score_distribution.png` - Similarity distributions
- `umap_visualization.png` - 2D embedding space

## Key Design Decisions

### 1. Streaming Data Pipeline
- Uses generators and HuggingFace datasets to handle 10GB+ without RAM overflow
- Processes in chunks of 10k messages

### 2. Session Aggregation
- Groups consecutive messages from (user, channel) if Δt < 5 minutes
- Creates context-rich blocks (20-512 tokens)
- Better than single messages for style learning

### 3. Zero-Shot Test Set
- Bottom 1000 authors completely held out
- Model never sees these users during training
- Tests true authorship verification capability

### 4. Hard-Negative Mining
- Uses FAISS for fast similarity search (GPU-accelerated)
- Finds "confusing" examples (different authors, similar style)
- Prioritizes same-channel negatives to reduce topic bias

### 5. Autonomous Loop
- Iteratively improves by finding its own mistakes
- Each iteration:
  1. Identifies hard negatives (model's errors)
  2. Retrains on these hard cases
  3. Evaluates on zero-shot test

### 6. RoBERTa over BERT
- Better at byte-level noise handling
- More robust to Discord's informal text style

## File Structure

```
.
├── data/
│   ├── raw/                    # Place Discord JSON here
│   └── processed/              # Parquet files (train/val/test)
├── models/
│   ├── baseline/               # Initial MNRL model
│   └── loop/                   # Autonomous loop outputs
│       ├── iteration_1/
│       ├── iteration_2/
│       ├── iteration_3/
│       ├── final_model/        # Best model
│       └── results.json        # Metrics per iteration
├── outputs/
│   ├── baseline_evaluation/    # Baseline metrics
│   └── final_evaluation/       # Final metrics + visualizations
├── preprocess.py               # Phase 1: Data pipeline
├── train_baseline.py           # Phase 2: Baseline trainer
├── miner.py                    # Phase 3A: Hard-negative mining
├── train_triplet.py            # Phase 3B: Triplet training
├── run_loop.py                 # Phase 3: Autonomous loop
├── evaluate.py                 # Phase 4: Evaluation
├── run_full_pipeline.sh        # Master script
└── requirements.txt            # Python dependencies
```

## Performance Tuning

### Batch Size Optimization
Auto-calculated based on VRAM:
- Baseline training: 64-128 (MNRL)
- Triplet training: 32 (TripletLoss has 3x overhead)

### Memory Management
- Use `fp16=True` for 2x memory savings
- FAISS index on GPU for fast mining
- Streaming encoders for large corpora

### Hyperparameters
Tested defaults:
- Learning rate: 2e-5 (baseline), 1e-5 (refinement)
- Triplet margin: 0.5
- Mining k: 10 neighbors
- Sample size: 50k per iteration

## Expected Results

| Metric | Target | Typical |
|--------|--------|---------|
| EER | < 0.05 | 0.03-0.06 |
| ROC-AUC | > 0.95 | 0.94-0.98 |
| Same-Channel Negatives | > 30% | 35-45% |

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size`
- Reduce `--sample-size` in mining
- Ensure `--fp16` is enabled

### No Data Found
- Check `data/raw/` has .json files
- Verify JSON format matches expected structure

### Poor EER (> 0.10)
- Increase `--iterations` in loop
- Increase `--min-blocks` (more data per author)
- Check data quality (enough authors?)

### FAISS GPU Error
- Install: `pip install faiss-gpu`
- Fallback to CPU: `--no-gpu` flag in miner.py

## Citation

If you use this system, please cite:
```
Autonomous Authorship Verification System
Hard-Negative Mining Loop with RoBERTa + FAISS
```

## License

MIT
