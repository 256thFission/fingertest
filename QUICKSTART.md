# Quick Start Guide

## Current Data Status

✅ You have 20.8M Discord messages across 10 servers  
✅ Available in both JSON (27GB) and Parquet (551MB) formats  
✅ Parquet files are pre-cleaned and include metadata  

## Problem & Solution

**Problem**: Existing code (`preprocess.py`) expects different data format  
**Solution**: New script `preprocess_parquet.py` adapts existing Parquet files

## Run Preprocessing

### Option 1: Fast (Recommended)

```bash
uv run python preprocess_parquet.py --skip-channel-mapping
```

**Time**: ~2-3 minutes  
**Output**: `data/processed/{train,val,test}.parquet`

### Option 2: With Channel Mapping

```bash
uv run python preprocess_parquet.py
```

**Time**: ~5-10 minutes  
**Output**: Same, but with accurate channel IDs

## Next Steps

After preprocessing completes:

### 1. Train Baseline Model

```bash
uv run python train_baseline.py \
  --train-data data/processed/train.parquet \
  --val-data data/processed/val.parquet \
  --output-dir models/baseline
```

### 2. Mine Hard Negatives

```bash
uv run python miner.py \
  --model models/baseline \
  --data data/processed/train.parquet \
  --output data/processed/hard_negatives.parquet
```

### 3. Train with Triplet Loss

```bash
uv run python train_triplet.py \
  --hard-negatives data/processed/hard_negatives.parquet \
  --output-dir models/triplet
```

### 4. Evaluate

```bash
uv run python evaluate.py \
  --model models/triplet \
  --test-data data/processed/test.parquet
```

## File Reference

| File | Purpose |
|------|---------|
| `preprocess_parquet.py` | **NEW** - Process existing Parquet files |
| `preprocess.py` | Original JSON preprocessor |
| `train_baseline.py` | Train RoBERTa bi-encoder |
| `miner.py` | Mine hard negatives with FAISS |
| `train_triplet.py` | Train with hard negatives |
| `evaluate.py` | Evaluate model performance |
| `DATA_RECONCILIATION.md` | Detailed reconciliation report |

## Troubleshooting

### Setup (Using UV)
```bash
# Install dependencies
uv pip install -r requirements.txt

# Verify GPU FAISS
uv run python -c "import faiss; print(f'FAISS: {faiss.__version__}'); print(f'GPUs: {faiss.get_num_gpus()}')"
```

**Requirements**: Python 3.11+ with CUDA 12 for GPU acceleration

### Check preprocessing output
```bash
uv run python -c "
import pyarrow.parquet as pq
table = pq.read_table('data/processed/train.parquet')
print(f'Train samples: {table.num_rows:,}')
print(f'Schema: {table.schema}')
"
```
