#!/bin/bash
# Master Execution Script for Autonomous Authorship Verification System

set -e  # Exit on error

echo "========================================================================"
echo "Autonomous Authorship Verification System"
echo "========================================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 0: Check for data
echo -e "${YELLOW}[Step 0] Checking for Discord data...${NC}"
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw)" ]; then
    echo -e "${RED}ERROR: No data found in data/raw/${NC}"
    echo "Please place Discord JSON dumps in data/raw/"
    echo "Expected structure: data/raw/server_name/*.json"
    exit 1
fi
echo -e "${GREEN}✓ Data directory found${NC}"
echo ""

# Step 1: Install dependencies
echo -e "${YELLOW}[Step 1] Installing dependencies...${NC}"
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

source .venv/bin/activate
uv pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Preprocess data
echo -e "${YELLOW}[Step 2] Preprocessing Discord data...${NC}"
if [ ! -f "data/processed/train.parquet" ]; then
    python preprocess.py \
        --raw-dir data/raw \
        --output-dir data/processed \
        --min-blocks 5
    echo -e "${GREEN}✓ Preprocessing complete${NC}"
else
    echo "Processed data already exists. Skipping..."
fi
echo ""

# Step 3: Train baseline model
echo -e "${YELLOW}[Step 3] Training baseline model with MNRL...${NC}"
if [ ! -d "models/baseline" ]; then
    python train_baseline.py \
        --train-data data/processed/train.parquet \
        --val-data data/processed/val.parquet \
        --output-dir models/baseline \
        --epochs 1 \
        --fp16
    echo -e "${GREEN}✓ Baseline training complete${NC}"
else
    echo "Baseline model already exists. Skipping..."
fi
echo ""

# Step 4: Evaluate baseline
echo -e "${YELLOW}[Step 4] Evaluating baseline model...${NC}"
python evaluate.py \
    --model models/baseline \
    --test-data data/processed/test.parquet \
    --output outputs/baseline_evaluation
echo -e "${GREEN}✓ Baseline evaluation complete${NC}"
echo ""

# Step 5: Run autonomous hard-negative mining loop
echo -e "${YELLOW}[Step 5] Starting autonomous hard-negative mining loop...${NC}"
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
echo -e "${GREEN}✓ Autonomous loop complete${NC}"
echo ""

# Step 6: Final evaluation
echo -e "${YELLOW}[Step 6] Final evaluation on zero-shot test set...${NC}"
python evaluate.py \
    --model models/loop/final_model \
    --test-data data/processed/test.parquet \
    --output outputs/final_evaluation
echo -e "${GREEN}✓ Final evaluation complete${NC}"
echo ""

echo "========================================================================"
echo -e "${GREEN}SYSTEM BUILD COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Results:"
echo "  - Baseline evaluation: outputs/baseline_evaluation/"
echo "  - Final model: models/loop/final_model/"
echo "  - Final evaluation: outputs/final_evaluation/"
echo "  - Loop results: models/loop/results.json"
echo ""
echo "Key metrics are in outputs/final_evaluation/metrics.json"
echo "Visualizations are available in outputs/final_evaluation/*.png"
echo ""
