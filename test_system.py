#!/usr/bin/env python3
"""
Quick test script to verify the system works end-to-end on synthetic data.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{description}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(1)

    logger.info(f"✓ {description} complete")


def main():
    """Run quick test on synthetic data."""

    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     Authorship Verification System - Quick Test               ║
    ║     Testing on Synthetic Data (Small Scale)                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Generate synthetic data
    if not Path("data/raw/synthetic_server").exists():
        run_command(
            [
                "python",
                "generate_synthetic_data.py",
                "--users",
                "50",
                "--min-messages",
                "10",
                "--max-messages",
                "30",
                "--channels",
                "5",
                "--output",
                "data/raw/synthetic_server",
            ],
            "Generating synthetic Discord data",
        )
    else:
        logger.info("Synthetic data already exists. Skipping generation.")

    # Step 2: Preprocess
    if not Path("data/processed/train.parquet").exists():
        run_command(
            [
                "python",
                "preprocess.py",
                "--raw-dir",
                "data/raw/synthetic_server",
                "--output-dir",
                "data/processed",
                "--min-blocks",
                "3",
            ],
            "Preprocessing data",
        )
    else:
        logger.info("Processed data exists. Skipping preprocessing.")

    # Step 3: Train baseline (quick test - reduced epochs)
    if not Path("models/baseline").exists():
        run_command(
            [
                "python",
                "train_baseline.py",
                "--train-data",
                "data/processed/train.parquet",
                "--val-data",
                "data/processed/val.parquet",
                "--output-dir",
                "models/baseline",
                "--batch-size",
                "16",
                "--epochs",
                "1",
                "--fp16",
            ],
            "Training baseline model",
        )
    else:
        logger.info("Baseline model exists. Skipping training.")

    # Step 4: Quick evaluation
    run_command(
        [
            "python",
            "evaluate.py",
            "--model",
            "models/baseline",
            "--test-data",
            "data/processed/test.parquet",
            "--output",
            "outputs/test_evaluation",
            "--num-positive",
            "500",
            "--num-negative",
            "500",
        ],
        "Evaluating model",
    )

    # Step 5: Quick mining test
    run_command(
        [
            "python",
            "miner.py",
            "--model",
            "models/baseline",
            "--data",
            "data/processed/train.parquet",
            "--output",
            "data/processed/hard_negatives_test.parquet",
            "--sample-size",
            "1000",
            "--k",
            "5",
            "--batch-size",
            "32",
        ],
        "Testing hard negative mining",
    )

    logger.info(f"\n{'=' * 80}")
    logger.info("✓ QUICK TEST COMPLETE!")
    logger.info(f"{'=' * 80}")
    logger.info("\nTest Results:")
    logger.info("  - Metrics: outputs/test_evaluation/metrics.json")
    logger.info("  - Visualizations: outputs/test_evaluation/*.png")
    logger.info("\nTo run full pipeline with your real data:")
    logger.info("  1. Place Discord JSON in data/raw/")
    logger.info("  2. Run: ./run_full_pipeline.sh")


if __name__ == "__main__":
    main()
