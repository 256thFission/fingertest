#!/usr/bin/env python3
"""
Phase 3B: Triplet Fine-tuning
Fine-tunes model on hard negative triplets with TripletMarginLoss.

NOW USING: YAML Config System
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from datetime import datetime

from src.config_v2 import ExperimentConfig, apply_overrides
from src.wandb_utils import init_wandb_with_metadata, log_model_artifact
from src.git_utils import get_git_info
from src.reproducibility import set_random_seeds, get_system_info

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplet training."""

    def __init__(self, triplet_path: str):
        self.data = pd.read_parquet(triplet_path)
        logger.info(f"Loaded {len(self.data)} triplets")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return InputExample(
            texts=[row["anchor"], row["positive"], row["negative"]]
        )


class TripletTrainer:
    """Fine-tune model on hard negative triplets."""

    def __init__(
        self,
        config: ExperimentConfig,
        wandb_run: Optional["wandb.Run"] = None,
        model_path: Optional[str] = None,
        iteration: Optional[int] = None
    ):
        self.config = config
        self.wandb_run = wandb_run
        self.iteration = iteration

        # Use provided model path or get from config
        if model_path:
            self.model_path = Path(model_path)
        elif config.baseline_training:
            self.model_path = Path(config.baseline_training.output_dir)
        else:
            raise ValueError("No model path found")

        self.output_dir = Path(config.triplet_training.output_dir)
        if iteration is not None:
            self.output_dir = self.output_dir.parent / f"iteration_{iteration}" / "model"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from {self.model_path}")
        self.model = SentenceTransformer(str(self.model_path))

    def train(self, triplet_path: str):
        """Train on triplets with TripletMarginLoss."""
        cfg = self.config.triplet_training

        logger.info("=" * 80)
        logger.info("Triplet Fine-tuning")
        if self.iteration is not None:
            logger.info(f"Iteration: {self.iteration}")
        logger.info("=" * 80)

        # Load dataset
        dataset = TripletDataset(triplet_path)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=self.config.hardware.num_workers
        )

        # Define loss
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=cfg.loss.margin,
        )

        steps_per_epoch = len(dataloader)
        warmup_steps = int(steps_per_epoch * cfg.warmup_ratio)

        logger.info(f"Training configuration:")
        logger.info(f"  Batch size: {cfg.batch_size}")
        logger.info(f"  Epochs: {cfg.num_epochs}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Learning rate: {cfg.learning_rate}")
        logger.info(f"  Margin: {cfg.loss.margin}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  FP16: {cfg.fp16}")

        if self.wandb_run:
            self.wandb_run.log({
                "triplet/num_triplets": len(dataset),
                "triplet/steps_per_epoch": steps_per_epoch,
                "triplet/iteration": self.iteration or 0,
            })

        # Train
        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=cfg.num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": cfg.learning_rate},
            output_path=str(self.output_dir),
            show_progress_bar=True,
            use_amp=cfg.fp16,
        )

        logger.info("=" * 80)
        logger.info("Triplet Training Complete!")
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info("=" * 80)

        # Log model artifact
        if self.wandb_run and self.config.wandb.log_model:
            log_model_artifact(
                self.wandb_run,
                self.output_dir,
                self.config.experiment.id,
                artifact_type="triplet_model"
            )

        return str(self.output_dir)


def main():
    """Main training function using YAML config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train triplet model using YAML configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--triplets",
        required=True,
        help="Path to triplet parquet file"
    )
    parser.add_argument(
        "--model",
        help="Override model path (optional)"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        help="Iteration number (for loop)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        help="Config overrides: key=value"
    )
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config: {args.config}")
    config = ExperimentConfig.from_yaml(args.config)

    if args.overrides:
        config = apply_overrides(config, args.overrides)

    # Validate
    errors = config.validate()
    if errors:
        logger.error("Config validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Set seeds
    set_random_seeds(config.reproducibility.random_seed)

    # Initialize wandb
    wandb_run = None
    if config.wandb.enabled:
        git_info = get_git_info()
        system_info = get_system_info()
        wandb_run = init_wandb_with_metadata(config, git_info, system_info)

    # Train
    trainer = TripletTrainer(
        config,
        wandb_run,
        model_path=args.model,
        iteration=args.iteration
    )
    model_path = trainer.train(args.triplets)

    if wandb_run:
        wandb_run.finish()

    logger.info(f" Triplet training complete: {model_path}")


if __name__ == "__main__":
    main()
