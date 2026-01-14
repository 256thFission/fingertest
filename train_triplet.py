#!/usr/bin/env python3
"""
Triplet Training: Fine-tune model on hard negatives with TripletMarginLoss.
"""

import logging
from pathlib import Path
from typing import Optional
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

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
            texts=[row["anchor_text"], row["positive_text"], row["negative_text"]]
        )


class TripletTrainer:
    """Fine-tune model on hard negative triplets."""

    def __init__(
        self, model_path: str, output_dir: str, wandb_config: Optional[dict] = None
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_config = wandb_config or {}
        self.wandb_run = None

        logger.info(f"Loading model from {model_path}")
        self.model = SentenceTransformer(str(model_path))

    def train(
        self,
        triplet_path: str,
        batch_size: int = 32,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
        margin: float = 0.5,
        fp16: bool = True,
        use_wandb: bool = True,
        iteration: Optional[int] = None,
    ):
        """Train on triplets with TripletMarginLoss."""

        # Initialize wandb if available and enabled
        if use_wandb and WANDB_AVAILABLE and self.wandb_config.get("enabled", True):
            run_name = self.wandb_config.get("name")
            if run_name is None:
                run_name = f"triplet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if iteration is not None:
                    run_name += f"_iter{iteration}"

            self.wandb_run = wandb.init(
                project=self.wandb_config.get("project", "authorship-verification"),
                entity=self.wandb_config.get("entity"),
                name=run_name,
                tags=self.wandb_config.get("tags", []) + ["triplet", "phase3b"],
                notes=self.wandb_config.get("notes"),
                config={
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "margin": margin,
                    "fp16": fp16,
                    "loss": "TripletLoss",
                    "phase": "triplet_training",
                    "iteration": iteration,
                },
            )
            logger.info(f"Wandb run initialized: {self.wandb_run.name}")
        else:
            self.wandb_run = None
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("Wandb not available. Install with: pip install wandb")

        logger.info("=" * 80)
        logger.info("Triplet Fine-tuning")
        logger.info("=" * 80)

        # Load triplet dataset
        dataset = TripletDataset(triplet_path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Define triplet loss
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=margin,
        )

        steps_per_epoch = len(dataloader)
        warmup_steps = int(steps_per_epoch * 0.1)  # 10% warmup

        logger.info(f"Training configuration:")
        logger.info(f"  Triplets: {len(dataset)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Margin: {margin}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  FP16: {fp16}")
        logger.info(f"  Wandb: {self.wandb_run is not None}")

        # Log dataset info to wandb
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "dataset/num_triplets": len(dataset),
                    "config/warmup_steps": warmup_steps,
                    "config/steps_per_epoch": steps_per_epoch,
                }
            )

        # Train
        self.model.fit(
            train_objectives=[(dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(self.output_dir),
            save_best_model=True,
            use_amp=fp16,
            show_progress_bar=True,
        )

        logger.info("=" * 80)
        logger.info(f"Fine-tuning complete! Model saved to {self.output_dir}")
        logger.info("=" * 80)

        # Finish wandb run
        if self.wandb_run is not None:
            # Log final model artifact
            if self.wandb_config.get("log_model", True):
                artifact = wandb.Artifact(
                    name=f"triplet-model",
                    type="model",
                    description="Triplet-refined authorship verification model",
                )
                artifact.add_dir(str(self.output_dir))
                self.wandb_run.log_artifact(artifact)

            self.wandb_run.finish()

        return self.model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune on hard negative triplets")
    parser.add_argument(
        "--model", type=str, default="models/baseline", help="Path to base model"
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default="data/processed/hard_negatives.parquet",
        help="Path to mined triplets",
    )
    parser.add_argument(
        "--output", type=str, default="models/triplet_refined", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet margin")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="authorship-verification",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (username or team)",
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    if not Path(args.triplets).exists():
        logger.error(f"Triplet data not found: {args.triplets}")
        logger.info("Please run miner.py first")
        return

    # Prepare wandb config
    wandb_config = {
        "enabled": args.wandb,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
    }

    trainer = TripletTrainer(args.model, args.output, wandb_config=wandb_config)
    trainer.train(
        triplet_path=args.triplets,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        margin=args.margin,
        fp16=args.fp16,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
