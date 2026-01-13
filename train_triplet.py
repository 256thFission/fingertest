#!/usr/bin/env python3
"""
Triplet Training: Fine-tune model on hard negatives with TripletMarginLoss.
"""

import logging
from pathlib import Path
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
    ):
        """Train on triplets with TripletMarginLoss."""

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

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    if not Path(args.triplets).exists():
        logger.error(f"Triplet data not found: {args.triplets}")
        logger.info("Please run miner.py first")
        return

    trainer = TripletTrainer(args.model, args.output)
    trainer.train(
        triplet_path=args.triplets,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        margin=args.margin,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
