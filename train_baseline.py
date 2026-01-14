#!/usr/bin/env python3
"""
Phase 2: Baseline Metric Learning Trainer
Trains a RoBERTa-based bi-encoder with MultipleNegativesRankingLoss.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
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


class WandbCallback:
    """Custom callback for logging SentenceTransformer training to Weights & Biases."""

    def __init__(self, run=None):
        self.run = run
        self.step = 0

    def on_step_end(self, score, epoch, steps):
        """Called after each training step."""
        if self.run is not None:
            self.step += 1
            # Log basic step info
            self.run.log(
                {
                    "train/step": self.step,
                    "train/epoch": epoch,
                },
                step=self.step,
            )

    def on_epoch_end(self, epoch, steps, evaluator_scores):
        """Called after each epoch."""
        if self.run is not None:
            # Log evaluation scores
            if evaluator_scores:
                for evaluator_name, score in evaluator_scores.items():
                    self.run.log(
                        {
                            f"eval/{evaluator_name}": score,
                            "train/epoch": epoch,
                        },
                        step=self.step,
                    )


class WandbEvaluator(EmbeddingSimilarityEvaluator):
    """Extended evaluator that logs to wandb."""

    def __init__(self, *args, wandb_run=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_run = wandb_run

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        """Run evaluation and log to wandb."""
        score = super().__call__(model, output_path, epoch, steps)

        if self.wandb_run is not None and score is not None:
            # Log evaluation metrics
            self.wandb_run.log(
                {
                    f"eval/{self.name}_cosine_spearman": score,
                    "train/epoch": epoch,
                    "train/steps": steps,
                }
            )

        return score


class AuthorshipDataset(Dataset):
    """Dataset for authorship verification with in-batch negatives."""

    def __init__(self, data_path: str):
        """Load data from parquet file."""
        self.data = pq.read_table(data_path).to_pandas()

        # Group by author for positive pair sampling
        self.author_groups = self.data.groupby("author_id").groups

        # Filter authors with at least 2 blocks (needed for pairs)
        self.valid_authors = [
            author_id
            for author_id, indices in self.author_groups.items()
            if len(indices) >= 2
        ]

        logger.info(
            f"Loaded {len(self.data)} blocks from {len(self.valid_authors)} authors"
        )

    def __len__(self):
        return len(self.valid_authors) * 10  # Multiple epochs worth

    def __getitem__(self, idx):
        """Sample a positive pair (anchor, positive) from same author."""
        import random

        # Select random author
        author_id = self.valid_authors[idx % len(self.valid_authors)]
        author_indices = self.author_groups[author_id].tolist()

        # Sample two different blocks from same author
        if len(author_indices) < 2:
            # Fallback: use same block twice (rare edge case)
            anchor_idx = positive_idx = author_indices[0]
        else:
            anchor_idx, positive_idx = random.sample(author_indices, 2)

        anchor_text = self.data.iloc[anchor_idx]["text"]
        positive_text = self.data.iloc[positive_idx]["text"]

        return InputExample(texts=[anchor_text, positive_text], label=1.0)


class ValidationDataset:
    """Generate validation pairs (positive and negative)."""

    def __init__(self, data_path: str, num_pairs: int = 1000):
        self.data = pq.read_table(data_path).to_pandas()
        self.author_groups = self.data.groupby("author_id").groups
        self.valid_authors = [
            author_id
            for author_id, indices in self.author_groups.items()
            if len(indices) >= 2
        ]
        self.num_pairs = num_pairs

    def generate_pairs(self):
        """Generate balanced positive and negative pairs."""
        import random

        pairs = []
        labels = []

        num_positive = self.num_pairs // 2
        num_negative = self.num_pairs - num_positive

        # Positive pairs (same author)
        for _ in range(num_positive):
            author_id = random.choice(self.valid_authors)
            indices = self.author_groups[author_id].tolist()

            if len(indices) >= 2:
                idx1, idx2 = random.sample(indices, 2)
                text1 = self.data.iloc[idx1]["text"]
                text2 = self.data.iloc[idx2]["text"]
                pairs.append((text1, text2))
                labels.append(1.0)

        # Negative pairs (different authors)
        for _ in range(num_negative):
            author1, author2 = random.sample(self.valid_authors, 2)
            idx1 = random.choice(self.author_groups[author1].tolist())
            idx2 = random.choice(self.author_groups[author2].tolist())

            text1 = self.data.iloc[idx1]["text"]
            text2 = self.data.iloc[idx2]["text"]
            pairs.append((text1, text2))
            labels.append(0.0)

        return pairs, labels


class BaselineTrainer:
    """Trains RoBERTa bi-encoder with MNRL."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        output_dir: str = "models/baseline",
        max_seq_length: int = 512,
        wandb_config: Optional[dict] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_config = wandb_config or {}
        self.wandb_run = None

        # Build bi-encoder with mean pooling
        logger.info(f"Initializing model: {model_name}")

        word_embedding_model = models.Transformer(
            model_name, max_seq_length=max_seq_length
        )

        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        logger.info(
            f"Model initialized with {self.model.get_sentence_embedding_dimension()}D embeddings"
        )

    def train(
        self,
        train_path: str,
        val_path: str,
        batch_size: int = 64,
        num_epochs: int = 1,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        fp16: bool = True,
        checkpoint_save_steps: int = 5000,
        use_wandb: bool = True,
    ):
        """Train the model with MultipleNegativesRankingLoss."""

        # Initialize wandb if available and enabled
        if use_wandb and WANDB_AVAILABLE and self.wandb_config.get("enabled", True):
            self.wandb_run = wandb.init(
                project=self.wandb_config.get("project", "authorship-verification"),
                entity=self.wandb_config.get("entity"),
                name=self.wandb_config.get("name")
                or f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=self.wandb_config.get("tags", []) + ["baseline", "phase2"],
                notes=self.wandb_config.get("notes"),
                config={
                    "model_name": self.model.get_sentence_embedding_dimension(),
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "warmup_steps": warmup_steps,
                    "fp16": fp16,
                    "loss": "MultipleNegativesRankingLoss",
                    "loss_scale": 100.0,
                    "loss_temperature": 0.01,
                    "phase": "baseline_training",
                },
            )
            logger.info(f"Wandb run initialized: {self.wandb_run.name}")
        else:
            self.wandb_run = None
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("Wandb not available. Install with: pip install wandb")

        logger.info("=" * 80)
        logger.info("Starting Baseline Training")
        logger.info("=" * 80)

        # Load datasets
        train_dataset = AuthorshipDataset(train_path)

        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Define loss: MultipleNegativesRankingLoss (in-batch negatives)
        # Scale = 1/temperature. Lower temperature (0.01) = higher scale (100.0)
        # This creates a sharper softmax distribution, punishing close negatives harder
        train_loss = losses.MultipleNegativesRankingLoss(self.model, scale=100.0)

        # Validation evaluator
        val_dataset = ValidationDataset(val_path, num_pairs=1000)
        val_pairs, val_labels = val_dataset.generate_pairs()

        evaluator = WandbEvaluator(
            sentences1=[p[0] for p in val_pairs],
            sentences2=[p[1] for p in val_pairs],
            scores=val_labels,
            name="validation",
            wandb_run=self.wandb_run,
        )

        # Calculate total steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * num_epochs

        logger.info(f"Training configuration:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  FP16: {fp16}")
        logger.info(f"  Wandb: {self.wandb_run is not None}")

        # Log dataset info to wandb
        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "dataset/train_samples": len(train_dataset),
                    "dataset/train_authors": len(train_dataset.valid_authors),
                    "dataset/val_pairs": len(val_pairs),
                    "config/steps_per_epoch": steps_per_epoch,
                    "config/total_steps": total_steps,
                }
            )

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(self.output_dir),
            save_best_model=True,
            checkpoint_path=str(self.output_dir / "checkpoints"),
            checkpoint_save_steps=checkpoint_save_steps,
            checkpoint_save_total_limit=3,
            use_amp=fp16,  # Automatic Mixed Precision for FP16
            show_progress_bar=True,
        )

        logger.info("=" * 80)
        logger.info(f"Training complete! Model saved to {self.output_dir}")
        logger.info("=" * 80)

        # Finish wandb run
        if self.wandb_run is not None:
            # Log final model artifact
            if self.wandb_config.get("log_model", True):
                artifact = wandb.Artifact(
                    name=f"baseline-model",
                    type="model",
                    description="Baseline authorship verification model",
                )
                artifact.add_dir(str(self.output_dir))
                self.wandb_run.log_artifact(artifact)

            self.wandb_run.finish()

        return self.model

    def load_model(self, model_path: str = None):
        """Load a trained model."""
        if model_path is None:
            model_path = str(self.output_dir)

        logger.info(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        return self.model


def optimize_batch_size(vram_gb: int = 24) -> int:
    """
    Calculate optimal batch size for given VRAM.
    RoBERTa-base with 512 seq length uses ~400MB per sample in fp16.
    """
    # Conservative estimate: 350MB per sample with fp16
    mb_per_sample = 350
    available_mb = vram_gb * 1024 * 0.8  # Use 80% of VRAM

    batch_size = int(available_mb / mb_per_sample)

    # Round to nearest power of 2 for efficiency
    batch_size = 2 ** int(torch.log2(torch.tensor(batch_size)).item())

    # Clamp to reasonable range
    batch_size = max(16, min(batch_size, 128))

    return batch_size


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train baseline authorship verification model"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.parquet",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/baseline",
        help="Output directory for model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use mixed precision training"
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable mixed precision training",
    )
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
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name",
    )

    args = parser.parse_args()

    # Check if training data exists
    if not Path(args.train_data).exists():
        logger.error(f"Training data not found: {args.train_data}")
        logger.info("Please run preprocess.py first")
        return

    # Auto-calculate batch size if not specified
    if args.batch_size is None:
        args.batch_size = optimize_batch_size(vram_gb=24)
        logger.info(f"Auto-selected batch size: {args.batch_size}")

    # Prepare wandb config
    wandb_config = {
        "enabled": args.wandb,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_name,
    }

    # Initialize trainer
    trainer = BaselineTrainer(
        model_name="roberta-base",
        output_dir=args.output_dir,
        wandb_config=wandb_config,
    )

    # Train
    trainer.train(
        train_path=args.train_data,
        val_path=args.val_data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
