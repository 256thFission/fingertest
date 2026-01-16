#!/usr/bin/env python3
"""
Phase 4: Forensic Evaluation
Implements EER, ROC-AUC, and visualization for authorship verification.

NOW USING: YAML Config System
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import json

from src.config_v2 import ExperimentConfig, apply_overrides
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata, log_evaluation_artifacts
from src.git_utils import get_git_info
from src.reproducibility import set_random_seeds, get_system_info

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


class Evaluator:
    """Forensic evaluation for authorship verification."""

    def __init__(
        self,
        config: ExperimentConfig,
        wandb_run: Optional["wandb.Run"] = None,
        model_path: Optional[str] = None
    ):
        self.config = config
        self.wandb_run = wandb_run

        # Use provided model path or get from config
        if model_path:
            self.model_path = Path(model_path)
        elif config.baseline_training:
            self.model_path = Path(config.baseline_training.output_dir)
        elif config.loop:
            self.model_path = Path(config.loop.output_dir) / "final_model"
        else:
            raise ValueError("No model path found in config")

        self.test_data_path = Path(config.data.test_path)
        self.train_data_path = Path(config.data.train_path) if config.data.train_path else None
        self.use_whitening = config.evaluation.use_whitening
        self.mean_vector = None
        self.output_dir = Path(config.evaluation.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = SentenceTransformer(str(self.model_path))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Load test data
        logger.info(f"Loading test data from {self.test_data_path}")
        self.data = pq.read_table(str(self.test_data_path)).to_pandas()

        # Group by author
        self.author_groups = self.data.groupby("author_id").groups
        self.valid_authors = [
            author_id
            for author_id, indices in self.author_groups.items()
            if len(indices) >= 2
        ]

        logger.info(
            f"Test set: {len(self.data)} blocks, {len(self.valid_authors)} authors"
        )

        # Compute mean vector for whitening
        if self.use_whitening:
            if self.train_data_path and self.train_data_path.exists():
                logger.info("Computing mean vector from training data for whitening...")
                self.mean_vector = self._compute_mean_vector(str(self.train_data_path))
                logger.info(f"Mean vector computed (shape: {self.mean_vector.shape})")
            else:
                logger.warning("Whitening enabled but no training data. Disabling whitening.")
                self.use_whitening = False

    def _compute_mean_vector(
        self, train_data_path: str, max_samples: int = 50000, batch_size: int = 64
    ) -> np.ndarray:
        """Compute mean vector from training data for whitening."""
        logger.info(f"Loading training data from {train_data_path}")
        train_data = pq.read_table(train_data_path).to_pandas()

        if len(train_data) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(train_data)} training samples")
            train_data = train_data.sample(n=max_samples, random_state=42)

        texts = train_data["text"].tolist()

        logger.info(f"Encoding {len(texts)} training samples for mean computation...")
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Computing mean vector"):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=False
            )
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        mean_vector = np.mean(all_embeddings, axis=0)

        logger.info(f"Mean vector computed with shape {mean_vector.shape}")
        return mean_vector

    def compute_similarities(
        self, pairs: List[Tuple[str, str]], batch_size: int = 64
    ) -> np.ndarray:
        """Compute cosine similarities with optional whitening."""
        all_sims = []

        for i in tqdm(range(0, len(pairs), batch_size), desc="Computing similarities"):
            batch_pairs = pairs[i : i + batch_size]
            texts1 = [p[0] for p in batch_pairs]
            texts2 = [p[1] for p in batch_pairs]

            batch = texts1 + texts2

            # Encode without normalization first
            embeddings = self.model.encode(
                batch, batch_size=batch_size * 2, convert_to_numpy=True,
                normalize_embeddings=False
            )

            # Apply whitening (mean-subtraction) if enabled
            if self.use_whitening and self.mean_vector is not None:
                embeddings = embeddings - self.mean_vector

            # Normalize after whitening
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            # Split back
            emb1 = embeddings[: len(texts1)]
            emb2 = embeddings[len(texts1) :]

            # Compute cosine similarities
            sims = np.sum(emb1 * emb2, axis=1)
            all_sims.append(sims)

        return np.concatenate(all_sims)

    def generate_test_pairs(
        self, num_positive: int = 2000, num_negative: int = 2000
    ) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        """Generate balanced test pairs."""
        import random

        pairs = []
        labels = []

        # Positive pairs (same author)
        for _ in range(num_positive):
            author_id = random.choice(self.valid_authors)
            indices = self.author_groups[author_id].tolist()

            if len(indices) >= 2:
                idx1, idx2 = random.sample(indices, 2)
                text1 = self.data.iloc[idx1]["text"]
                text2 = self.data.iloc[idx2]["text"]
                pairs.append((text1, text2))
                labels.append(1)

        # Negative pairs (different authors)
        for _ in range(num_negative):
            author1, author2 = random.sample(self.valid_authors, 2)
            idx1 = random.choice(self.author_groups[author1].tolist())
            idx2 = random.choice(self.author_groups[author2].tolist())

            text1 = self.data.iloc[idx1]["text"]
            text2 = self.data.iloc[idx2]["text"]
            pairs.append((text1, text2))
            labels.append(0)

        return pairs, np.array(labels)

    def compute_eer(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Equal Error Rate."""
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr

        # Find threshold where FAR = FRR
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return eer, eer_threshold

    def evaluate(self) -> Dict[str, float]:
        """Run complete evaluation."""
        logger.info("=" * 80)
        logger.info("Starting Forensic Evaluation")
        logger.info("=" * 80)

        # Generate test pairs
        logger.info(f"Generating {self.config.evaluation.num_positive_pairs} positive and {self.config.evaluation.num_negative_pairs} negative pairs...")
        pairs, labels = self.generate_test_pairs(
            num_positive=self.config.evaluation.num_positive_pairs,
            num_negative=self.config.evaluation.num_negative_pairs
        )

        # Compute similarities
        logger.info(f"Computing similarities (whitening={'ON' if self.use_whitening else 'OFF'})...")
        similarities = self.compute_similarities(pairs)

        # Compute metrics
        logger.info("Computing metrics...")
        eer, eer_threshold = self.compute_eer(similarities, labels)
        fpr, tpr, _ = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)

        # Accuracy at EER threshold
        predictions = (similarities >= eer_threshold).astype(int)
        accuracy_at_eer = np.mean(predictions == labels)

        metrics = {
            "experiment_id": self.config.experiment.id,
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "roc_auc": float(roc_auc),
            "accuracy_at_eer": float(accuracy_at_eer),
            "whitening_enabled": self.use_whitening,
            "num_test_pairs": len(pairs),
            "target_eer": self.config.evaluation.target_eer,
            "target_roc_auc": self.config.evaluation.target_roc_auc,
        }

        # Log to console
        logger.info("=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        logger.info(f"EER: {eer:.4f} ({eer * 100:.2f}%)")
        logger.info(f"EER Threshold: {eer_threshold:.4f}")
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Accuracy @ EER: {accuracy_at_eer:.4f}")
        logger.info(f"Whitening: {'ON' if self.use_whitening else 'OFF'}")
        logger.info("=" * 80)

        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log({
                "eval/eer": eer,
                "eval/eer_threshold": eer_threshold,
                "eval/roc_auc": roc_auc,
                "eval/accuracy_at_eer": accuracy_at_eer,
            })

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Generate visualizations
        self._plot_roc_curve(fpr, tpr, roc_auc)
        self._plot_far_frr_curves(similarities, labels, eer, eer_threshold)
        self._plot_score_distribution(similarities, labels)

        # Log artifacts to wandb
        if self.wandb_run:
            log_evaluation_artifacts(self.wandb_run, self.output_dir, metrics)

        return metrics

    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")

    def _plot_far_frr_curves(self, similarities, labels, eer, eer_threshold):
        """Plot FAR/FRR curves."""
        thresholds = np.linspace(similarities.min(), similarities.max(), 1000)
        fars = []
        frrs = []

        for thresh in thresholds:
            predictions = (similarities >= thresh).astype(int)

            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            tp = np.sum((predictions == 1) & (labels == 1))

            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0

            fars.append(far)
            frrs.append(frr)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, fars, label='FAR (False Accept Rate)', color='red')
        plt.plot(thresholds, frrs, label='FRR (False Reject Rate)', color='blue')
        plt.axvline(eer_threshold, color='green', linestyle='--', label=f'EER Threshold = {eer_threshold:.4f}')
        plt.axhline(eer, color='green', linestyle='--', label=f'EER = {eer:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Error Rate')
        plt.title('FAR/FRR Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "far_frr_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"FAR/FRR curves saved to {output_path}")

    def _plot_score_distribution(self, similarities, labels):
        """Plot similarity score distributions."""
        plt.figure(figsize=(10, 6))

        positive_scores = similarities[labels == 1]
        negative_scores = similarities[labels == 0]

        plt.hist(positive_scores, bins=50, alpha=0.5, label='Same Author', color='green')
        plt.hist(negative_scores, bins=50, alpha=0.5, label='Different Author', color='red')

        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.title('Similarity Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "score_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Score distribution saved to {output_path}")


def main():
    """Main evaluation function using YAML config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model using YAML configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--model",
        help="Override model path (optional)"
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
        logger.warning(f"Applied {len(args.overrides)} config overrides")

    # Validate
    errors = config.validate()
    if errors:
        logger.error("Config validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Set seeds
    set_random_seeds(config.reproducibility.random_seed)

    # Initialize wandb if needed
    wandb_run = None
    if config.wandb.enabled:
        git_info = get_git_info()
        system_info = get_system_info()
        wandb_run = init_wandb_with_metadata(config, git_info, system_info)

    # Run evaluation
    evaluator = Evaluator(config, wandb_run, model_path=args.model)
    metrics = evaluator.evaluate()

    # Update experiment tracker if this is part of an experiment
    if hasattr(config, 'experiment') and config.experiment:
        tracker = ExperimentTracker(config)
        tracker.log_results(metrics, wandb_run.url if wandb_run else None)

    # Finish
    if wandb_run:
        wandb_run.finish()

    logger.info(f" Evaluation complete! Results: {config.evaluation.output_dir}")


if __name__ == "__main__":
    main()
