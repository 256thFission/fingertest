#!/usr/bin/env python3
"""
Phase 4: Forensic Evaluation
Implements EER, ROC-AUC, and visualization for authorship verification.
"""

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


class AuthorshipEvaluator:
    """Forensic evaluation for authorship verification."""

    def __init__(
        self, model_path: str, test_data_path: str, wandb_config: Optional[dict] = None,
        train_data_path: Optional[str] = None, use_whitening: bool = True
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.wandb_config = wandb_config or {}
        self.wandb_run = None
        self.use_whitening = use_whitening
        self.mean_vector = None

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = SentenceTransformer(str(model_path))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        self.data = pq.read_table(str(test_data_path)).to_pandas()

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

        # Compute mean vector from training set for whitening
        if self.use_whitening:
            if self.train_data_path and self.train_data_path.exists():
                logger.info("Computing mean vector from training data for whitening...")
                self.mean_vector = self._compute_mean_vector(str(self.train_data_path))
                logger.info(f"Mean vector computed (shape: {self.mean_vector.shape})")
            else:
                logger.warning("Whitening enabled but no training data path provided. Whitening will be disabled.")
                self.use_whitening = False

    def _compute_mean_vector(
        self, train_data_path: str, max_samples: int = 50000, batch_size: int = 64
    ) -> np.ndarray:
        """
        Compute mean vector from training data for whitening.
        This removes the "common component" shared by all embeddings.
        """
        logger.info(f"Loading training data from {train_data_path}")
        train_data = pq.read_table(train_data_path).to_pandas()

        # Sample if too large
        if len(train_data) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(train_data)} training samples")
            train_data = train_data.sample(n=max_samples, random_state=42)

        texts = train_data["text"].tolist()

        # Encode all training texts
        logger.info(f"Encoding {len(texts)} training samples for mean computation...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing mean vector"):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Don't normalize yet
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)

        # Compute mean
        mean_vector = np.mean(all_embeddings, axis=0)

        return mean_vector

    def generate_test_pairs(
        self, num_positive: int = 2000, num_negative: int = 2000
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Generate balanced positive and negative pairs for evaluation."""
        import random

        pairs = []
        labels = []

        logger.info(
            f"Generating {num_positive} positive and {num_negative} negative pairs..."
        )

        # Positive pairs (same author)
        for _ in tqdm(range(num_positive), desc="Generating positive pairs"):
            author_id = random.choice(self.valid_authors)
            indices = self.author_groups[author_id].tolist()

            if len(indices) < 2:
                continue

            idx1, idx2 = random.sample(indices, 2)
            text1 = self.data.iloc[idx1]["text"]
            text2 = self.data.iloc[idx2]["text"]

            pairs.append((text1, text2))
            labels.append(1)

        # Negative pairs (different authors)
        for _ in tqdm(range(num_negative), desc="Generating negative pairs"):
            author1, author2 = random.sample(self.valid_authors, 2)

            idx1 = random.choice(self.author_groups[author1].tolist())
            idx2 = random.choice(self.author_groups[author2].tolist())

            text1 = self.data.iloc[idx1]["text"]
            text2 = self.data.iloc[idx2]["text"]

            pairs.append((text1, text2))
            labels.append(0)

        logger.info(
            f"Generated {len(pairs)} pairs ({sum(labels)} positive, {len(labels) - sum(labels)} negative)"
        )

        return pairs, labels

    def compute_similarities(
        self, pairs: List[Tuple[str, str]], batch_size: int = 64
    ) -> np.ndarray:
        """Compute cosine similarities for all pairs with optional whitening."""

        logger.info("Computing embeddings...")
        if self.use_whitening and self.mean_vector is not None:
            logger.info("Whitening enabled: applying mean-subtraction before similarity computation")

        # Extract all unique texts
        all_texts = list(set([text for pair in pairs for text in pair]))

        # Encode all texts
        text_to_embedding = {}
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
            batch = all_texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,  # Don't normalize yet if whitening
                show_progress_bar=False,
            )

            # Apply whitening (mean-subtraction) if enabled
            if self.use_whitening and self.mean_vector is not None:
                embeddings = embeddings - self.mean_vector

            # Now normalize to unit length for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # Add epsilon to avoid division by zero

            for text, emb in zip(batch, embeddings):
                text_to_embedding[text] = emb

        # Compute similarities
        logger.info("Computing similarities...")
        similarities = []
        for text1, text2 in tqdm(pairs, desc="Computing cosine similarities"):
            emb1 = text_to_embedding[text1]
            emb2 = text_to_embedding[text2]
            sim = np.dot(emb1, emb2)  # Cosine similarity (vectors are normalized)
            similarities.append(sim)

        return np.array(similarities)

    def calculate_eer(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Equal Error Rate (EER).
        Returns (EER, threshold at EER).
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # Calculate False Rejection Rate (FRR = 1 - TPR)
        frr = 1 - tpr

        # Find EER (where FAR = FRR)
        # FAR = FPR (False Positive Rate)
        eer_idx = np.nanargmin(np.abs(fpr - frr))
        eer = (fpr[eer_idx] + frr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return eer, eer_threshold

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        output_path: Path,
        roc_auc: float,
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FAR)", fontsize=12)
        plt.ylabel("True Positive Rate (1 - FRR)", fontsize=12)
        plt.title("ROC Curve - Authorship Verification", fontsize=14)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"ROC curve saved to {output_path}")

    def plot_far_frr_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        output_path: Path,
        eer: float,
        eer_threshold: float,
    ):
        """Plot FAR/FRR curves with EER point."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        frr = 1 - tpr

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, fpr, label="FAR (False Accept Rate)", lw=2)
        plt.plot(thresholds, frr, label="FRR (False Reject Rate)", lw=2)
        plt.axvline(
            eer_threshold,
            color="red",
            linestyle="--",
            label=f"EER = {eer:.4f} @ {eer_threshold:.4f}",
        )
        plt.axhline(eer, color="red", linestyle="--", alpha=0.5)
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Error Rate", fontsize=12)
        plt.title("FAR/FRR Curves with EER", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"FAR/FRR curves saved to {output_path}")

    def plot_score_distribution(
        self, y_true: np.ndarray, y_scores: np.ndarray, output_path: Path
    ):
        """Plot score distributions for positive and negative pairs."""
        positive_scores = y_scores[y_true == 1]
        negative_scores = y_scores[y_true == 0]

        plt.figure(figsize=(12, 6))
        plt.hist(
            positive_scores,
            bins=50,
            alpha=0.7,
            label="Positive pairs (same author)",
            color="green",
        )
        plt.hist(
            negative_scores,
            bins=50,
            alpha=0.7,
            label="Negative pairs (different authors)",
            color="red",
        )
        plt.xlabel("Cosine Similarity", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title("Score Distribution", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Score distribution saved to {output_path}")

    def generate_umap_visualization(
        self, output_dir: Path, num_authors: int = 50, samples_per_author: int = 10
    ):
        """Generate UMAP visualization of embeddings."""
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not installed. Skipping visualization.")
            return

        logger.info("Generating UMAP visualization...")

        # Sample authors
        import random

        sampled_authors = random.sample(
            self.valid_authors, min(num_authors, len(self.valid_authors))
        )

        texts = []
        labels = []
        author_names = {}

        for i, author_id in enumerate(sampled_authors):
            author_names[i] = author_id
            indices = self.author_groups[author_id].tolist()
            sampled_indices = random.sample(
                indices, min(samples_per_author, len(indices))
            )

            for idx in sampled_indices:
                texts.append(self.data.iloc[idx]["text"])
                labels.append(i)

        # Encode
        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=False  # Don't normalize yet if whitening
        )

        # Apply whitening if enabled
        if self.use_whitening and self.mean_vector is not None:
            logger.info("Applying whitening to UMAP embeddings")
            embeddings = embeddings - self.mean_vector
            # Normalize after whitening
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        # UMAP
        logger.info("Running UMAP...")
        reducer = umap.UMAP(
            n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(16, 12))
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap="tab20",
            s=20,
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Author ID")
        plt.title(
            f"UMAP Visualization ({num_authors} authors, {len(texts)} samples)",
            fontsize=14,
        )
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "umap_visualization.png", dpi=300)
        plt.close()

        logger.info(
            f"UMAP visualization saved to {output_dir / 'umap_visualization.png'}"
        )

    def evaluate(
        self,
        output_dir: str,
        num_positive: int = 2000,
        num_negative: int = 2000,
        use_wandb: bool = True,
    ) -> Dict:
        """Run full evaluation pipeline."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if available and enabled
        if use_wandb and WANDB_AVAILABLE and self.wandb_config.get("enabled", True):
            from datetime import datetime

            self.wandb_run = wandb.init(
                project=self.wandb_config.get("project", "authorship-verification"),
                entity=self.wandb_config.get("entity"),
                name=self.wandb_config.get("name")
                or f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=self.wandb_config.get("tags", []) + ["evaluation", "phase4"],
                notes=self.wandb_config.get("notes"),
                config={
                    "num_positive_pairs": num_positive,
                    "num_negative_pairs": num_negative,
                    "model_path": str(self.model_path),
                    "use_whitening": self.use_whitening,
                    "phase": "evaluation",
                },
            )
            logger.info(f"Wandb run initialized: {self.wandb_run.name}")
        else:
            self.wandb_run = None
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("Wandb not available. Install with: pip install wandb")

        logger.info("=" * 80)
        logger.info("Forensic Evaluation")
        logger.info("=" * 80)

        # Generate pairs
        pairs, labels = self.generate_test_pairs(num_positive, num_negative)

        # Compute similarities
        similarities = self.compute_similarities(pairs)

        # Convert to numpy
        y_true = np.array(labels)
        y_scores = similarities

        # Calculate metrics
        logger.info("Calculating metrics...")

        # EER
        eer, eer_threshold = self.calculate_eer(y_true, y_scores)
        logger.info(f"EER: {eer:.4f} (threshold: {eer_threshold:.4f})")

        # ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        logger.info(f"ROC-AUC: {roc_auc:.4f}")

        # Calculate accuracy at EER threshold
        predictions = (y_scores >= eer_threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        logger.info(f"Accuracy at EER threshold: {accuracy:.4f}")

        # Plot visualizations
        logger.info("Generating plots...")
        self.plot_roc_curve(y_true, y_scores, output_dir / "roc_curve.png", roc_auc)
        self.plot_far_frr_curves(
            y_true, y_scores, output_dir / "far_frr_curves.png", eer, eer_threshold
        )
        self.plot_score_distribution(
            y_true, y_scores, output_dir / "score_distribution.png"
        )

        # UMAP visualization
        self.generate_umap_visualization(output_dir)

        # Save metrics
        metrics = {
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "roc_auc": float(roc_auc),
            "accuracy_at_eer": float(accuracy),
            "num_test_pairs": len(pairs),
            "num_positive_pairs": int(sum(labels)),
            "num_negative_pairs": int(len(labels) - sum(labels)),
        }

        metrics_path = output_dir / "metrics.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

        # Log to wandb
        if self.wandb_run is not None:
            # Log metrics
            self.wandb_run.log(
                {
                    "eval/eer": eer,
                    "eval/eer_threshold": eer_threshold,
                    "eval/roc_auc": roc_auc,
                    "eval/accuracy_at_eer": accuracy,
                    "eval/num_test_pairs": len(pairs),
                    "eval/num_positive_pairs": int(sum(labels)),
                    "eval/num_negative_pairs": int(len(labels) - sum(labels)),
                }
            )

            # Log plots
            if (output_dir / "roc_curve.png").exists():
                self.wandb_run.log(
                    {"eval/roc_curve": wandb.Image(str(output_dir / "roc_curve.png"))}
                )
            if (output_dir / "far_frr_curves.png").exists():
                self.wandb_run.log(
                    {
                        "eval/far_frr_curves": wandb.Image(
                            str(output_dir / "far_frr_curves.png")
                        )
                    }
                )
            if (output_dir / "score_distribution.png").exists():
                self.wandb_run.log(
                    {
                        "eval/score_distribution": wandb.Image(
                            str(output_dir / "score_distribution.png")
                        )
                    }
                )
            if (output_dir / "umap_visualization.png").exists():
                self.wandb_run.log(
                    {
                        "eval/umap_visualization": wandb.Image(
                            str(output_dir / "umap_visualization.png")
                        )
                    }
                )

            self.wandb_run.finish()

        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info("=" * 80)

        return metrics


def evaluate_model(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    wandb_config: Optional[dict] = None,
    train_data_path: Optional[str] = None,
    use_whitening: bool = True,
) -> Dict:
    """Convenience function for evaluation."""
    evaluator = AuthorshipEvaluator(
        model_path, test_data_path, wandb_config=wandb_config,
        train_data_path=train_data_path, use_whitening=use_whitening
    )
    return evaluator.evaluate(
        output_dir,
        use_wandb=wandb_config is not None and wandb_config.get("enabled", False),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate authorship verification model"
    )
    parser.add_argument(
        "--model", type=str, default="models/baseline", help="Path to trained model"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test.parquet",
        help="Path to test data",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.parquet",
        help="Path to training data (for whitening mean vector computation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--whitening",
        action="store_true",
        default=True,
        help="Enable whitening (mean-subtraction)",
    )
    parser.add_argument(
        "--no-whitening",
        dest="whitening",
        action="store_false",
        help="Disable whitening",
    )
    parser.add_argument(
        "--num-positive", type=int, default=2000, help="Number of positive pairs"
    )
    parser.add_argument(
        "--num-negative", type=int, default=2000, help="Number of negative pairs"
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

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return

    # Prepare wandb config
    wandb_config = {
        "enabled": args.wandb,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
    }

    evaluator = AuthorshipEvaluator(
        args.model, args.test_data, wandb_config=wandb_config,
        train_data_path=args.train_data if args.whitening else None,
        use_whitening=args.whitening
    )
    evaluator.evaluate(
        output_dir=args.output,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
