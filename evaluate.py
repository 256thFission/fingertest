#!/usr/bin/env python3
"""
Phase 4: Forensic Evaluation
Implements EER, ROC-AUC, and visualization for authorship verification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AuthorshipEvaluator:
    """Forensic evaluation for authorship verification."""

    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)

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
        """Compute cosine similarities for all pairs."""

        logger.info("Computing embeddings...")

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
                normalize_embeddings=True,
                show_progress_bar=False,
            )
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
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )

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
        self, output_dir: str, num_positive: int = 2000, num_negative: int = 2000
    ) -> Dict:
        """Run full evaluation pipeline."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info("=" * 80)

        return metrics


def evaluate_model(model_path: str, test_data_path: str, output_dir: str) -> Dict:
    """Convenience function for evaluation."""
    evaluator = AuthorshipEvaluator(model_path, test_data_path)
    return evaluator.evaluate(output_dir)


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
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-positive", type=int, default=2000, help="Number of positive pairs"
    )
    parser.add_argument(
        "--num-negative", type=int, default=2000, help="Number of negative pairs"
    )

    args = parser.parse_args()

    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return

    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return

    evaluator = AuthorshipEvaluator(args.model, args.test_data)
    evaluator.evaluate(
        output_dir=args.output,
        num_positive=args.num_positive,
        num_negative=args.num_negative,
    )


if __name__ == "__main__":
    main()
