#!/usr/bin/env python3
"""
Phase 3: Hard Negative Mining with FAISS
Finds difficult negatives (different authors with similar writing styles).
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """FAISS-based hard negative mining for metric learning."""

    def __init__(
        self,
        model_path: str,
        data_path: str,
        output_path: str = "data/processed/hard_negatives.parquet",
        use_gpu: bool = True,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_path = Path(output_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        if self.use_gpu:
            self.model = self.model.cuda()

        # Load data
        logger.info(f"Loading data from {data_path}")
        self.data = pq.read_table(data_path).to_pandas()
        logger.info(f"Loaded {len(self.data)} blocks")

        self.index = None
        self.embeddings = None

    def encode_corpus(
        self, batch_size: int = 128, sample_size: int = 50000
    ) -> np.ndarray:
        """
        Encode a subset of the corpus for mining.
        Using a sample to avoid OOM and speed up mining.
        """
        # Sample data if needed
        if len(self.data) > sample_size:
            logger.info(f"Sampling {sample_size} blocks from {len(self.data)}")
            self.data = self.data.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )

        texts = self.data["text"].tolist()

        logger.info(f"Encoding {len(texts)} texts...")

        # Encode in batches
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        logger.info(f"Encoded embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def build_index(self, use_gpu: bool = None) -> faiss.Index:
        """
        Build FAISS index for fast similarity search.
        Using FlatIP (Inner Product) since embeddings are normalized (cosine similarity).
        """
        if use_gpu is None:
            use_gpu = self.use_gpu

        if self.embeddings is None:
            raise ValueError("Must encode corpus first")

        dimension = self.embeddings.shape[1]

        logger.info(f"Building FAISS index (dimension={dimension})...")

        # Use FlatIP for exact search (Inner Product = Cosine for normalized vectors)
        index = faiss.IndexFlatIP(dimension)

        # Move to GPU if available
        if use_gpu:
            logger.info("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add vectors
        index.add(self.embeddings.astype("float32"))

        logger.info(f"Index built with {index.ntotal} vectors")
        self.index = index
        return index

    def mine_hard_negatives(
        self, k: int = 10, prioritize_same_channel: bool = True,
        min_similarity: float = 0.7, max_similarity: float = 0.95
    ) -> List[Dict]:
        """
        Mine hard negatives using FAISS nearest neighbor search with distance filtering.

        For each block:
        - Find top-k nearest neighbors
        - Filter for different authors (hard negatives)
        - Apply distance filtering: keep negatives with min_sim < similarity < max_sim
        - Discard if similarity > max_sim (likely duplicates/false negatives)
        - Discard if similarity < min_sim (too easy, not hard enough)
        - Prioritize same channel if enabled (kills topic bias)

        Distance filtering converts: distance = 1 - similarity
        - distance < 0.3 means similarity > 0.7 (hard enough)
        - distance < 0.05 means similarity > 0.95 (too hard, discard)
        """
        if self.index is None:
            raise ValueError("Must build index first")

        logger.info(f"Mining hard negatives (k={k}, {min_similarity:.2f} < sim < {max_similarity:.2f})...")

        triplets = []
        filtered_stats = {
            "too_similar": 0,  # similarity > max_sim (likely duplicates)
            "too_dissimilar": 0,  # similarity < min_sim (too easy)
            "same_author": 0,  # same author (not negative)
            "kept": 0,
        }

        # Search for each query
        similarities, indices = self.index.search(
            self.embeddings.astype("float32"), k + 1
        )

        for i in tqdm(range(len(self.data)), desc="Mining"):
            anchor_row = self.data.iloc[i]
            anchor_author = anchor_row["author_id"]
            anchor_channel = anchor_row["channel_id"]

            # Get neighbors (skip first one, which is the query itself)
            neighbor_indices = indices[i][1:]
            neighbor_sims = similarities[i][1:]

            # Find hard negatives (different author)
            hard_negatives = []

            for neighbor_idx, sim in zip(neighbor_indices, neighbor_sims):
                neighbor_row = self.data.iloc[neighbor_idx]
                neighbor_author = neighbor_row["author_id"]
                neighbor_channel = neighbor_row["channel_id"]

                # Skip if same author
                if neighbor_author == anchor_author:
                    filtered_stats["same_author"] += 1
                    continue

                # Apply distance filtering
                # Discard if too similar (likely duplicate/false negative)
                if sim > max_similarity:
                    filtered_stats["too_similar"] += 1
                    continue

                # Discard if too dissimilar (too easy, not hard enough)
                if sim < min_similarity:
                    filtered_stats["too_dissimilar"] += 1
                    continue

                # This is a valid hard negative
                filtered_stats["kept"] += 1

                # Calculate score (prioritize same channel)
                score = sim
                if prioritize_same_channel and neighbor_channel == anchor_channel:
                    score += 0.1  # Boost same-channel negatives

                hard_negatives.append(
                    {
                        "idx": neighbor_idx,
                        "similarity": float(sim),
                        "score": float(score),
                        "same_channel": neighbor_channel == anchor_channel,
                    }
                )

            if not hard_negatives:
                continue

            # Sort by score and take top negative
            hard_negatives.sort(key=lambda x: x["score"], reverse=True)
            best_negative = hard_negatives[0]

            # Now find a positive (same author, different text)
            same_author_blocks = self.data[
                (self.data["author_id"] == anchor_author) & (self.data.index != i)
            ]

            if len(same_author_blocks) == 0:
                continue

            # Sample random positive
            positive_idx = same_author_blocks.sample(1).index[0]

            triplets.append(
                {
                    "anchor_text": anchor_row["text"],
                    "anchor_author": anchor_author,
                    "anchor_channel": anchor_channel,
                    "positive_text": self.data.iloc[positive_idx]["text"],
                    "positive_author": self.data.iloc[positive_idx]["author_id"],
                    "negative_text": self.data.iloc[best_negative["idx"]]["text"],
                    "negative_author": self.data.iloc[best_negative["idx"]][
                        "author_id"
                    ],
                    "negative_channel": self.data.iloc[best_negative["idx"]][
                        "channel_id"
                    ],
                    "negative_similarity": best_negative["similarity"],
                    "same_channel_negative": best_negative["same_channel"],
                }
            )

        logger.info(f"Mined {len(triplets)} hard triplets")

        # Log filtering statistics
        logger.info("Filtering statistics:")
        total_candidates = sum(filtered_stats.values())
        logger.info(f"  Total candidates examined: {total_candidates}")
        logger.info(f"  Same author (skipped): {filtered_stats['same_author']} ({100 * filtered_stats['same_author'] / max(total_candidates, 1):.1f}%)")
        logger.info(f"  Too similar (sim > {max_similarity}): {filtered_stats['too_similar']} ({100 * filtered_stats['too_similar'] / max(total_candidates, 1):.1f}%)")
        logger.info(f"  Too dissimilar (sim < {min_similarity}): {filtered_stats['too_dissimilar']} ({100 * filtered_stats['too_dissimilar'] / max(total_candidates, 1):.1f}%)")
        logger.info(f"  Kept as hard negatives: {filtered_stats['kept']} ({100 * filtered_stats['kept'] / max(total_candidates, 1):.1f}%)")

        # Calculate statistics
        if len(triplets) > 0:
            same_channel_count = sum(1 for t in triplets if t["same_channel_negative"])
            logger.info(
                f"Same-channel negatives: {same_channel_count}/{len(triplets)} ({100 * same_channel_count / len(triplets):.1f}%)"
            )

            avg_sim = np.mean([t["negative_similarity"] for t in triplets])
            min_sim_actual = np.min([t["negative_similarity"] for t in triplets])
            max_sim_actual = np.max([t["negative_similarity"] for t in triplets])
            logger.info(f"Negative similarity range: [{min_sim_actual:.4f}, {max_sim_actual:.4f}]")
            logger.info(f"Average negative similarity: {avg_sim:.4f}")
        else:
            logger.warning("No triplets mined! Consider relaxing distance filtering constraints.")

        return triplets

    def save_triplets(self, triplets: List[Dict]):
        """Save triplets to parquet."""
        if not triplets:
            logger.warning("No triplets to save")
            return

        df = pd.DataFrame(triplets)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, compression="snappy")

        logger.info(f"Saved {len(triplets)} triplets to {self.output_path}")

    def run(
        self,
        batch_size: int = 128,
        sample_size: int = 50000,
        k: int = 10,
        prioritize_same_channel: bool = True,
        min_similarity: float = 0.7,
        max_similarity: float = 0.95,
    ) -> List[Dict]:
        """Run full mining pipeline with distance filtering."""
        logger.info("=" * 80)
        logger.info("Hard Negative Mining with Distance Filtering")
        logger.info("=" * 80)
        logger.info(f"Distance filtering: {min_similarity:.2f} < similarity < {max_similarity:.2f}")
        logger.info(f"  Equivalent to: {1-max_similarity:.2f} < distance < {1-min_similarity:.2f}")

        # Encode
        self.encode_corpus(batch_size=batch_size, sample_size=sample_size)

        # Build index
        self.build_index()

        # Mine
        triplets = self.mine_hard_negatives(
            k=k, prioritize_same_channel=prioritize_same_channel,
            min_similarity=min_similarity, max_similarity=max_similarity
        )

        # Save
        self.save_triplets(triplets)

        logger.info("=" * 80)
        logger.info("Mining complete!")
        logger.info("=" * 80)

        return triplets


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mine hard negatives with FAISS")
    parser.add_argument(
        "--model", type=str, default="models/baseline", help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train.parquet",
        help="Path to training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/hard_negatives.parquet",
        help="Output path for mined triplets",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50000, help="Number of samples to mine from"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of neighbors to retrieve"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Encoding batch size"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument(
        "--prioritize-same-channel",
        action="store_true",
        default=True,
        help="Prioritize same-channel negatives",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.7,
        help="Minimum similarity for hard negatives (distance < 0.3)",
    )
    parser.add_argument(
        "--max-similarity",
        type=float,
        default=0.95,
        help="Maximum similarity for hard negatives (distance > 0.05, avoid duplicates)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("Please run train_baseline.py first")
        return

    # Initialize miner
    miner = HardNegativeMiner(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        use_gpu=not args.no_gpu,
    )

    # Run mining
    miner.run(
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        k=args.k,
        prioritize_same_channel=args.prioritize_same_channel,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
    )


if __name__ == "__main__":
    main()
