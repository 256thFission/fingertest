#!/usr/bin/env python3
"""
Phase 3A: Hard Negative Mining
FAISS-based mining with distance filtering.

NOW USING: YAML Config System
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config_v2 import ExperimentConfig, apply_overrides
from src.git_utils import get_git_info
from src.reproducibility import set_random_seeds, get_system_info

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """FAISS-based hard negative mining for metric learning."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_path: Optional[str] = None,
    ):
        self.config = config

        # Use provided model path or get from config
        if model_path:
            self.model_path = model_path
        elif config.baseline_training:
            self.model_path = config.baseline_training.output_dir
        else:
            raise ValueError("No model path found")

        self.data_path = config.data.train_path
        self.use_gpu = config.hardware.use_gpu and torch.cuda.is_available()

        # Load model
        logger.info(f"Loading model from {self.model_path}")
        self.model = SentenceTransformer(self.model_path)
        if self.use_gpu:
            self.model = self.model.cuda()

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.data = pq.read_table(self.data_path).to_pandas()
        logger.info(f"Loaded {len(self.data)} blocks")

        self.index = None
        self.embeddings = None

    def encode_corpus(self) -> np.ndarray:
        """Encode a subset of the corpus for mining."""
        cfg = self.config.mining

        # Sample data if needed
        if len(self.data) > cfg.sample_size:
            logger.info(f"Sampling {cfg.sample_size} blocks from {len(self.data)}")
            self.data = self.data.sample(n=cfg.sample_size, random_state=42).reset_index(drop=True)

        texts = self.data["text"].tolist()

        logger.info(f"Encoding {len(texts)} texts...")

        self.embeddings = self.model.encode(
            texts,
            batch_size=cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Encoded embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def build_index(self) -> faiss.Index:
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None:
            raise ValueError("Must encode corpus first")

        dimension = self.embeddings.shape[1]

        logger.info(f"Building FAISS index (dimension={dimension})...")

        index = faiss.IndexFlatIP(dimension)

        # Move to GPU if available
        if self.use_gpu:
            logger.info("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(self.embeddings)

        logger.info(f"Index built with {index.ntotal} vectors")
        self.index = index
        return index

    def mine_hard_negatives(self) -> pd.DataFrame:
        """Mine hard negatives with distance filtering."""
        if self.index is None:
            raise ValueError("Must build index first")

        cfg = self.config.mining

        logger.info("=" * 80)
        logger.info("Mining Hard Negatives")
        logger.info("=" * 80)
        logger.info(f"K-neighbors: {cfg.k_neighbors}")
        logger.info(f"Distance filter: {cfg.min_similarity:.2f} - {cfg.max_similarity:.2f}")
        logger.info(f"Prioritize same channel: {cfg.prioritize_same_channel}")

        triplets = []
        stats = {
            "too_similar": 0,
            "too_dissimilar": 0,
            "same_author": 0,
            "kept": 0,
        }

        for anchor_idx in tqdm(range(len(self.data)), desc="Mining"):
            anchor_author = self.data.iloc[anchor_idx]["author_id"]
            anchor_channel = self.data.iloc[anchor_idx].get("channel_id", None)
            anchor_text = self.data.iloc[anchor_idx]["text"]

            # Find k nearest neighbors
            anchor_emb = self.embeddings[anchor_idx : anchor_idx + 1]
            neighbor_sims, neighbor_indices = self.index.search(anchor_emb, cfg.k_neighbors + 1)

            neighbor_sims = neighbor_sims[0]
            neighbor_indices = neighbor_indices[0]

            # Find positive (same author, different text)
            positive_idx = None
            for idx, sim in zip(neighbor_indices, neighbor_sims):
                if idx == anchor_idx:
                    continue
                if self.data.iloc[idx]["author_id"] == anchor_author:
                    positive_idx = idx
                    break

            if positive_idx is None:
                continue

            positive_text = self.data.iloc[positive_idx]["text"]

            # Find hard negative
            hard_negative = None
            for idx, sim in zip(neighbor_indices, neighbor_sims):
                if idx == anchor_idx:
                    continue

                neg_author = self.data.iloc[idx]["author_id"]

                # Skip same author
                if neg_author == anchor_author:
                    stats["same_author"] += 1
                    continue

                # Apply distance filtering
                if sim > cfg.max_similarity:
                    stats["too_similar"] += 1
                    continue

                if sim < cfg.min_similarity:
                    stats["too_dissimilar"] += 1
                    continue

                # Check channel priority
                score = sim
                if cfg.prioritize_same_channel and anchor_channel:
                    neg_channel = self.data.iloc[idx].get("channel_id", None)
                    if neg_channel == anchor_channel:
                        score += 0.1

                hard_negative = (idx, score)
                break

            if hard_negative is None:
                continue

            stats["kept"] += 1

            neg_idx, _ = hard_negative
            negative_text = self.data.iloc[neg_idx]["text"]

            triplets.append({
                "anchor": anchor_text,
                "positive": positive_text,
                "negative": negative_text,
                "anchor_author": anchor_author,
                "negative_author": self.data.iloc[neg_idx]["author_id"],
            })

        # Log statistics
        logger.info("=" * 80)
        logger.info("Mining Statistics")
        logger.info("=" * 80)
        logger.info(f"Total triplets mined: {len(triplets)}")
        logger.info(f"Filtered too similar: {stats['too_similar']}")
        logger.info(f"Filtered too dissimilar: {stats['too_dissimilar']}")
        logger.info(f"Filtered same author: {stats['same_author']}")
        logger.info(f"Kept: {stats['kept']}")
        logger.info("=" * 80)

        return pd.DataFrame(triplets)

    def run(self, output_path: str) -> str:
        """Complete mining pipeline."""
        # Encode corpus
        self.encode_corpus()

        # Build index
        self.build_index()

        # Mine hard negatives
        triplets_df = self.mine_hard_negatives()

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        triplets_df.to_parquet(output_path)

        logger.info(f"Saved {len(triplets_df)} triplets to {output_path}")
        return str(output_path)


def main():
    """Main mining function using YAML config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mine hard negatives using YAML configuration"
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
        "--output",
        required=True,
        help="Output path for triplets"
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

    # Mine
    miner = HardNegativeMiner(config, model_path=args.model)
    triplet_path = miner.run(args.output)

    logger.info(f" Mining complete: {triplet_path}")


if __name__ == "__main__":
    main()
