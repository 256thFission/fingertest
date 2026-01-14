#!/usr/bin/env python3
"""
Phase 3: Autonomous Hard-Negative Mining Loop
Master script that iterates: Train -> Mine -> Retrain
"""

import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from miner import HardNegativeMiner
from train_triplet import TripletTrainer
from evaluate import evaluate_model

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


class AutonomousLoop:
    """Autonomous training loop with hard negative mining."""

    def __init__(
        self,
        base_model_dir: str = "models/baseline",
        data_dir: str = "data/processed",
        output_dir: str = "models/loop",
        num_iterations: int = 3,
        wandb_config: Optional[dict] = None,
    ):
        self.base_model_dir = Path(base_model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_iterations = num_iterations
        self.wandb_config = wandb_config or {}
        self.wandb_run = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.train_data = self.data_dir / "train.parquet"
        self.val_data = self.data_dir / "val.parquet"
        self.test_data = self.data_dir / "test.parquet"

        # Results tracking
        self.results = []

    def run(
        self,
        sample_size: int = 50000,
        mining_k: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        triplet_margin: float = 0.5,
        fp16: bool = True,
        use_wandb: bool = True,
        min_similarity: float = 0.7,
        max_similarity: float = 0.95,
    ):
        """Run the autonomous training loop."""

        # Initialize wandb if available and enabled
        if use_wandb and WANDB_AVAILABLE and self.wandb_config.get("enabled", True):
            self.wandb_run = wandb.init(
                project=self.wandb_config.get("project", "authorship-verification"),
                entity=self.wandb_config.get("entity"),
                name=self.wandb_config.get("name")
                or f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=self.wandb_config.get("tags", []) + ["loop", "phase3"],
                notes=self.wandb_config.get("notes"),
                config={
                    "num_iterations": self.num_iterations,
                    "sample_size": sample_size,
                    "mining_k": mining_k,
                    "min_similarity": min_similarity,
                    "max_similarity": max_similarity,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "triplet_margin": triplet_margin,
                    "fp16": fp16,
                    "phase": "autonomous_loop",
                },
            )
            logger.info(f"Wandb run initialized: {self.wandb_run.name}")
        else:
            self.wandb_run = None
            if use_wandb and not WANDB_AVAILABLE:
                logger.warning("Wandb not available. Install with: pip install wandb")

        logger.info("=" * 100)
        logger.info("AUTONOMOUS HARD-NEGATIVE MINING LOOP")
        logger.info("=" * 100)
        logger.info(f"Base model: {self.base_model_dir}")
        logger.info(f"Iterations: {self.num_iterations}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Wandb: {self.wandb_run is not None}")
        logger.info("=" * 100)

        current_model = str(self.base_model_dir)

        for iteration in range(self.num_iterations):
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"ITERATION {iteration + 1}/{self.num_iterations}")
            logger.info("=" * 100)

            # Create iteration directory
            iter_dir = self.output_dir / f"iteration_{iteration + 1}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Step A: Mine hard negatives
            logger.info(f"\n[Step A] Mining hard negatives from current model...")

            triplet_path = iter_dir / "hard_negatives.parquet"

            miner = HardNegativeMiner(
                model_path=current_model,
                data_path=str(self.train_data),
                output_path=str(triplet_path),
                use_gpu=True,
            )

            triplets = miner.run(
                batch_size=128,
                sample_size=sample_size,
                k=mining_k,
                prioritize_same_channel=True,
                min_similarity=min_similarity,
                max_similarity=max_similarity,
            )

            if not triplets:
                logger.warning("No triplets mined! Stopping loop.")
                break

            # Step B: Fine-tune on hard negatives
            logger.info(f"\n[Step B] Fine-tuning on {len(triplets)} hard triplets...")

            refined_model_dir = iter_dir / "model"

            # Pass wandb config for nested run (or use same run)
            trainer_wandb_config = {
                "enabled": False,  # Disable nested wandb runs, use parent run
            }

            trainer = TripletTrainer(
                model_path=current_model,
                output_dir=str(refined_model_dir),
                wandb_config=trainer_wandb_config,
            )

            trainer.train(
                triplet_path=str(triplet_path),
                batch_size=batch_size,
                num_epochs=1,
                learning_rate=learning_rate,
                margin=triplet_margin,
                fp16=fp16,
                use_wandb=False,  # We'll log to parent wandb run
                iteration=iteration + 1,
            )

            # Log to parent wandb run
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        f"iteration_{iteration + 1}/num_triplets": len(triplets),
                    }
                )

            # Step C: Evaluate on held-out test set
            logger.info(f"\n[Step C] Evaluating on zero-shot test set...")

            metrics = evaluate_model(
                model_path=str(refined_model_dir),
                test_data_path=str(self.test_data),
                output_dir=str(iter_dir / "evaluation"),
            )

            # Store results
            result = {
                "iteration": iteration + 1,
                "num_triplets": len(triplets),
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(result)

            logger.info(f"\nIteration {iteration + 1} Results:")
            logger.info(f"  EER: {metrics.get('eer', -1):.4f}")
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', -1):.4f}")

            # Log iteration metrics to wandb
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        f"iteration_{iteration + 1}/eer": metrics.get("eer", -1),
                        f"iteration_{iteration + 1}/roc_auc": metrics.get(
                            "roc_auc", -1
                        ),
                        f"iteration_{iteration + 1}/num_triplets": len(triplets),
                        "iteration": iteration + 1,
                    }
                )

            # Update current model for next iteration
            current_model = str(refined_model_dir)

            # Save iteration summary
            self.save_results()

        logger.info("\n" + "=" * 100)
        logger.info("AUTONOMOUS LOOP COMPLETE")
        logger.info("=" * 100)

        self.print_summary()

        # Save final model to root output dir
        final_model_path = self.output_dir / "final_model"
        logger.info(f"\nCopying final model to: {final_model_path}")

        if Path(current_model).exists():
            shutil.copytree(current_model, final_model_path, dirs_exist_ok=True)

        # Finish wandb run
        if self.wandb_run is not None:
            # Create summary table
            summary_data = []
            for result in self.results:
                summary_data.append(
                    [
                        result["iteration"],
                        result["num_triplets"],
                        result["metrics"].get("eer", -1),
                        result["metrics"].get("roc_auc", -1),
                    ]
                )

            table = wandb.Table(
                columns=["Iteration", "Triplets", "EER", "ROC-AUC"], data=summary_data
            )
            self.wandb_run.log({"results_summary": table})

            # Log final model
            if self.wandb_config.get("log_model", True):
                artifact = wandb.Artifact(
                    name="loop-final-model",
                    type="model",
                    description="Final model after autonomous loop",
                )
                artifact.add_dir(str(final_model_path))
                self.wandb_run.log_artifact(artifact)

            self.wandb_run.finish()

        return self.results

    def save_results(self):
        """Save results to JSON."""
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    def print_summary(self):
        """Print summary of all iterations."""
        logger.info("\n" + "=" * 100)
        logger.info("SUMMARY")
        logger.info("=" * 100)

        for result in self.results:
            metrics = result["metrics"]
            logger.info(f"Iteration {result['iteration']}:")
            logger.info(f"  Triplets: {result['num_triplets']}")
            logger.info(f"  EER: {metrics.get('eer', -1):.4f}")
            logger.info(f"  ROC-AUC: {metrics.get('roc_auc', -1):.4f}")
            logger.info("")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run autonomous hard-negative mining loop"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/baseline",
        help="Path to baseline model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with processed data",
    )
    parser.add_argument(
        "--output", type=str, default="models/loop", help="Output directory"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of mining iterations"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Samples to mine from each iteration",
    )
    parser.add_argument(
        "--mining-k", type=int, default=10, help="Number of neighbors for mining"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for triplet training"
    )
    parser.add_argument("--margin", type=float, default=0.5, help="Triplet margin")
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

    # Verify base model exists
    if not Path(args.base_model).exists():
        logger.error(f"Base model not found: {args.base_model}")
        logger.info("Please run train_baseline.py first")
        return

    # Prepare wandb config
    wandb_config = {
        "enabled": args.wandb,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
    }

    # Initialize and run loop
    loop = AutonomousLoop(
        base_model_dir=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output,
        num_iterations=args.iterations,
        wandb_config=wandb_config,
    )

    loop.run(
        sample_size=args.sample_size,
        mining_k=args.mining_k,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        triplet_margin=args.margin,
        fp16=args.fp16,
        use_wandb=args.wandb,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
    )


if __name__ == "__main__":
    main()
