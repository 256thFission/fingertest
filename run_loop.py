#!/usr/bin/env python3
"""
Phase 3: Autonomous Training Loop
Iterative hard negative mining and fine-tuning.

NOW USING: YAML Config System
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from miner import HardNegativeMiner
from train_triplet import TripletTrainer
from evaluate import Evaluator

from src.config_v2 import ExperimentConfig, apply_overrides
from src.experiment_tracker import ExperimentTracker
from src.wandb_utils import init_wandb_with_metadata
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


class AutonomousLoop:
    """Autonomous training loop with hard negative mining."""

    def __init__(
        self,
        config: ExperimentConfig,
        wandb_run: Optional["wandb.Run"] = None,
        base_model: Optional[str] = None,
    ):
        self.config = config
        self.wandb_run = wandb_run

        # Use provided base model or get from config
        if base_model:
            self.base_model_dir = Path(base_model)
        elif config.baseline_training:
            self.base_model_dir = Path(config.baseline_training.output_dir)
        else:
            raise ValueError("No base model path found")

        self.output_dir = Path(config.loop.output_dir)
        self.num_iterations = config.loop.num_iterations

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = []

    def run(self):
        """Run the autonomous training loop."""
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
            iter_dir.mkdir(exist_ok=True)

            # Step 1: Mine hard negatives
            logger.info("")
            logger.info(f"[Step 1/{3}] Mining hard negatives...")
            logger.info("")

            triplet_path = iter_dir / "triplets.parquet"

            miner = HardNegativeMiner(self.config, model_path=current_model)
            miner.run(str(triplet_path))

            # Step 2: Train on triplets
            logger.info("")
            logger.info(f"[Step 2/{3}] Training on triplets...")
            logger.info("")

            trainer = TripletTrainer(
                self.config,
                self.wandb_run,
                model_path=current_model,
                iteration=iteration + 1
            )
            refined_model = trainer.train(str(triplet_path))

            # Step 3: Evaluate
            logger.info("")
            logger.info(f"[Step 3/{3}] Evaluating...")
            logger.info("")

            evaluator = Evaluator(self.config, self.wandb_run, model_path=refined_model)
            metrics = evaluator.evaluate()

            # Log iteration results
            result = {
                "iteration": iteration + 1,
                "eer": metrics["eer"],
                "roc_auc": metrics["roc_auc"],
                "eer_threshold": metrics["eer_threshold"],
            }
            self.results.append(result)

            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    f"loop/iteration": iteration + 1,
                    f"loop/eer_iter{iteration + 1}": metrics["eer"],
                    f"loop/roc_auc_iter{iteration + 1}": metrics["roc_auc"],
                })

            logger.info("")
            logger.info(f"Iteration {iteration + 1} Results:")
            logger.info(f"  EER: {metrics['eer']:.4f} ({metrics['eer'] * 100:.2f}%)")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info("")

            # Update current model for next iteration
            current_model = refined_model

        # Save final model
        final_model_dir = self.output_dir / "final_model"
        logger.info(f"Copying final model to {final_model_dir}")

        import shutil
        if final_model_dir.exists():
            shutil.rmtree(final_model_dir)
        shutil.copytree(current_model, final_model_dir)

        # Print summary
        logger.info("")
        logger.info("=" * 100)
        logger.info("LOOP COMPLETE - SUMMARY")
        logger.info("=" * 100)

        for result in self.results:
            logger.info(
                f"Iteration {result['iteration']}: "
                f"EER={result['eer']:.4f}, ROC-AUC={result['roc_auc']:.4f}"
            )

        logger.info("=" * 100)
        logger.info(f"Final model saved to: {final_model_dir}")
        logger.info("=" * 100)

        # Log summary table to wandb
        if self.wandb_run:
            import wandb
            summary_table = wandb.Table(
                columns=["Iteration", "EER", "ROC-AUC", "EER Threshold"],
                data=[
                    [r["iteration"], r["eer"], r["roc_auc"], r["eer_threshold"]]
                    for r in self.results
                ]
            )
            self.wandb_run.log({"loop/summary": summary_table})

        return str(final_model_dir), self.results


def main():
    """Main loop function using YAML config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run autonomous loop using YAML configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--base-model",
        help="Override base model path (optional)"
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

    # Initialize experiment tracker
    tracker = ExperimentTracker(config)
    tracker.create_experiment_doc()
    tracker.update_status("running")
    tracker.log_start()

    # Initialize wandb
    wandb_run = None
    if config.wandb.enabled:
        git_info = get_git_info()
        system_info = get_system_info()
        wandb_run = init_wandb_with_metadata(config, git_info, system_info)

    # Run loop
    loop = AutonomousLoop(config, wandb_run, base_model=args.base_model)
    final_model, results = loop.run()

    # Get final metrics
    final_metrics = results[-1] if results else {}

    # Update experiment doc
    if wandb_run:
        tracker.log_results(final_metrics, wandb_run.url)
        wandb_run.finish()
    else:
        tracker.log_results(final_metrics)

    tracker.update_status("complete")

    logger.info(f" Autonomous loop complete! Final model: {final_model}")


if __name__ == "__main__":
    main()
