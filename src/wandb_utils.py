#!/usr/bin/env python3
"""
Enhanced Wandb integration with full metadata logging.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Wandb not available. Install with: pip install wandb")


def init_wandb_with_metadata(
    config: "ExperimentConfig",
    git_info: dict,
    system_info: dict
) -> Optional["wandb.Run"]:
    """Initialize wandb with full experiment metadata."""
    if not WANDB_AVAILABLE or not config.wandb.enabled:
        return None

    # Create run name
    run_name = f"exp-{config.experiment.id}-{config.experiment.name}"

    # Prepare tags
    tags = [
        f"exp-{config.experiment.id}",
        *config.wandb.tags,
    ]

    # Add parent tag if exists
    if config.experiment.parent_experiment:
        tags.append(f"parent-{config.experiment.parent_experiment}")

    # Initialize wandb
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        tags=tags,
        group=config.wandb.group,
        notes=config.experiment.hypothesis,
        config={
            # Experiment metadata
            "experiment_id": config.experiment.id,
            "experiment_name": config.experiment.name,
            "experiment_description": config.experiment.description,
            "hypothesis": config.experiment.hypothesis,
            "expected_results": config.experiment.expected_results,
            "parent_experiment": config.experiment.parent_experiment,
            "status": config.experiment.status,

            # Reproducibility
            "git_hash": git_info.get("hash"),
            "git_hash_short": git_info.get("hash_short"),
            "git_branch": git_info.get("branch"),
            "git_remote": git_info.get("remote"),
            "git_dirty": git_info.get("dirty", False),
            "git_untracked": git_info.get("untracked", False),

            # System info
            **{f"system_{k}": v for k, v in system_info.items()},

            # Data version
            "data_version": config.data.version,
            "data_train_path": config.data.train_path,
            "data_val_path": config.data.val_path,
            "data_test_path": config.data.test_path,

            # Config file
            "config_file": config._yaml_path,

            # All config params (flattened)
            **config.to_flat_dict(),
        }
    )

    logger.info(f"Wandb run initialized: {run.name}")
    logger.info(f"Wandb URL: {run.url}")

    # Log config as artifact
    log_config_artifact(run, config)

    return run


def log_config_artifact(run: "wandb.Run", config: "ExperimentConfig"):
    """Log config file as wandb artifact."""
    if not WANDB_AVAILABLE or not config.wandb.log_model:
        return

    try:
        artifact = wandb.Artifact(
            f"config-exp-{config.experiment.id}",
            type="config",
            description=f"Configuration for experiment {config.experiment.id}",
            metadata={
                "experiment_id": config.experiment.id,
                "experiment_name": config.experiment.name,
                "hypothesis": config.experiment.hypothesis,
            }
        )

        # Add config YAML if path exists
        if config._yaml_path and Path(config._yaml_path).exists():
            artifact.add_file(config._yaml_path)

        # Add experiment doc if exists
        exp_doc = Path(f"experiments/{config.experiment.id}_{config.experiment.name}.md")
        if exp_doc.exists():
            artifact.add_file(str(exp_doc))

        run.log_artifact(artifact)
        logger.info("Config artifact logged to wandb")

    except Exception as e:
        logger.warning(f"Failed to log config artifact: {e}")


def log_evaluation_artifacts(
    run: "wandb.Run",
    output_dir: Path,
    metrics: Dict[str, Any]
):
    """Log evaluation results as artifacts."""
    if not WANDB_AVAILABLE:
        return

    try:
        artifact = wandb.Artifact(
            f"evaluation-exp-{metrics.get('experiment_id', 'unknown')}",
            type="evaluation",
            description="Evaluation results and visualizations",
            metadata=metrics
        )

        # Add plots
        plot_files = [
            "roc_curve.png",
            "far_frr_curves.png",
            "score_distribution.png",
            "umap_visualization.png"
        ]

        for plot_file in plot_files:
            plot_path = output_dir / plot_file
            if plot_path.exists():
                artifact.add_file(str(plot_path))
                # Also log as image to wandb UI
                run.log({plot_file.replace(".png", ""): wandb.Image(str(plot_path))})

        # Add metrics JSON
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists():
            artifact.add_file(str(metrics_path))

        run.log_artifact(artifact)
        logger.info("Evaluation artifacts logged to wandb")

    except Exception as e:
        logger.warning(f"Failed to log evaluation artifacts: {e}")


def log_model_artifact(
    run: "wandb.Run",
    model_dir: Path,
    experiment_id: str,
    artifact_type: str = "model"
):
    """Log model as wandb artifact."""
    if not WANDB_AVAILABLE or not model_dir.exists():
        return

    try:
        artifact = wandb.Artifact(
            f"{artifact_type}-exp-{experiment_id}",
            type=artifact_type,
            description=f"Trained model for experiment {experiment_id}"
        )

        # Add model directory
        artifact.add_dir(str(model_dir))

        run.log_artifact(artifact)
        logger.info(f"Model artifact logged to wandb: {model_dir}")

    except Exception as e:
        logger.warning(f"Failed to log model artifact: {e}")
