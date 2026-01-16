#!/usr/bin/env python3
"""
Show status of all experiments.

Usage:
    python scripts/experiment_status.py
"""

import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_v2 import ExperimentConfig


def main():
    print("Experiment Status")
    print("=" * 80)

    # Find all experiment configs
    configs_dir = Path("configs/experiments")
    if not configs_dir.exists():
        print("No experiments found.")
        return

    config_files = sorted(configs_dir.glob("*.yaml"))

    if not config_files:
        print("No experiments found.")
        return

    # Print table header
    print(f"{'ID':<5} {'Name':<30} {'Status':<15} {'EER':<10} {'ROC-AUC':<10}")
    print("-" * 80)

    for config_file in config_files:
        try:
            config = ExperimentConfig.from_yaml(str(config_file))

            # Get metrics if available
            metrics = get_experiment_metrics(config)

            eer = metrics.get("eer", "-")
            if eer != "-":
                eer = f"{eer:.2%}"

            auc = metrics.get("roc_auc", "-")
            if auc != "-":
                auc = f"{auc:.4f}"

            status_emoji = {
                "planning": "",
                "running": "",
                "complete": "",
                "failed": "",
            }.get(config.experiment.status, "")

            print(f"{config.experiment.id:<5} {config.experiment.name:<30} {status_emoji} {config.experiment.status:<13} {eer:<10} {auc:<10}")

        except Exception as e:
            print(f"️  Failed to load {config_file.name}: {e}")

    print("=" * 80)


def get_experiment_metrics(config: ExperimentConfig) -> dict:
    """Extract metrics from experiment doc if available."""
    exp_doc = Path(f"experiments/{config.experiment.id}_{config.experiment.name}.md")

    if not exp_doc.exists():
        return {}

    try:
        content = exp_doc.read_text()

        # Parse metrics from table
        eer_match = re.search(r'\| \*\*EER\*\* \| ([\d.]+)%', content)
        auc_match = re.search(r'\| \*\*ROC-AUC\*\* \| ([\d.]+)', content)

        metrics = {}
        if eer_match:
            metrics["eer"] = float(eer_match.group(1)) / 100
        if auc_match:
            metrics["roc_auc"] = float(auc_match.group(1))

        return metrics
    except Exception:
        return {}


if __name__ == "__main__":
    main()
