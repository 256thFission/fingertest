#!/usr/bin/env python3
"""
Create a new experiment with auto-incremented ID.

Usage:
    python scripts/new_experiment.py --name distance_filtering
    python scripts/new_experiment.py --name my_experiment --parent 002
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import re

def get_next_experiment_id() -> str:
    """Find next available experiment ID by scanning configs."""
    configs_dir = Path("configs/experiments")
    if not configs_dir.exists():
        return "001"

    # Find all existing IDs
    existing_ids = []
    for config_file in configs_dir.glob("*.yaml"):
        match = re.match(r'(\d{3})_', config_file.name)
        if match:
            existing_ids.append(int(match.group(1)))

    if not existing_ids:
        return "001"

    next_id = max(existing_ids) + 1
    return f"{next_id:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Create new experiment with auto-incremented ID"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name"
    )
    parser.add_argument(
        "--description",
        help="Short description"
    )
    parser.add_argument(
        "--hypothesis",
        help="Experiment hypothesis"
    )
    parser.add_argument(
        "--parent",
        help="Parent experiment ID"
    )
    parser.add_argument(
        "--preset",
        choices=["base", "test", "debug", "full_training"],
        default="base",
        help="Preset to base config on"
    )
    args = parser.parse_args()

    # Get next ID
    exp_id = get_next_experiment_id()

    # Check if experiment with this name already exists
    config_path = Path(f"configs/experiments/{exp_id}_{args.name}.yaml")
    if config_path.exists():
        print(f" Experiment already exists: {config_path}")
        sys.exit(1)

    # Determine base config
    if args.preset == "base":
        base_config = "../base.yaml"
    else:
        base_config = f"../presets/{args.preset}.yaml"

    # Create config
    config = {
        "base": base_config,
        "experiment": {
            "id": exp_id,
            "name": args.name,
            "description": args.description or f"Experiment {exp_id}: {args.name}",
            "hypothesis": args.hypothesis or "TODO: Document hypothesis",
            "expected_results": {
                "eer": "TODO",
                "roc_auc": "TODO",
            },
            "parent_experiment": args.parent,
            "status": "planning",
        },
        # Users can add specific config overrides below
    }

    # Save config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f" Created config: {config_path}")

    # Create experiment doc
    doc_path = Path(f"experiments/{exp_id}_{args.name}.md")
    doc_content = create_experiment_doc(config)
    with open(doc_path, "w") as f:
        f.write(doc_content)

    print(f" Created doc: {doc_path}")

    # Update experiment log
    update_experiment_log(config)

    print(f"\n Experiment {exp_id} created!")
    print(f"\n Next steps:")
    print(f"1. Edit config: {config_path}")
    print(f"2. Update hypothesis in: {doc_path}")
    print(f"3. Run training:")
    print(f"   python train_baseline.py --config {config_path}")


def create_experiment_doc(config: dict) -> str:
    """Create initial experiment doc."""
    exp = config["experiment"]
    date = datetime.now().strftime("%Y-%m-%d")

    return f"""# Experiment {exp['id']}: {exp['name']}

**Date:** {date}
**Status:** {exp['status']}
**Parent:** {exp.get('parent_experiment') or 'None'}

## Hypothesis

{exp['hypothesis']}

## Description

{exp['description']}

## Expected Results

{format_expected_results(exp['expected_results'])}

## Configuration

**Config file:** `configs/experiments/{exp['id']}_{exp['name']}.yaml`

## Results

*Results will be automatically updated after training completes.*

## Notes

"""


def format_expected_results(results: dict) -> str:
    """Format expected results."""
    lines = []
    for key, value in results.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def update_experiment_log(config: dict):
    """Add new experiment to log."""
    log_path = Path("experiments/README.md")
    if not log_path.exists():
        return

    exp = config["experiment"]
    date = datetime.now().strftime("%Y-%m-%d")
    new_row = f"| {exp['id']} | {date} | [{exp['name']}]({exp['id']}_{exp['name']}.md) |  Planning | - | - |"

    content = log_path.read_text()

    # Find table and insert after header
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("| ID |"):
            # Insert after separator line
            if i + 1 < len(lines):
                lines.insert(i + 2, new_row)
                break

    log_path.write_text("\n".join(lines))
    print(f" Updated experiment log")


if __name__ == "__main__":
    main()
