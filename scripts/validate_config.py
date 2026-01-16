#!/usr/bin/env python3
"""
Validate experiment config before running.

Usage:
    python scripts/validate_config.py configs/experiments/002_lower_temperature.yaml
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_v2 import ExperimentConfig
from src.git_utils import get_git_info


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    print(f"Validating: {config_path}")
    print("=" * 80)

    # Load config
    try:
        config = ExperimentConfig.from_yaml(config_path)
        print(" Config loaded successfully")
    except Exception as e:
        print(f" Failed to load config: {e}")
        sys.exit(1)

    # Validate
    errors = config.validate()
    if errors:
        print("\n Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(" Config is valid")

    # Check paths
    print("\n" + "=" * 80)
    print("Checking file paths...")
    print("=" * 80)

    path_checks = [
        ("Train data", config.data.train_path),
        ("Val data", config.data.val_path),
        ("Test data", config.data.test_path),
    ]

    all_paths_exist = True
    for name, path in path_checks:
        if Path(path).exists():
            print(f"   {name}: {path}")
        else:
            print(f"   {name}: {path} (NOT FOUND)")
            all_paths_exist = False

    # Check git
    print("\n" + "=" * 80)
    print("Checking git status...")
    print("=" * 80)

    try:
        git_info = get_git_info()
        print(f"  Branch: {git_info['branch']}")
        print(f"  Commit: {git_info['hash_short']}")

        if git_info.get('dirty'):
            print(f"  ️  Uncommitted changes detected")
            if config.git.require_clean:
                print(f"   Config requires clean git (git.require_clean: true)")
                all_paths_exist = False
        else:
            print(f"   Clean git state")

        if git_info.get('untracked'):
            print(f"  ️  Untracked files detected")
    except Exception as e:
        print(f"  ️  Git info unavailable: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"  Experiment ID: {config.experiment.id}")
    print(f"  Experiment name: {config.experiment.name}")
    print(f"  Status: {config.experiment.status}")

    if config.baseline_training:
        print(f"  Training: Baseline")
        print(f"    - Batch size: {config.baseline_training.batch_size}")
        print(f"    - Epochs: {config.baseline_training.num_epochs}")
        print(f"    - Loss: {config.baseline_training.loss.type}")

    if config.triplet_training:
        print(f"  Training: Triplet")
        print(f"    - Batch size: {config.triplet_training.batch_size}")
        print(f"    - Epochs: {config.triplet_training.num_epochs}")

    if config.loop:
        print(f"  Training: Autonomous Loop")
        print(f"    - Iterations: {config.loop.num_iterations}")

    print(f"  Wandb: {'enabled' if config.wandb.enabled else 'disabled'}")
    print(f"  Data version: {config.data.version}")

    print("\n" + "=" * 80)

    if all_paths_exist and not errors:
        print(" Config ready to use!")
        print(f"\nRun:")
        print(f"  python train_baseline.py --config {config_path}")
        sys.exit(0)
    else:
        print(" Config has issues, please fix before running")
        sys.exit(1)


if __name__ == "__main__":
    main()
