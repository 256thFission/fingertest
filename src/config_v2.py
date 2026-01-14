#!/usr/bin/env python3
"""
YAML-based configuration system with inheritance and validation.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime


@dataclass
class ExperimentMetadata:
    """Experiment metadata."""
    id: str  # Auto-generated or specified
    name: str
    description: str
    hypothesis: str
    expected_results: Dict[str, Any] = field(default_factory=dict)
    parent_experiment: Optional[str] = None
    status: Literal["planning", "running", "complete", "failed"] = "planning"
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = "data/processed/train.parquet"
    val_path: str = "data/processed/val.parquet"
    test_path: str = "data/processed/test.parquet"
    version: str = "v1.0"
    min_tokens: int = 20
    max_tokens: int = 512


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    base_model: str = "roberta-base"
    max_seq_length: int = 512
    embedding_dim: int = 768
    pooling: str = "mean"


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str = "MultipleNegativesRankingLoss"
    scale: float = 100.0
    temperature: float = 0.01
    margin: Optional[float] = None  # For TripletLoss


@dataclass
class BaselineTrainingConfig:
    """Baseline training configuration."""
    batch_size: int = 64
    num_epochs: int = 1
    learning_rate: float = 2.0e-5
    warmup_steps: int = 1000
    fp16: bool = True
    checkpoint_save_steps: int = 5000
    loss: LossConfig = field(default_factory=LossConfig)
    output_dir: str = "models/baseline"


@dataclass
class TripletTrainingConfig:
    """Triplet training configuration."""
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 1.0e-5
    fp16: bool = True
    warmup_ratio: float = 0.1
    loss: LossConfig = field(default_factory=lambda: LossConfig(
        type="TripletMarginLoss",
        margin=0.5
    ))
    output_dir: str = "models/triplet"


@dataclass
class MiningConfig:
    """Hard negative mining configuration."""
    sample_size: int = 50000
    k_neighbors: int = 10
    batch_size: int = 128
    prioritize_same_channel: bool = True
    min_similarity: float = 0.7
    max_similarity: float = 0.95


@dataclass
class LoopConfig:
    """Autonomous loop configuration."""
    num_iterations: int = 3
    output_dir: str = "models/loop"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    use_whitening: bool = True
    num_positive_pairs: int = 2000
    num_negative_pairs: int = 2000
    target_eer: float = 0.15
    target_roc_auc: float = 0.95
    output_dir: str = "outputs/evaluation"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "authorship-verification"
    entity: str = "thephilliplin-duke-university/Fingerprint"
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_model: bool = True
    log_interval: int = 100

    def __post_init__(self):
        # Allow disabling via environment variable
        if os.getenv("WANDB_DISABLED", "").lower() in ("true", "1"):
            self.enabled = False


@dataclass
class GitConfig:
    """Git configuration."""
    require_clean: bool = False
    auto_commit_results: bool = False


@dataclass
class ReproducibilityConfig:
    """Reproducibility configuration."""
    random_seed: int = 42
    deterministic: bool = True
    log_system_info: bool = True


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    vram_gb: int = 24
    use_gpu: bool = True
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment: ExperimentMetadata
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline_training: Optional[BaselineTrainingConfig] = None
    triplet_training: Optional[TripletTrainingConfig] = None
    mining: Optional[MiningConfig] = None
    loop: Optional[LoopConfig] = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    git: GitConfig = field(default_factory=GitConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Store original path for reference
    _yaml_path: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML with inheritance."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

        # Handle inheritance
        if "base" in config_dict:
            base_path = yaml_path.parent / config_dict.pop("base")
            with open(base_path) as f:
                base_dict = yaml.safe_load(f)
            config_dict = merge_dicts(base_dict, config_dict)

        # Convert nested dicts to dataclasses
        config = cls._from_dict(config_dict)
        config._yaml_path = str(yaml_path)

        return config

    @classmethod
    def _from_dict(cls, d: dict) -> "ExperimentConfig":
        """Convert dict to ExperimentConfig."""
        # Handle nested configs
        if "experiment" in d:
            d["experiment"] = ExperimentMetadata(**d["experiment"])
        if "data" in d:
            d["data"] = DataConfig(**d["data"])
        if "model" in d:
            d["model"] = ModelConfig(**d["model"])
        if "baseline_training" in d:
            if "loss" in d["baseline_training"]:
                d["baseline_training"]["loss"] = LossConfig(**d["baseline_training"]["loss"])
            d["baseline_training"] = BaselineTrainingConfig(**d["baseline_training"])
        if "triplet_training" in d:
            if "loss" in d["triplet_training"]:
                d["triplet_training"]["loss"] = LossConfig(**d["triplet_training"]["loss"])
            d["triplet_training"] = TripletTrainingConfig(**d["triplet_training"])
        if "mining" in d:
            d["mining"] = MiningConfig(**d["mining"])
        if "loop" in d:
            d["loop"] = LoopConfig(**d["loop"])
        if "evaluation" in d:
            d["evaluation"] = EvaluationConfig(**d["evaluation"])
        if "wandb" in d:
            d["wandb"] = WandbConfig(**d["wandb"])
        if "git" in d:
            d["git"] = GitConfig(**d["git"])
        if "reproducibility" in d:
            d["reproducibility"] = ReproducibilityConfig(**d["reproducibility"])
        if "hardware" in d:
            d["hardware"] = HardwareConfig(**d["hardware"])

        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for saving/logging)."""
        return asdict(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for wandb logging."""
        flat = {}

        def flatten(d: dict, prefix: str = ""):
            for key, value in d.items():
                if key.startswith("_"):
                    continue
                full_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, dict):
                    flatten(value, f"{full_key}.")
                elif isinstance(value, list):
                    flat[full_key] = str(value)
                else:
                    flat[full_key] = value

        flatten(self.to_dict())
        return flat

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Check required fields
        if not self.experiment.id:
            errors.append("experiment.id is required")
        if not self.experiment.name:
            errors.append("experiment.name is required")
        if not self.experiment.hypothesis:
            errors.append("experiment.hypothesis is required")

        # Check paths exist
        for path_name in ["train_path", "val_path", "test_path"]:
            path = getattr(self.data, path_name)
            if not Path(path).exists():
                errors.append(f"data.{path_name} does not exist: {path}")

        # Check at least one training config
        if not any([self.baseline_training, self.triplet_training, self.loop]):
            errors.append("At least one training config required (baseline_training, triplet_training, or loop)")

        return errors


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dicts, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(config: ExperimentConfig, overrides: List[str]) -> ExperimentConfig:
    """Apply CLI overrides to config.

    Args:
        config: ExperimentConfig to modify
        overrides: List of "key=value" strings

    Example:
        overrides = ["baseline_training.batch_size=16", "wandb.enabled=false"]
    """
    config_dict = config.to_dict()

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Use key=value")

        key, value = override.split("=", 1)

        # Parse value
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        # Set nested key
        keys = key.split(".")
        target = config_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    return ExperimentConfig._from_dict(config_dict)
