#!/usr/bin/env python3
"""
Centralized Configuration Management
Provides easy-to-modify configuration for all training parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import configparser
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration."""

    min_tokens: int = 20
    max_tokens: int = 512
    time_window_seconds: int = 300
    min_blocks_per_author: int = 5
    test_authors: int = 1000

    # Data paths
    train_data: str = "data/processed/train.parquet"
    val_data: str = "data/processed/val.parquet"
    test_data: str = "data/processed/test.parquet"


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    base_model: str = "roberta-base"
    max_seq_length: int = 512
    embedding_dim: int = 768


@dataclass
class BaselineTrainingConfig:
    """Baseline training (Phase 2) configuration."""

    batch_size: Optional[int] = None  # Auto-calculated if None
    num_epochs: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    fp16: bool = True
    checkpoint_save_steps: int = 5000
    checkpoint_save_total_limit: int = 3

    # Output
    output_dir: str = "models/baseline"


@dataclass
class TripletTrainingConfig:
    """Triplet training (Phase 3B) configuration."""

    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 1e-5
    margin: float = 0.5
    fp16: bool = True
    warmup_ratio: float = 0.1

    # Output
    output_dir: str = "models/triplet_refined"


@dataclass
class MiningConfig:
    """Hard negative mining (Phase 3A) configuration."""

    sample_size: int = 50000
    k_neighbors: int = 10
    batch_size: int = 128
    prioritize_same_channel: bool = True


@dataclass
class LoopConfig:
    """Autonomous loop (Phase 3) configuration."""

    num_iterations: int = 3
    base_model_dir: str = "models/baseline"
    output_dir: str = "models/loop"


@dataclass
class EvaluationConfig:
    """Evaluation (Phase 4) configuration."""

    num_positive_pairs: int = 2000
    num_negative_pairs: int = 2000
    target_eer: float = 0.05
    target_roc_auc: float = 0.95


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    umap_authors: int = 50
    samples_per_author: int = 10


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    vram_gb: int = 24
    use_gpu: bool = True
    num_workers: int = 4


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "authorship-verification"
    entity: Optional[str] = None  # Your wandb username/team

    # Run settings
    name: Optional[str] = None  # Auto-generated if None
    tags: list = field(default_factory=list)
    notes: Optional[str] = None

    # Logging settings
    log_model: bool = True  # Upload model checkpoints
    log_interval: int = 100  # Log every N steps

    # Environment variable override
    def __post_init__(self):
        # Allow disabling via environment variable
        if os.getenv("WANDB_DISABLED", "").lower() in ("true", "1"):
            self.enabled = False

        # Override project/entity from env
        if os.getenv("WANDB_PROJECT"):
            self.project = os.getenv("WANDB_PROJECT")
        if os.getenv("WANDB_ENTITY"):
            self.entity = os.getenv("WANDB_ENTITY")


@dataclass
class Config:
    """Main configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    baseline: BaselineTrainingConfig = field(default_factory=BaselineTrainingConfig)
    triplet: TripletTrainingConfig = field(default_factory=TripletTrainingConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_ini(cls, ini_path: str = "config.ini"):
        """Load configuration from INI file."""
        config = cls()

        if not Path(ini_path).exists():
            return config

        parser = configparser.ConfigParser()
        parser.read(ini_path)

        # Data
        if "data" in parser:
            section = parser["data"]
            config.data.min_tokens = section.getint(
                "min_tokens", config.data.min_tokens
            )
            config.data.max_tokens = section.getint(
                "max_tokens", config.data.max_tokens
            )
            config.data.time_window_seconds = section.getint(
                "time_window_seconds", config.data.time_window_seconds
            )
            config.data.min_blocks_per_author = section.getint(
                "min_blocks_per_author", config.data.min_blocks_per_author
            )
            config.data.test_authors = section.getint(
                "test_authors", config.data.test_authors
            )

        # Model
        if "model" in parser:
            section = parser["model"]
            config.model.base_model = section.get("base_model", config.model.base_model)
            config.model.max_seq_length = section.getint(
                "max_seq_length", config.model.max_seq_length
            )
            config.model.embedding_dim = section.getint(
                "embedding_dim", config.model.embedding_dim
            )

        # Baseline
        if "baseline" in parser:
            section = parser["baseline"]
            config.baseline.batch_size = section.getint(
                "batch_size", config.baseline.batch_size or 64
            )
            config.baseline.num_epochs = section.getint(
                "num_epochs", config.baseline.num_epochs
            )
            config.baseline.learning_rate = section.getfloat(
                "learning_rate", config.baseline.learning_rate
            )
            config.baseline.fp16 = section.getboolean("fp16", config.baseline.fp16)

        # Triplet
        if "triplet" in parser:
            section = parser["triplet"]
            config.triplet.batch_size = section.getint(
                "batch_size", config.triplet.batch_size
            )
            config.triplet.num_epochs = section.getint(
                "num_epochs", config.triplet.num_epochs
            )
            config.triplet.learning_rate = section.getfloat(
                "learning_rate", config.triplet.learning_rate
            )
            config.triplet.margin = section.getfloat("margin", config.triplet.margin)
            config.triplet.fp16 = section.getboolean("fp16", config.triplet.fp16)

        # Mining
        if "mining" in parser:
            section = parser["mining"]
            config.mining.sample_size = section.getint(
                "sample_size", config.mining.sample_size
            )
            config.mining.k_neighbors = section.getint(
                "k_neighbors", config.mining.k_neighbors
            )
            config.mining.batch_size = section.getint(
                "batch_size", config.mining.batch_size
            )
            config.mining.prioritize_same_channel = section.getboolean(
                "prioritize_same_channel", config.mining.prioritize_same_channel
            )

        # Loop
        if "loop" in parser:
            section = parser["loop"]
            config.loop.num_iterations = section.getint(
                "num_iterations", config.loop.num_iterations
            )

        # Evaluation
        if "evaluation" in parser:
            section = parser["evaluation"]
            config.evaluation.num_positive_pairs = section.getint(
                "num_positive_pairs", config.evaluation.num_positive_pairs
            )
            config.evaluation.num_negative_pairs = section.getint(
                "num_negative_pairs", config.evaluation.num_negative_pairs
            )
            config.evaluation.target_eer = section.getfloat(
                "target_eer", config.evaluation.target_eer
            )
            config.evaluation.target_roc_auc = section.getfloat(
                "target_roc_auc", config.evaluation.target_roc_auc
            )

        # Visualization
        if "visualization" in parser:
            section = parser["visualization"]
            config.visualization.umap_authors = section.getint(
                "umap_authors", config.visualization.umap_authors
            )
            config.visualization.samples_per_author = section.getint(
                "samples_per_author", config.visualization.samples_per_author
            )

        # Hardware
        if "hardware" in parser:
            section = parser["hardware"]
            config.hardware.vram_gb = section.getint("vram_gb", config.hardware.vram_gb)
            config.hardware.use_gpu = section.getboolean(
                "use_gpu", config.hardware.use_gpu
            )
            config.hardware.num_workers = section.getint(
                "num_workers", config.hardware.num_workers
            )

        # Wandb
        if "wandb" in parser:
            section = parser["wandb"]
            config.wandb.enabled = section.getboolean("enabled", config.wandb.enabled)
            config.wandb.project = section.get("project", config.wandb.project)
            config.wandb.entity = section.get("entity", config.wandb.entity) or None
            config.wandb.log_model = section.getboolean(
                "log_model", config.wandb.log_model
            )
            config.wandb.log_interval = section.getint(
                "log_interval", config.wandb.log_interval
            )

        return config

    def to_dict(self):
        """Convert config to dictionary for logging."""
        from dataclasses import asdict

        return asdict(self)


# Global config instance
_config: Optional[Config] = None


def get_config(ini_path: str = "config.ini") -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config.from_ini(ini_path)
    return _config


def reload_config(ini_path: str = "config.ini") -> Config:
    """Force reload config from INI file."""
    global _config
    _config = Config.from_ini(ini_path)
    return _config


# Convenience functions for quick access
def get_data_config() -> DataConfig:
    return get_config().data


def get_model_config() -> ModelConfig:
    return get_config().model


def get_baseline_config() -> BaselineTrainingConfig:
    return get_config().baseline


def get_triplet_config() -> TripletTrainingConfig:
    return get_config().triplet


def get_mining_config() -> MiningConfig:
    return get_config().mining


def get_loop_config() -> LoopConfig:
    return get_config().loop


def get_evaluation_config() -> EvaluationConfig:
    return get_config().evaluation


def get_wandb_config() -> WandbConfig:
    return get_config().wandb


def get_hardware_config() -> HardwareConfig:
    return get_config().hardware
