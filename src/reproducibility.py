#!/usr/bin/env python3
"""
Reproducibility utilities for deterministic experiments.
"""

import os
import random
import logging
import platform
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_random_seeds(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    logger.info(f"Setting random seed: {seed} (deterministic={deterministic})")

    random.seed(seed)

    if NUMPY_AVAILABLE:
        np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            logger.info("Enabled deterministic mode (may be slower)")
    else:
        logger.warning("PyTorch not available, skipping torch seed")


def get_system_info() -> Dict[str, any]:
    """Get comprehensive system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    if TORCH_AVAILABLE:
        info.update({
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })

    if NUMPY_AVAILABLE:
        info["numpy_version"] = np.__version__

    return info


def log_environment():
    """Log complete environment information."""
    info = get_system_info()
    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
