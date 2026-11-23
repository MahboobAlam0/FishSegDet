"""Optional seeding for repeatable training runs (train.py --seed)."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive each worker's NumPy and Python seeds
    from the torch seed PyTorch already assigns to that worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
