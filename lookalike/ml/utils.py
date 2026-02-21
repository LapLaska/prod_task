import logging
import random
from pathlib import Path
import numpy as np
import torch


LOGGER = logging.getLogger("lookalike_ml")
logging.basicConfig(level=logging.INFO)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        LOGGER.warning("CUDA is available and will be used for text embeddings.")
        return device
    LOGGER.warning("CUDA is not available, falling back to CPU for text embeddings.")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
