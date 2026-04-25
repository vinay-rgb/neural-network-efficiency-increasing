"""Global configuration and reproducibility settings."""

import random
import numpy as np
import torch

# -- Reproducibility ------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -- Paths ----------------------------------------------------------------------
import pathlib

ROOT      = pathlib.Path(__file__).parent.parent
RESULTS   = ROOT / "results"
MODELS    = ROOT / "results" / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# -- Hyper-parameters -----------------------------------------------------------
BATCH_SIZE        = 256
EPOCHS_BASELINE   = 10
EPOCHS_FINETUNE   = 5
LEARNING_RATE     = 1e-3

HIDDEN_DIMS       = [128, 64]   # layer widths
INPUT_DIM         = 784
OUTPUT_DIM        = 10

DATASET_DIMS = {
    "mnist": (784, 10),
    "car": (21, 4),    # 21 one-hot encoded features, 4 classes
    "sonar": (60, 2),  # 60 features, 2 classes
}

# -- Importance weights ---------------------------------------------------------
ALPHA = 0.4   # weight magnitude
BETA  = 0.3   # activation mean
GAMMA = 0.3   # granger score

# -- Pruning --------------------------------------------------------------------
PRUNE_PERCENTAGES = [0, 10, 20, 30, 40, 50]   # % neurons removed

# -- Granger causality ---------------------------------------------------------
GRANGER_MAX_LAG = 3
GRANGER_SAMPLE  = 200   # number of steps to use (keep memory sane)



