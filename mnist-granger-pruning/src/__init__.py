"""MNIST pruning pipeline - src package."""

from .config      import set_seed, SEED
from .data        import get_loaders
from .model       import MLP
from .trainer     import train, log_activations, save_history_to_csv
from .granger     import compute_granger_matrix
from .pruning     import compute_importance, prune_model
from .evaluation  import evaluate_full, save_metrics
from .visualization import (
    plot_accuracy_vs_pruning,
    plot_granger_heatmap,
    plot_training_curves,
    plot_loss_recovery,
)

__all__ = [
    "set_seed", "SEED",
    "get_loaders",
    "MLP",
    "train", "log_activations", "save_history_to_csv",
    "compute_granger_matrix",
    "compute_importance", "prune_model",
    "evaluate_full", "save_metrics",
    "plot_accuracy_vs_pruning", "plot_granger_heatmap", "plot_training_curves", "plot_loss_recovery",
]



