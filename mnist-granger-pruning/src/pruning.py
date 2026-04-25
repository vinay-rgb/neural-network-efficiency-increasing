"""Neuron importance scoring and structural pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ALPHA, BETA, GAMMA, RESULTS
from .model import MLP


# -- Importance scoring --------------------------------------------------------

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def compute_importance(
    model:           MLP,
    activations_csv: Path = RESULTS / "activations.csv",
    granger_csv:     Path = RESULTS / "granger_matrix.csv",
    alpha:           float = ALPHA,
    beta:            float = BETA,
    gamma:           float = GAMMA,
) -> dict[str, np.ndarray]:
    """Return per-layer importance vectors (normalized, combined score).

    importance = alpha * weight_magnitude + beta * activation_mean + gamma * granger_score
    """
    act_df     = pd.read_csv(activations_csv)
    granger_df = pd.read_csv(granger_csv, index_col=0)

    linears = [m for m in model.net if isinstance(m, nn.Linear)][:-1]  # hidden only

    importance_per_layer: dict[str, np.ndarray] = {}

    for layer_idx, linear in enumerate(linears):
        name = f"hidden_{layer_idx}"

        # 1. Weight magnitude (mean |W| across input dimension)
        W       = linear.weight.data.cpu().numpy()         # [out, in]
        w_mag   = np.abs(W).mean(axis=1)                   # [out]

        # 2. Activation mean (mean activation across all logged steps)
        act_cols = [c for c in act_df.columns if c.startswith(f"{name}_n")]
        act_mean = act_df[act_cols].values.mean(axis=0)    # [neurons]

        # 3. Granger score (mean incoming F-statistic for each neuron)
        layer_cols = [c for c in granger_df.columns if c.startswith(f"{name}_")]
        if layer_cols:
            g_sub   = granger_df.loc[
                [r for r in granger_df.index if r.startswith(f"{name}_")],
                layer_cols,
            ].values
            g_score = g_sub.mean(axis=0)                   # mean influence received
        else:
            g_score = np.zeros(len(w_mag))

        # Align lengths
        n = min(len(w_mag), len(act_mean), len(g_score))
        w_mag, act_mean, g_score = w_mag[:n], act_mean[:n], g_score[:n]

        # Normalize each component to [0, 1]
        w_norm = _minmax(w_mag)
        a_norm = _minmax(np.abs(act_mean))
        g_norm = _minmax(g_score)

        score = alpha * w_norm + beta * a_norm + gamma * g_norm
        importance_per_layer[name] = score

    return importance_per_layer


# -- Pruning -------------------------------------------------------------------

def prune_model(
    model:            MLP,
    prune_pct:        float,
    importance:       dict[str, np.ndarray],
    structural:       bool = True,
) -> tuple[MLP, list[list[int]]]:
    """Remove bottom `prune_pct`% neurons by importance.

    Returns (pruned_model, keep_indices_per_hidden_layer).
    Prefers structural rebuild; falls back to masking if structural=False.
    """
    if prune_pct == 0:
        keep_indices = [list(range(len(v))) for v in importance.values()]
        return model, keep_indices

    keep_indices: list[list[int]] = []

    for name, scores in importance.items():
        n_total   = len(scores)
        n_remove  = int(n_total * prune_pct / 100)
        n_keep    = max(1, n_total - n_remove)   # always keep at least 1 neuron

        # sort descending -> keep top-n_keep
        ranked    = np.argsort(scores)[::-1]
        surviving = sorted(ranked[:n_keep].tolist())
        keep_indices.append(surviving)

    if structural:
        # -- Structural rebuild ---------------------------------------------
        new_hidden_dims = [len(ki) for ki in keep_indices]
        new_model       = MLP.from_dims(new_hidden_dims)
        new_model.copy_weights_from(model, keep_indices)
        return new_model, keep_indices

    else:
        # -- Masking fallback -----------------------------------------------
        linears = [m for m in model.net if isinstance(m, nn.Linear)][:-1]
        with torch.no_grad():
            for layer_idx, (linear, ki) in enumerate(zip(linears, keep_indices)):
                mask = torch.zeros(linear.out_features)
                mask[ki] = 1.0
                linear.weight.data *= mask.unsqueeze(1)
                linear.bias.data   *= mask
        return model, keep_indices



