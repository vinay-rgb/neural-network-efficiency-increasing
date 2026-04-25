"""Granger causality between neuron activation time series."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from .config import GRANGER_MAX_LAG, GRANGER_SAMPLE, RESULTS


def _ols_rss(y: np.ndarray, X: np.ndarray) -> float:
    """OLS residual sum of squares (no intercept needed; X already includes it)."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(resid @ resid)


def granger_pairwise(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = GRANGER_MAX_LAG,
) -> tuple[float, float]:
    """Returns (F-statistic, p-value) for 'x Granger-causes y'.

    Higher F => stronger causal influence of x on y.
    Returns (0.0, 1.0) on failure.
    """
    T = len(y)
    if T <= 2 * max_lag + 1:
        return 0.0, 1.0

    try:
        # Build lagged matrices
        Y  = y[max_lag:]
        n  = len(Y)

        # Restricted: only lags of y
        Xr = np.column_stack(
            [np.ones(n)] + [y[max_lag - k: T - k] for k in range(1, max_lag + 1)]
        )
        # Unrestricted: lags of y + lags of x
        Xu = np.column_stack(
            [Xr] + [x[max_lag - k: T - k] for k in range(1, max_lag + 1)]
        )

        rss_r = _ols_rss(Y, Xr)
        rss_u = _ols_rss(Y, Xu)

        if rss_u < 1e-12:
            return 0.0, 1.0

        df1 = max_lag
        df2 = n - Xu.shape[1]
        if df2 <= 0:
            return 0.0, 1.0

        F = ((rss_r - rss_u) / df1) / (rss_u / df2)
        F_val = float(max(F, 0.0))
        p_val = float(f_dist.sf(F_val, df1, df2))
        return F_val, p_val

    except Exception:
        return 0.0, 1.0


def compute_granger_matrix(
    activations_csv: Path = RESULTS / "activations.csv",
    max_lag:         int  = GRANGER_MAX_LAG,
    sample:          int  = GRANGER_SAMPLE,
    save_path:       Path = RESULTS / "granger_matrix.csv",
) -> pd.DataFrame:
    """Load activation CSV, compute NxN Granger matrix, save and return it."""

    df = pd.read_csv(activations_csv)

    # Keep only hidden-layer neuron columns
    neuron_cols = [c for c in df.columns if c.startswith("hidden_")]
    if not neuron_cols:
        raise ValueError("No hidden-layer activation columns found in activations CSV.")

    # Subsample rows to keep computation tractable
    data = df[neuron_cols].values[:sample].astype(np.float32)
    T, N = data.shape

    print(f"[granger] computing {N}x{N} matrix on {T} timesteps ...")

    matrix = np.zeros((N, N), dtype=np.float32)

    significant_count = 0
    total_tests = N * (N - 1)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            f_stat, p_val = granger_pairwise(data[:, i], data[:, j], max_lag)
            if p_val < 0.05:
                matrix[i, j] = f_stat
                significant_count += 1
            else:
                matrix[i, j] = 0.0

    print(f"[granger] Statistical filtering (p < 0.05) kept {significant_count}/{total_tests} causal links ({(significant_count/total_tests)*100:.1f}%)")

    granger_df = pd.DataFrame(matrix, index=neuron_cols, columns=neuron_cols)
    granger_df.to_csv(save_path)
    print(f"[granger] saved {N}x{N} filtered matrix -> {save_path}")
    return granger_df



