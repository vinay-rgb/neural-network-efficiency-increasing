"""Plotting utilities: accuracy-vs-pruning and Granger heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RESULTS


# -- style ---------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def plot_accuracy_vs_pruning(
    metrics_csv: Path = RESULTS / "metrics.csv",
    save_path:   Path = RESULTS / "accuracy_vs_pruning.png",
) -> None:
    """Line chart of accuracy as a function of pruning percentage."""
    df = pd.read_csv(metrics_csv)

    # keep only pruning rows
    pruning_rows = df[df["label"].str.startswith("pruned_")].copy()
    baseline_row = df[df["label"] == "baseline"]

    if pruning_rows.empty:
        print("[plot] no pruning rows found, skipping accuracy-vs-pruning plot")
        return

    pruning_rows["pct"] = pruning_rows["label"].str.extract(r"pruned_(\d+)").astype(int)
    pruning_rows = pruning_rows.sort_values("pct")

    baseline_acc = baseline_row["accuracy"].values[0] if not baseline_row.empty else None

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        pruning_rows["pct"],
        pruning_rows["accuracy"] * 100,
        marker="o", linewidth=2.5, color="#2d6a9f", label="Pruned + retrained",
    )

    if baseline_acc is not None:
        ax.axhline(
            baseline_acc * 100, color="#e05c2d", linestyle="--",
            linewidth=1.8, label=f"Baseline ({baseline_acc*100:.2f}%)",
        )

    ax.set_xlabel("Neurons pruned (%)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Pruning Percentage", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] accuracy_vs_pruning -> {save_path}")


def plot_granger_heatmap(
    granger_csv: Path = RESULTS / "granger_matrix.csv",
    save_path:   Path = RESULTS / "granger_heatmap.png",
    max_neurons: int  = 40,
) -> None:
    """Heatmap of the Granger F-statistic matrix (clipped to max_neurons)."""
    df = pd.read_csv(granger_csv, index_col=0)

    # Clip to max_neurons for readability
    df = df.iloc[:max_neurons, :max_neurons]

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0,
        cbar_kws={"label": "F-statistic", "shrink": 0.8},
        xticklabels=False,
        yticklabels=False,
    )

    ax.set_title(
        f"Granger Causality Heatmap (first {len(df)} neurons)\n"
        "Row i -> Col j: influence of neuron i on neuron j",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Target neuron", fontsize=10)
    ax.set_ylabel("Source neuron", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] granger_heatmap -> {save_path}")


def plot_training_curves(
    history:   dict,
    save_path: Path = RESULTS / "training_curves.png",
) -> None:
    """Loss and accuracy curves from training history dict."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train", color="#2d6a9f")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="#e05c2d")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [v * 100 for v in history["train_acc"]], label="Train", color="#2d6a9f")
    ax2.plot(epochs, [v * 100 for v in history["val_acc"]],   label="Val",   color="#e05c2d")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Baseline Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] training_curves -> {save_path}")

def plot_loss_recovery(
    baseline_history: dict,
    pruned_history: dict,
    pct: int,
    save_path: Path,
) -> None:
    """Plot baseline vs fine-tuning loss recovery for a specific pruned model."""
    b_loss = baseline_history["train_loss"]
    b_val  = baseline_history["val_loss"]
    p_loss = pruned_history["train_loss"]
    p_val  = pruned_history["val_loss"]
    
    b_epochs = list(range(1, len(b_loss) + 1))
    p_epochs = list(range(len(b_loss) + 1, len(b_loss) + len(p_loss) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b_epochs, b_loss, marker="o", label="Baseline Train Loss", color="#2d6a9f")
    ax.plot(b_epochs, b_val, marker="s", label="Baseline Val Loss", color="#4a90e2", linestyle=":")
    ax.plot(p_epochs, p_loss, marker="o", label=f"Pruned {pct}% Train Loss", color="#e05c2d", linestyle="--")
    ax.plot(p_epochs, p_val, marker="s", label=f"Pruned {pct}% Val Loss", color="#f5a623", linestyle="-.")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Loss Recovery after {pct}% Pruning", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="both", alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plot] loss_recovery_{pct} -> {save_path}")



