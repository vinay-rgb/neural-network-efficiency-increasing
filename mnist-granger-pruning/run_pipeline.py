#!/usr/bin/env python3
"""
run_pipeline.py - end-to-end MNIST pruning pipeline.

Usage:
    python run_pipeline.py [--prune-pcts 0 10 20 30 40 50] [--epochs-baseline 10]
"""

import argparse
import copy
import sys
from pathlib import Path

# -- allow running from project root ------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import torch

import src
from src.config import (
    EPOCHS_BASELINE, EPOCHS_FINETUNE, PRUNE_PERCENTAGES,
    RESULTS, MODELS, GRANGER_SAMPLE,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="MNIST Granger-pruning pipeline")
    p.add_argument("--prune-pcts",       nargs="+", type=int,
                   default=PRUNE_PERCENTAGES)
    p.add_argument("--epochs-baseline",  type=int, default=EPOCHS_BASELINE)
    p.add_argument("--epochs-finetune",  type=int, default=EPOCHS_FINETUNE)
    p.add_argument("--dataset",          type=str, default="mnist", choices=["mnist", "car", "sonar"])
    p.add_argument("--granger-sample",   type=int, default=GRANGER_SAMPLE)
    p.add_argument("--no-structural",    action="store_true",
                   help="Use masking fallback instead of structural rebuild")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    src.set_seed()

    print(f"\n{'='*60}")
    print(f"  MNIST Granger-Pruning Pipeline   device={device}")
    print(f"{'='*60}\n")

    # -- 1. Data ---------------------------------------------------------------
    print(f"-- Step 1: Loading {args.dataset.upper()} ------------------------------------")
    train_loader, test_loader = src.get_loaders(dataset_name=args.dataset, batch_size=src.config.BATCH_SIZE)

    input_dim, output_dim = src.config.DATASET_DIMS[args.dataset]

    # -- 2. Baseline training --------------------------------------------------
    print("\n-- Step 2: Baseline training --------------------------------")
    model = src.MLP(input_dim=input_dim, output_dim=output_dim)
    model.to(device)

    print(f"   Parameters: {model.count_parameters():,}")

    history = src.train(
        model, train_loader, test_loader,
        epochs=args.epochs_baseline, device=device,
    )
    baseline_history = history
    src.save_history_to_csv(history, RESULTS / "baseline_history.csv")

    torch.save(model.state_dict(), MODELS / "baseline.pt")
    src.plot_training_curves(history, RESULTS / "training_curves.png")

    metrics_rows: list[dict] = []
    baseline_metrics = src.evaluate_full(
        model, test_loader, device, label="baseline", train_loss=history["train_loss"][-1]
    )
    metrics_rows.append(baseline_metrics)
    print(f"\n   Baseline -> train_loss={baseline_metrics['train_loss']:.4f}  "
          f"test_loss={baseline_metrics['test_loss']:.4f}  "
          f"acc={baseline_metrics['accuracy']:.4f}  "
          f"f1={baseline_metrics['f1_score']:.4f}  "
          f"params={baseline_metrics['n_params']:,}  "
          f"size={baseline_metrics['model_size_kb']:.1f}KB  "
          f"inf={baseline_metrics['inference_ms']:.2f}ms")

    # -- 3. Activation logging -------------------------------------------------
    print("\n-- Step 3: Activation logging -------------------------------")
    src.log_activations(
        model, train_loader, device,
        max_steps=args.granger_sample,
        csv_path=RESULTS / "activations.csv",
    )

    # -- 4. Granger causality --------------------------------------------------
    print("\n-- Step 4: Granger causality --------------------------------")
    granger_df = src.compute_granger_matrix(
        activations_csv=RESULTS / "activations.csv",
        save_path=RESULTS / "granger_matrix_filtered.csv",
    )
    src.plot_granger_heatmap(
        granger_csv=RESULTS / "granger_matrix_filtered.csv",
        save_path=RESULTS / "granger_heatmap.png",
    )

    # -- 5. Importance scoring -------------------------------------------------
    print("\n-- Step 5: Importance scoring -------------------------------")
    importance = src.compute_importance(
        model,
        activations_csv=RESULTS / "activations.csv",
        granger_csv=RESULTS / "granger_matrix_filtered.csv",
    )
    for name, scores in importance.items():
        print(f"   {name}: {len(scores)} neurons  "
              f"mean_score={scores.mean():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")

    # -- 6-7. Prune + retrain loop ---------------------------------------------
    print("\n-- Steps 6-7: Pruning + retraining --------------------------")
    structural = not args.no_structural

    for pct in sorted(args.prune_pcts):
        if pct == 0:
            # already have baseline
            continue

        print(f"\n   -- Prune {pct}% --------------------------")
        # always start from the original trained model
        model_copy = copy.deepcopy(model)

        pruned_model, keep_indices = src.prune_model(
            model_copy, prune_pct=pct,
            importance=importance,
            structural=structural,
        )
        pruned_model.to(device)

        surviving = [len(ki) for ki in keep_indices]
        print(f"   Surviving neurons per layer: {surviving}  "
              f"total params: {pruned_model.count_parameters():,}")

        # Fine-tune
        pruned_history = src.train(
            pruned_model, train_loader, test_loader,
            epochs=args.epochs_finetune, device=device, verbose=True,
        )
        src.save_history_to_csv(pruned_history, RESULTS / f"pruned_{pct}_history.csv")

        torch.save(pruned_model.state_dict(), MODELS / f"pruned_{pct}.pt")

        src.plot_loss_recovery(
            baseline_history, pruned_history, pct, RESULTS / f"loss_recovery_{pct}.png"
        )

        row = src.evaluate_full(
            pruned_model, test_loader, device, label=f"pruned_{pct}",
            train_loss=pruned_history["train_loss"][-1]
        )
        metrics_rows.append(row)
        print(f"   pruned_{pct}% -> train_loss={row['train_loss']:.4f}  "
              f"test_loss={row['test_loss']:.4f}  "
              f"acc={row['accuracy']:.4f}  "
              f"f1={row['f1_score']:.4f}  "
              f"params={row['n_params']:,}  "
              f"size={row['model_size_kb']:.1f}KB  "
              f"inf={row['inference_ms']:.2f}ms")

    # -- 8. Save metrics & plots -----------------------------------------------
    print("\n-- Step 8: Saving metrics & plots ---------------------------")
    src.save_metrics(metrics_rows, RESULTS / "metrics.csv")
    src.plot_accuracy_vs_pruning(
        metrics_csv=RESULTS / "metrics.csv",
        save_path=RESULTS / "accuracy_vs_pruning.png",
    )



    # -- Final summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Pipeline complete. Results:")
    print(f"{'='*60}")
    for row in metrics_rows:
        print(f"  {row['label']:20s}  train_loss={row['train_loss']:.4f}  "
              f"test_loss={row['test_loss']:.4f}  "
              f"acc={row['accuracy']:.4f}  "
              f"f1={row['f1_score']:.4f}  "
              f"params={row['n_params']:6,}  "
              f"size={row['model_size_kb']:6.1f}KB  "
              f"inf={row['inference_ms']:5.2f}ms")
    print(f"\n  Outputs in: {RESULTS.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()



