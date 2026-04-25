"""Evaluation utilities: accuracy, param count, model size, inference time."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .trainer import evaluate, measure_inference_time
from .config import RESULTS


def get_model_size_kb(model: nn.Module) -> float:
    """Serialize to temp file and return size in KB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    os.unlink(path)
    return size_kb


def evaluate_full(
    model:       nn.Module,
    test_loader: DataLoader,
    device:      torch.device,
    label:       str,
    train_loss:  float = 0.0,
) -> dict:
    """Return a complete metrics dict for one model state."""
    criterion = nn.CrossEntropyLoss()
    test_loss, acc = evaluate(model, test_loader, criterion, device)
    
    # Compute F1 Score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_kb   = get_model_size_kb(model)
    inf_ms    = measure_inference_time(model, test_loader, device)

    return {
        "label":          label,
        "train_loss":     round(train_loss, 4),
        "test_loss":      round(test_loss, 4),
        "accuracy":       round(acc, 4),
        "f1_score":       round(f1, 4),
        "n_params":       n_params,
        "model_size_kb":  round(size_kb, 2),
        "inference_ms":   round(inf_ms, 3),
    }


def save_metrics(rows: list[dict], path: Path = RESULTS / "metrics.csv") -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[metrics] saved {len(rows)} rows -> {path}")



