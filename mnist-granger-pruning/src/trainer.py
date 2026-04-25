"""Training loop, evaluation, and activation logging."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import LEARNING_RATE, RESULTS


# -- Training ------------------------------------------------------------------

def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> tuple[float, float]:
    """Single training epoch -> (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += logits.argmax(1).eq(y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Evaluate -> (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct    += logits.argmax(1).eq(y).sum().item()
            total      += x.size(0)

    return total_loss / total, correct / total


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    epochs:       int,
    device:       torch.device,
    lr:           float = LEARNING_RATE,
    verbose:      bool  = True,
) -> dict:
    """Full training run -> metrics dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"
            )

    return history


def save_history_to_csv(history: dict, csv_path: Path) -> None:
    """Save training history (loss, acc) per epoch to a CSV."""
    import pandas as pd
    df = pd.DataFrame(history)
    df.index.name = "epoch"
    df.index += 1
    df.to_csv(csv_path)
    print(f"[trainer] saved history ({len(df)} epochs) -> {csv_path}")


# -- Activation logging --------------------------------------------------------

class ActivationLogger:
    """Log per-step activations, logits, and loss to a CSV file.

    Streams rows to disk to avoid memory blow-up.
    """

    def __init__(self, csv_path: Path = RESULTS / "activations.csv") -> None:
        self.csv_path = csv_path
        self._file: Optional[object]   = None
        self._writer: Optional[object] = None
        self._hooks: list = []

        self._step          = 0
        self._activations   : dict[str, list[float]] = {}

    # -- context manager -------------------------------------------------------
    def __enter__(self) -> "ActivationLogger":
        self._file   = open(self.csv_path, "w", newline="")
        return self

    def __exit__(self, *_) -> None:
        for h in self._hooks:
            h.remove()
        if self._file:
            self._file.close()

    # -- hooks -----------------------------------------------------------------
    def register_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to all Linear layers (hidden only)."""
        linears = [m for m in model.net if isinstance(m, nn.Linear)]
        for idx, layer in enumerate(linears[:-1]):   # exclude output layer
            name = f"hidden_{idx}"
            h = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # store mean activation per neuron (shape: [batch, neurons])
            self._activations[name] = output.detach().mean(0).cpu().tolist()
        return hook

    # -- logging ---------------------------------------------------------------
    def log_step(
        self,
        logits: torch.Tensor,
        loss:   float,
    ) -> None:
        row = {"step": self._step, "loss": round(loss, 6)}

        for name, acts in self._activations.items():
            for i, v in enumerate(acts):
                row[f"{name}_n{i}"] = round(v, 6)

        # log mean logit per class
        mean_logits = logits.detach().mean(0).cpu().tolist()
        for i, v in enumerate(mean_logits):
            row[f"logit_{i}"] = round(v, 6)

        # lazy header write
        if self._writer is None:
            import csv as _csv
            self._writer = _csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()

        self._writer.writerow(row)
        self._step += 1


def log_activations(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    max_steps:   int = 200,
    csv_path:    Path = RESULTS / "activations.csv",
) -> None:
    """Run one pass over `loader` and stream activations to CSV."""
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with ActivationLogger(csv_path) as logger:
        logger.register_hooks(model)

        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                if step >= max_steps:
                    break
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                loss   = criterion(logits, y).item()
                logger.log_step(logits, loss)

    print(f"[activation_log] saved {min(max_steps, step+1)} steps -> {csv_path}")


# -- Inference timing ----------------------------------------------------------

def measure_inference_time(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_runs: int = 50,
) -> float:
    """Return mean inference time (ms) per batch."""
    model.eval()
    batch = next(iter(loader))[0][:64].to(device)

    # warm-up
    with torch.no_grad():
        for _ in range(5):
            model(batch)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(batch)
            times.append((time.perf_counter() - t0) * 1000)

    return float(torch.tensor(times).mean().item())



