"""MLP model definition with optional structural pruning support."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import HIDDEN_DIMS, INPUT_DIM, OUTPUT_DIM


class MLP(nn.Module):
    """Fully-connected MLP: Input -> Dense(128) -> Dense(64) -> Output(10).

    Supports structural rebuild after pruning via ``from_dims``.
    """

    def __init__(
        self,
        hidden_dims: list[int] = HIDDEN_DIMS,
        input_dim:   int       = INPUT_DIM,
        output_dim:  int       = OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim   = input_dim
        self.output_dim  = output_dim

        dims = [input_dim] + hidden_dims + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation after last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        # convenient handles for activation logging
        self._hidden_layers: list[nn.Linear] = [
            m for m in self.net if isinstance(m, nn.Linear)
        ][:-1]

    # -- forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)

    # -- helpers ---------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_dims(
        cls,
        hidden_dims: list[int],
        input_dim:   int = INPUT_DIM,
        output_dim:  int = OUTPUT_DIM,
    ) -> "MLP":
        """Build a new MLP with arbitrary hidden widths (used after pruning)."""
        return cls(hidden_dims=hidden_dims, input_dim=input_dim, output_dim=output_dim)

    # -- weight copying for structural pruning --------------------------------
    def copy_weights_from(
        self,
        source:       "MLP",
        keep_indices: list[Optional[list[int]]],
    ) -> None:
        """Copy surviving neuron weights from *source* into this (smaller) model.

        ``keep_indices[i]`` is the list of neuron indices retained from layer i
        of the *source* hidden layers.  None means keep all.
        """
        src_linears  = [m for m in source.net if isinstance(m, nn.Linear)]
        self_linears = [m for m in self.net   if isinstance(m, nn.Linear)]

        n_layers = len(src_linears)
        assert len(keep_indices) == len(src_linears) - 1, \
            "keep_indices must have one entry per hidden layer"

        with torch.no_grad():
            for layer_idx in range(n_layers):
                src_layer  = src_linears[layer_idx]
                dst_layer  = self_linears[layer_idx]

                # rows = output neurons of this layer
                out_idx = keep_indices[layer_idx] if layer_idx < len(keep_indices) else None
                # cols = input neurons (surviving from previous layer)
                in_idx  = keep_indices[layer_idx - 1] if layer_idx > 0 else None

                W = src_layer.weight.data
                b = src_layer.bias.data

                if out_idx is not None:
                    W = W[out_idx, :]
                    b = b[out_idx]
                if in_idx is not None:
                    W = W[:, in_idx]

                dst_layer.weight.data.copy_(W)
                dst_layer.bias.data.copy_(b)



