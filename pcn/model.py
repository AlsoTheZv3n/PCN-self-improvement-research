"""PCN model container: weights, biases, activation. No autograd is used anywhere.

Convention (following Goemaere et al. 2025 / docs/09, feedforward indexing):
    states[0]      = input  (clamped to x)
    states[1..n-1] = hidden (always free / settling)
    states[n]      = output (clamped to one-hot y in training, free at test)
    W[i], b[i] map states[i] -> prediction of states[i+1]:
        pred_{i+1} = W[i] @ phi(states[i]) + b[i]      (batched: phi(s) @ W[i].T + b[i])
"""
from __future__ import annotations

import torch


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(x) ** 2


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _identity_deriv(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _sigmoid_deriv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


# name -> (phi, phi'). The M6 CUDA kernel supports tanh (act 0), identity (act 1) and sigmoid
# (act 2, added for Song et al. 2024's Fig-4e replication / §4g); see kernels/settling_cuda.py.
ACTIVATIONS = {
    "tanh": (_tanh, _tanh_deriv),
    "identity": (_identity, _identity_deriv),
    "sigmoid": (_sigmoid, _sigmoid_deriv),
}


class PCN:
    """A hierarchical Predictive Coding Network (State-Optimization formulation)."""

    def __init__(
        self,
        layer_sizes,
        activation: str = "tanh",
        weight_init: str = "orthogonal",
        weight_scale: float = 0.5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
        self.layer_sizes = list(layer_sizes)
        self.n = len(self.layer_sizes) - 1  # number of weight matrices
        if self.n < 1:
            raise ValueError("need at least two layer sizes (input, output)")
        self.device = device
        self.dtype = dtype
        self.phi, self.phi_deriv = ACTIVATIONS[activation]

        # Orthogonal init keeps eigenvalues ~1 and mitigates SO signal decay (docs/09).
        torch.manual_seed(seed)
        self.W, self.b = [], []
        for i in range(self.n):
            out_dim, in_dim = self.layer_sizes[i + 1], self.layer_sizes[i]
            w = torch.empty(out_dim, in_dim, dtype=dtype)
            if weight_init == "orthogonal":
                torch.nn.init.orthogonal_(w)
            elif weight_init == "normal":
                w.normal_(0.0, weight_scale)
            elif weight_init == "xavier":          # Song et al. 2024 use Xavier-normal
                torch.nn.init.xavier_normal_(w)
            else:
                raise ValueError(f"unknown weight_init {weight_init!r}")
            self.W.append(w.to(device))
            self.b.append(torch.zeros(out_dim, dtype=dtype, device=device))

    @torch.no_grad()
    def predict(self, i: int, state_i: torch.Tensor) -> torch.Tensor:
        """Top-down prediction of layer i+1 from layer i: phi(state_i) @ W[i].T + b[i]."""
        return self.phi(state_i) @ self.W[i].t() + self.b[i]
