"""
Mixture of Experts (MoE) Feed-Forward Network
=============================================
Sparse MoE FFN with top‑k routing.  Replaces the dense SwiGLU FFN in
Transformer blocks when ``num_experts > 0`` in the config.

Each expert is an independent SwiGLU FFN.  A learned router selects up
to ``top_k`` experts per token; the output is the weighted sum of the
selected expert outputs (load‑balanced via auxiliary loss hint).

Reference:
    Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer" https://arxiv.org/abs/1701.06538
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLUExpert(nn.Module):
    """Single SwiGLU expert."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.fc1(x))
        return self.fc3(gate * self.fc2(x))


class MoEFeedForward(nn.Module):
    """Sparse Mixture‑of‑Experts FFN.

    Args:
        dim: Hidden dimension of the model.
        hidden_dim: Intermediate dimension *per expert*.
        num_experts: Total number of experts.  Default: 8.
        top_k: Experts activated per token.  Default: 2.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_SwiGLUExpert(dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_balance_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass with top‑k routing.

        Args:
            x: Input ``[B, T, D]``.
            return_balance_loss: If True, return ``(output, loss)`` tuple.

        Returns:
            Output ``[B, T, D]``, or ``(output, balance_loss)``.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [B*T, D]

        router_logits = self.router(x_flat)  # [B*T, E]
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_idx = torch.topk(
            router_probs, self.top_k, dim=-1, sorted=False
        )
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_idx == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.nonzero(as_tuple=True)[0]

            idx_mask = top_k_idx == expert_idx
            weights = (top_k_weights * idx_mask.float()).sum(dim=-1)
            weights = weights[token_indices]

            token_in = x_flat[token_indices]
            token_out = self.experts[expert_idx](token_in)
            out[token_indices] += token_out * weights.unsqueeze(-1)

        result = out.view(B, T, D)

        if not return_balance_loss:
            return result

        # Auxiliary load-balancing loss
        # f_i = fraction of tokens dispatched to expert i
        # P_i = average router probability for expert i
        # loss = E * sum(f_i * P_i)
        with torch.no_grad():
            expert_mask_batch = torch.zeros(B * T, self.num_experts, device=x.device)
            expert_mask_batch.scatter_(1, top_k_idx, 1.0)
            f_i = expert_mask_batch.mean(dim=0)  # [E]

        P_i = router_probs.mean(dim=0)  # [E]
        balance_loss = self.num_experts * (f_i * P_i).sum()

        return result, balance_loss
