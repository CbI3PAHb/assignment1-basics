from __future__ import annotations

import functools
import json
import logging
import math
import os

import einops
import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


# uv run pytest -k test_linear
class LinearModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_feautres = out_features
        data = torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        mean = 0.0
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            tensor=data, mean=mean, std=std, a=-3 * std, b=3 * std
        )
        self.weight = nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )


# uv run pytest -k test_embedding
class EmbeddingModule(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        data = torch.empty(
            size=(num_embeddings, embedding_dim),
            device=device,
            dtype=dtype,
        )
        torch.nn.init.trunc_normal_(tensor=data, mean=0, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


# uv run pytest -k test_rmsnorm
class RootMeanSquareLayerNormalizationModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        data = torch.ones(size=(d_model,), device=device, dtype=dtype)
        self.weight = nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (
            x
            / (x.pow(2).sum(-1, keepdim=True) / self.d_model + self.eps).sqrt()
            * self.weight
        )
        return result.to(in_dtype)


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = (
            d_ffn
            if d_ffn is not None
            else self._make_d_ffn_divisible_by_k(self.d_model, 64)
        )

        self.w1 = LinearModule(self.d_model, self.d_ffn, device=device, dtype=dtype)
        self.w3 = LinearModule(self.d_model, self.d_ffn, device=device, dtype=dtype)
        self.w2 = LinearModule(self.d_ffn, self.d_model, device=device, dtype=dtype)

    def _make_d_ffn_divisible_by_k(self, d_model: int, k: int = 64):
        d_ffn_approx = int(d_model * 8 / 3)
        d_ffn = (d_ffn_approx + k - 1) // k * k
        return d_ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(SiLU(self.w1(x)) * self.w3(x))
        return x


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        seq_dim = torch.arange(0, self.max_seq_len, dtype=torch.float32, device=device)
        inv_freqs = theta ** -(
            torch.arange(0, self.d_k, 2, dtype=torch.float32, device=device) / self.d_k
        )
        freqs = einops.einsum(seq_dim, inv_freqs, "i, j -> i j")

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq d"],
        pos_ids: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d"]:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence len = ({seq_len}) is greater than max seq len = ({self.max_seq_len})."
            )
        if pos_ids is None:
            sin = self.sin[:seq_len, :]
            cos = self.cos[:seq_len, :]
        else:
            sin = self.sin[pos_ids, :]
            cos = self.cos[pos_ids, :]

        x1, x2 = einops.rearrange(
            x, "... (half_d_model x1x2) -> x1x2 ... half_d_model", x1x2=2
        )
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        return einx.rearrange(
            "... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot
        ).contiguous()


def softmax(x: torch.Tensor, dim: int = -1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[Tensor, "... seq_len d_k"],
    k: Float[Tensor, "... seq_len d_k"],
    v: Float[Tensor, "... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
):
    o = einops.einsum(
        q, k, "... q_seq_len d_k, ... k_seq_len d_k -> ... q_seq_len k_seq_len"
    )
    o = o / q.shape[-1] ** 0.5

    if mask is not None:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(~mask, -torch.inf)
        o = o + mask

    p = softmax(o)
    return einops.einsum(
        p, v, "... q_seq_len k_seq_len, ... k_seq_len d_v -> ... q_seq_len d_v"
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        # batch
        # seq_len_query
        # seq_len_key
        # d_model, embedding_dim
        # head_dim
        # n_heads_query
        # n_heads_key_value
        # gqa_factor = n_heads_query // n_heads_key_value
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = self.d_model // self.n_heads
        self.w_qkv = LinearModule(d_model, 3 * self.n_heads * self.head_dim)
        self.output_proj = LinearModule(self.n_heads * self.head_dim, d_model)
        self.rope = (
            RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)
            if theta
            else None
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        q, k, v = einops.rearrange(
            self.w_qkv(x),
            "... seq_len (qkv n_heads head_size) -> qkv n_heads ... seq_len head_size",
            qkv=3,
            n_heads=self.n_heads,
        )
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(size=(seq_len, seq_len), dtype=torch.bool)).T
        a = scaled_dot_product_attention(q, k, v, mask)
        a = einops.rearrange(
            a, "n_heads ... seq_len head_dim -> ... seq_len (n_heads head_dim)"
        )
        return self.output_proj(a)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super().__init__()
        self.ln1 = RootMeanSquareLayerNormalizationModule(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RootMeanSquareLayerNormalizationModule(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = EmbeddingModule(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RootMeanSquareLayerNormalizationModule(d_model)
        self.lm_head = LinearModule(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ):
        hidden_state = self.token_embeddings(in_indices)
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        hidden_state = self.ln_final(hidden_state)
        logits = self.lm_head(hidden_state)
        return logits
