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
from torch import Tensor
from jaxtyping import Float, Bool, Int


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
        std = (2. / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(tensor=data, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


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
        data = torch.ones(size=(d_model, ), device=device, dtype=dtype)
        self.weight = nn.Parameter(data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / (x.pow(2).sum(-1, keepdim=True) / self.d_model + self.eps).sqrt() * self.weight
        return result.to(in_dtype)


def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

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
        self.d_ffn = d_ffn if d_ffn is not None else self._make_d_ffn_divisible_by_k(self.d_model, 64)

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
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        seq_dim = torch.arange(0, self.max_seq_len, dtype=torch.float32, device=device)
        inv_freqs = theta ** -(torch.arange(0, self.d_k, 2, dtype=torch.float32, device=device) / self.d_k)
        freqs = einops.einsum(seq_dim, inv_freqs, "i, j -> i j")
        
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        # input (..., seq_len, d_k)
        # output (..., seq_len, d_k)
        # einops.einsum()
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
             raise ValueError(
                f"Sequence len = ({seq_len}) is greater than max seq len = ({self.max_seq_len})."
             )
        sin = self.sin[pos_ids, :]
        cos = self.cos[pos_ids, :]

        x1, x2 = einops.rearrange(x, "... (half_d_model x1x2) -> x1x2 ... half_d_model", x1x2=2)
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        return einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()


