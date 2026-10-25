from __future__ import annotations

import attrs
import einops
import einx
import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor


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

    def forward(
        self, x: Float[Tensor, "... in_features"]
    ) -> Float[Tensor, "... out_features"]:
        return einops.einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )


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

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]


class RMSNormModule(nn.Module):
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (
            x
            / (x.pow(2).sum(-1, keepdim=True) / self.d_model + self.eps).sqrt()
            * self.weight
        )
        return result.to(in_dtype)


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        x = self.w2(silu(self.w1(x)) * self.w3(x))
        return x


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        qk_head_dim: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.qk_head_dim = qk_head_dim
        self.max_seq_len = max_seq_len

        seq_dim: Float[Tensor, "max_seq_len"] = torch.arange(
            0, self.max_seq_len, dtype=torch.float32, device=device
        )
        inv_freqs: Float[Tensor, "qk_head_dim"] = theta ** -(
            torch.arange(0, self.qk_head_dim, 2, dtype=torch.float32, device=device)
            / self.qk_head_dim
        )
        freqs: Float[Tensor, "max_seq_len qk_head_dim"] = einops.einsum(
            seq_dim, inv_freqs, "i, j -> i j"
        )

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq d"],
        pos_ids: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d"]:
        input_dtype = x.dtype
        seq_len = x.shape[-2]

        if pos_ids is not None and seq_len != pos_ids.shape[-1]:
            raise ValueError(f"got {pos_ids.shape[-1]=}, {x.shape[-2]=}")

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

        odds, evens = einops.rearrange(
            x, "... (half_d_model two) -> two ... half_d_model", two=2
        )
        new_odds = odds * cos - evens * sin
        new_evens = odds * sin + evens * cos

        # re-interleave odds and evens:
        # odds       = [0,    2,    4,    ...]
        # evens      = [   1,    3,    5, ...]
        # rearranged = [0, 1, 2, 3, 4, 5, ...]
        return (
            einx.rearrange(
                "... x_half, ... x_half -> ... (x_half (1 + 1))", new_odds, new_evens
            )
            .contiguous()
            .to(input_dtype)
        )


def softmax(x: torch.Tensor, dim: int = -1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[Tensor, "... q_seq_len qk_head_dim"],
    k: Float[Tensor, "... kv_seq_len qk_head_dim"],
    v: Float[Tensor, "... kv_seq_len v_head_dim"],
    mask: Bool[Tensor, "q_seq_len kv_seq_len"] | None = None,
):
    scores = einops.einsum(
        q,
        k,
        "... q_seq_len qk_head_dim, ... kv_seq_len qk_head_dim -> ... q_seq_len kv_seq_len",
    )
    scores = scores / q.shape[-1] ** 0.5

    if mask is not None:
        mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(~mask, -torch.inf)
        scores = scores + mask

    probs = softmax(scores)
    return einops.einsum(
        probs,
        v,
        "... q_seq_len kv_seq_len, ... kv_seq_len v_head_dim -> ... q_seq_len v_head_dim",
    )


@attrs.define(kw_only=True)
class LayerKVCache:
    _k_cache: Float[Tensor, "n_heads batch_size max_seq_len head_dim"] | None = (
        attrs.field(default=None)
    )
    _v_cache: Float[Tensor, "n_heads batch_size max_seq_len head_dim"] | None = (
        attrs.field(default=None)
    )
    valid_kv_lengths: Int[Tensor, "batch_size"] = attrs.field(default=None)

    def check_kv_cache_consistency(self) -> None:
        k, v = self._k_cache, self._v_cache

        if k is None and v is None:
            return

        if k is None or v is None:
            raise RuntimeError("K and V caches must both be present or both be None")

        if k.device != v.device:
            raise RuntimeError("K cache and V cache are lives in different devices")

        if k.shape != v.shape:
            # TODO k.shape[-1] and v.shape[-1] may not be equal cause
            # qk_head_dim may not equal to v_head_dim, but all remain dimensions must be equal(?)
            # but current MultiHeadSelfAttention implementation doesn't support different qk and v head sizes

            raise RuntimeError(f"KV cache shapes differ: K={k.shape}, V={v.shape}")

        # TODO: find max every time - perf sucks, maybe find max with torch?
        if k.shape[-2] < max(self.valid_kv_lengths):
            raise RuntimeError(f"KV storage is less then max valid_kv_lenght")

    @property
    def is_empty(self) -> bool:
        self.check_kv_cache_consistency()
        return self._k_cache is None

    def k_cache(self) -> torch.Tensor:
        self.check_kv_cache_consistency()
        if self._k_cache is None:
            raise RuntimeError("KV cache is empty")
        return self._k_cache

    def v_cache(self) -> torch.Tensor:
        self.check_kv_cache_consistency()
        if self._v_cache is None:
            raise RuntimeError("KV cache is empty")
        return self._v_cache

    def _zero_padding(self):
        self.check_kv_cache_consistency()

        k = self.k_cache()
        v = self.v_cache()

        key_indices = torch.arange(k.shape[-2], device=k.device)
        invalid = (
            key_indices[None, None, :, None]
            >= self.valid_kv_lengths[None, :, None, None]
        )  # [1, B, K, 1]

        k.masked_fill_(invalid, 0)
        v.masked_fill_(invalid, 0)

    @jaxtyped(typechecker=beartype)
    def update_kv(
        self,
        k: Float[
            Tensor, "n_heads batch_size max_seq_len head_dim"
        ],  # qk_head_dim = v_head_dim for now)
        v: Float[Tensor, "n_heads batch_size max_seq_len head_dim"],
        *,
        input_lengths: Int[Tensor, "batch_size"] | None = None,
    ) -> None:
        self.check_kv_cache_consistency()

        if k.shape != v.shape:
            # TODO k.shape[-1] and v.shape[-1] may not be equal cause
            # qk_head_dim may not equal to v_head_dim, but all remain dimensions must be equal(?)
            raise RuntimeError(f"KV cache shapes differ: K={k.shape}, V={v.shape}")

        if input_lengths is None:
            input_lengths = torch.tensor(
                [int(k.shape[-2]) for _ in range(k.shape[1])],
                dtype=torch.int32,
                device=k.device,
            )

        if (input_lengths < 0).any():
            raise ValueError("Some input has negative length")

        if input_lengths.max() > k.shape[-2]:
            raise ValueError(
                f"Logical length is greater than physical storage, got {input_lengths.max()=}, {k.shape[-2]=}"
            )

        if self.is_empty:
            self._k_cache = k.clone()
            self._v_cache = v.clone()
            self.valid_kv_lengths = input_lengths
            self._zero_padding()
            return

        old_k = self.k_cache()
        old_v = self.v_cache()
        lengths = self.valid_kv_lengths
        if lengths is None:
            raise RuntimeError("Initialized cache has no lengths")

        if k.shape[0] != old_k.shape[0]:
            raise ValueError(
                f"Old k and new k has different n_heads, got {k.shape[0]=}, {old_k.shape[0]}"
            )

        if k.shape[1] != old_k.shape[1]:
            raise ValueError(
                f"Old k and new k has different batch_size, got {k.shape[1]=}, {old_k.shape[1]}"
            )

        if k.shape[3] != old_k.shape[3]:
            raise ValueError(
                f"Old k and new k has different k_head_dim, got {k.shape[3]=}, {old_k.shape[3]}"
            )

        new_valid_lengths: torch.Tensor = lengths + input_lengths
        n_heads, batch_size, max_seq_len, head_size = old_k.shape

        # because of padding, old k may have padding in old_k/v storage
        # [ + - - - - ], [1]
        # [ + + - - - ], [2]
        # [ + - - - - ], [1]
        new_max_seq_len_dim = max(
            int(new_valid_lengths.max().item()),
            old_k.shape[2],
        )
        new_shape = [n_heads, batch_size, new_max_seq_len_dim, head_size]

        # update k
        new_k_cache = old_k.new_empty(size=new_shape)
        new_k_cache[:, :, : old_k.shape[-2], :] = old_k

        for b in range(k.shape[1]):
            start = int(lengths[b].item())
            count = int(input_lengths[b].item())
            new_k_cache[:, b, start : start + count, :] = k[:, b, :count, :]
            self._k_cache = new_k_cache

        new_v_cache = old_v.new_empty(size=new_shape)
        new_v_cache[:, :, : old_v.shape[-2], :] = old_v

        for b in range(v.shape[1]):
            start = int(lengths[b].item())
            count = int(input_lengths[b].item())
            new_v_cache[:, b, start : start + count, :] = v[:, b, :count, :]
            self._v_cache = new_v_cache

        self.valid_kv_lengths = new_valid_lengths
        self._zero_padding()


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        # TODO: clean here
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
            RoPE(theta=theta, qk_head_dim=self.head_dim, max_seq_len=max_seq_len)
            if theta
            else None
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        *,
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
        kv_cache: LayerKVCache | None = None,
        input_lengths: Int[Tensor, "batch_size"] | None = None,
    ):

        q, k, v = einops.rearrange(
            self.w_qkv(x),
            "... seq_len (qkv n_heads head_size) -> qkv n_heads ... seq_len head_size",
            qkv=3,
            n_heads=self.n_heads,
        )

        device = q.device

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if kv_cache is not None:
            kv_cache.update_kv(k=k, v=v, input_lengths=input_lengths)
            k = kv_cache.k_cache()
            v = kv_cache.v_cache()

        # TODO: cpu sync?
        q_seq_len = int(q.shape[-2])
        kv_seq_len = int(k.shape[-2])
        batch_size = int(q.shape[1])

        # TODO: remove ru lang from comments
        # input_lengths - валидные длины, если none, то весь тензор валидный!
        if input_lengths is None:
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=(q_seq_len),
                device=device,
                dtype=torch.int32,
            )

        if kv_cache is not None:
            if token_positions is None:
                token_positions = torch.arange(q.shape[-2]).repeat(batch_size, 1)

            k_positions = torch.arange(
                start=0, end=kv_seq_len, dtype=torch.int32, device=device
            )[
                None, None, None, :
            ]  # [1, 1, 1, K] - logical token positions in K/V cache

            # [1, B, Q, 1] - logical token positions in Q
            q_positions = token_positions[None, :, :, None]

            # [1, B, 1, 1] - valid positions in K/V cache
            valid_k_lengths = kv_cache.valid_kv_lengths[None, :, None, None]

            # [1, B, Q, K] - bool causal mask
            mask = (k_positions <= q_positions) & (k_positions < valid_k_lengths)
        else:
            mask = torch.tril(
                torch.ones(
                    size=(q_seq_len, kv_seq_len), dtype=torch.bool, device=x.device
                ),
                diagonal=kv_seq_len - q_seq_len,
            )

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
        self.ln1 = RMSNormModule(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNormModule(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        *,
        token_positions: Int[Tensor, "batch_size sequence_length"] | None = None,
        kv_cache: LayerKVCache | None = None,
        input_lengths: Int[Tensor, "batch_size"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
            kv_cache=kv_cache,
            input_lengths=input_lengths,
        )
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
        self.ln_final = RMSNormModule(d_model)
        self.lm_head = LinearModule(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size sequence_length"],
        *,
        token_positions: Int[Tensor, "batch_size sequence_length"] | None = None,
        kv_caches: list[LayerKVCache] | None = None,
        input_lengths: Int[Tensor, "batch_size"] | None = None,
    ):
        if kv_caches is not None and len(kv_caches) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} layer caches, got {len(kv_caches)}"
            )

        hidden_state = self.token_embeddings(in_indices)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            hidden_state = layer(
                hidden_state,
                token_positions=token_positions,
                kv_cache=kv_cache,
                input_lengths=input_lengths,
            )
        hidden_state = self.ln_final(hidden_state)
        logits = self.lm_head(hidden_state)
        return logits


@jaxtyped(typechecker=beartype)
def cross_entropy_loss(
    logits: Float[Tensor, "... seq_len vocab_size"],
    targets: Int[Tensor, "... seq_len"],
) -> Float[Tensor, ""]:
    # -log(softmax(logits)) = -x_correct + log(sum(exp(x)))
    log_sum_exp_logits: Float[Tensor, "... seq_len"] = einx.logsumexp(
        "... vocab_size -> ...",
        logits,
    )

    selected_logits: Float[Tensor, "... seq_len"] = einx.get_at(
        "... seq_len [vocab_size], ... seq_len -> ... seq_len",
        logits,
        targets,
    )

    loss: Float[Tensor, "... seq_len"] = -selected_logits + log_sum_exp_logits

    return loss.mean()
