import einops
import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor


def softmax(x: torch.Tensor, dim: int = -1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


# @jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    q: Float[Tensor, "... q_seq_len qk_head_dim"],
    k: Float[Tensor, "... kv_seq_len qk_head_dim"],
    v: Float[Tensor, "... kv_seq_len v_head_dim"],
    mask: Bool[Tensor, "q_seq_len kv_seq_len"] | None = None,
) -> Float[Tensor, "... q_seq_len v_head_dim"]:
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
