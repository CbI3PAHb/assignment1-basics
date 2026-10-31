import attrs
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from cs336_basics.config._validation import _POSITIVE_INTEGER


@attrs.define(kw_only=True)
class PagedKVStore:
    num_layers: int = attrs.field(validator=_POSITIVE_INTEGER)
    num_blocks: int = attrs.field(validator=_POSITIVE_INTEGER)
    block_size: int = attrs.field(validator=_POSITIVE_INTEGER)
    num_heads: int = attrs.field(validator=_POSITIVE_INTEGER)
    head_dim: int = attrs.field(validator=_POSITIVE_INTEGER)
    device: torch.device = attrs.field(converter=torch.device)
    dtype: torch.dtype

    k_pages: Float[Tensor, "n_layers n_blocks block_size n_kv_heads qk_head_dim"] = attrs.field(init=False)
    v_pages: Float[Tensor, "n_layers n_blocks block_size n_kv_heads qk_head_dim"] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        shape = (
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_heads,
            self.head_dim,
        )
        self.k_pages = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.v_pages = torch.empty(shape, device=self.device, dtype=self.dtype)

    @jaxtyped(typechecker=beartype)
    def write(
        self,
        *,
        layer_idx: int,
        slots: Int[Tensor, "kv_len"],
        k: Float[Tensor, "n_kv_heads kv_len qk_head_dim"],
        v: Float[Tensor, "n_kv_heads kv_len qk_head_dim"],
    ) -> None:
        """stores k/v shape [n_kv_heads kv_len qk_head_dim] to kv cache storage"""
        block_ids = slots // self.block_size
        block_offsets = slots % self.block_size
        self.k_pages[layer_idx, block_ids, block_offsets] = k.transpose(0, 1)
        self.v_pages[layer_idx, block_ids, block_offsets] = v.transpose(0, 1)

    @jaxtyped(typechecker=beartype)
    def gather(
        self, *, layer_idx: int, block_table: Int[Tensor, "n_blocks"], kv_len: int
    ) -> tuple[Float[Tensor, "n_kv_heads kv_len qk_head_dim"], Float[Tensor, "n_kv_heads kv_len qk_head_dim"]]:
        """returns K/V [n_kv_heads kv_len qk_head_dim]"""
        positions = torch.arange(kv_len, device=block_table.device)  # [T]
        logical_blocks = positions // self.block_size  # [T]

        block_ids = block_table[logical_blocks]  # [T]
        block_offsets = positions % self.block_size  # [T]

        return (
            self.k_pages[layer_idx, block_ids, block_offsets].transpose(0, 1),
            self.v_pages[layer_idx, block_ids, block_offsets].transpose(0, 1),
        )
