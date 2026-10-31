"""Compute attention for requests whose NEW tokens are packed together.

Old K/V already live in PagedKVStore. This module saves new K/V and computes each
new query against its own request's causal history. It returns attention vectors.

Example: A has cached A0,A1 and decodes A2; B prefills B0,B1,B2.
    New Q/K/V: [A2,B0,B1,B2], each tensor [H,4,D].
    q_offsets_cpu=(0,1,4): A uses [:,0:1,:]; B uses [:,1:4,:].
    kv_lengths_cpu=(3,3): total history lengths AFTER storing the new tokens.
    A2 sees A0,A1,A2. B0 sees B0; B1 sees B0,B1; B2 sees B0,B1,B2.
Old queries are unnecessary. Requests never attend to one another's tokens.

Current exercise: metadata, for_layer and attend. Model integration comes later.
"""

import attrs
import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from cs336_basics.config._validation import _NONNEGATIVE_INTEGER

from .attention_ops import scaled_dot_product_attention
from .paged_kv import PagedKVStore

# B: Number of requests included in this forward.
# T: Total new input tokens: sum(q_i), where q_i is the count for request i. Excludes already cached tokens.
# S: Number of requests that need sampling after this forward. 0 <= S <= B; intermediate prefill chunks don’t sample.
# M: Maximum number of block IDs in any included request’s block table. block_tables has shape [B, M].


@attrs.define(kw_only=True)
class PagedBatchMetadata:
    """Describe boundaries and physical addresses for ONE packed forward.

    Args:
        q_offsets: Device int[B+1] packed query boundaries, from 0 to T.
            Request i contributes offsets[i+1] - offsets[i] new tokens.
        kv_lengths: Device int[B] total valid histories AFTER this step's write,
            including both cached prefixes and new tokens.
        block_tables: Device int[B,M] logical-page to physical-page mappings.
            Unused trailing entries are -1; never read them as physical pages.
        write_slots: Device int[T] destination of each new packed K/V token.
            A slot is block_id * block_size + offset; order matches packed input.
        q_offsets_cpu: Python tuple[int, ...], length B+1, with q_offsets values.
        kv_lengths_cpu: Python tuple[int, ...], length B, with kv_lengths values.

    Caller builds CPU/device versions from the same CPU data. The tuples supply
    loop bounds without GPU readback. Tests supply this object; later Engine will.
    This class does no sampling, scheduling, page allocation or attention math.

    Preconditions:
        Metadata is consistent; each request has positive work; addresses are
        valid. No new invalid-metadata exception policy is required in this task.
    """

    q_offsets: Int[Tensor, "num_offsets"] = attrs.field()  # [B+1]
    kv_lengths: Int[Tensor, "B"]
    block_tables: Int[Tensor, "B M"]
    write_slots: Int[Tensor, "T"]
    q_offsets_cpu: tuple[int, ...] = attrs.field(validator=attrs.validators.instance_of(tuple))  # [B+1]
    kv_lengths_cpu: tuple[int, ...] = attrs.field(validator=attrs.validators.instance_of(tuple))  # [B]

    @q_offsets.validator
    def _validate_q_offsets(self, attribute, value: Tensor) -> None:
        batch_size = self.kv_lengths.shape[0]
        expected_shape = (batch_size + 1,)

        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{attribute.name} must have shape {expected_shape}, got {tuple(value.shape)}")

    @q_offsets_cpu.validator
    def _validate_q_offsets_cpu(self, attribute, value: tuple[int, ...]) -> None:
        expected = self.q_offsets.shape[0]
        if len(value) != expected:
            raise ValueError(f"{attribute.name} must have length {expected}, got {len(value)}")

    @kv_lengths_cpu.validator
    def _validate_kv_lengths_cpu(self, attribute, value: tuple[int, ...]) -> None:
        expected = self.kv_lengths.shape[0]
        if len(value) != expected:
            raise ValueError(f"{attribute.name} must have length {expected}, got {len(value)}")


@attrs.define(kw_only=True)
class PagedLayerContext:
    """Bind the shared store and step metadata to one transformer layer.

    Implement construction so the object retains the store, metadata and layer
    index supplied by for_layer. Constructor/private field layout is your choice.
    Keep references; do not clone pools or metadata. attend uses this state to
    read/write exactly the selected layer's K/V storage.
    """

    # k_store: torch.Tensor # TODO
    # v_store: torch.Tensor # TODO
    kv_store: PagedKVStore
    metadata: PagedBatchMetadata
    layer_idx: int = attrs.field(validator=_NONNEGATIVE_INTEGER)

    def attend(
        self,
        q: Float[Tensor, "n_q_heads packed_tokens qk_head_dim"],
        k: Float[Tensor, "n_kv_heads packed_tokens qk_head_dim"],
        v: Float[Tensor, "n_kv_heads packed_tokens v_head_dim"],
    ) -> Float[Tensor, "n_kv_heads packed_tokens qk_head_dim"]:
        """Return attention outputs for all NEW queries in packed order.

        Args:
            q: New queries [H,T,D], already projected and RoPE-rotated.
            k: New keys [H,T,D], already projected and RoPE-rotated.
            v: New projected values [H,T,D]. Old K/V are already in the store.

        Work to implement:
            1. Write all new K/V once using this layer and metadata.write_slots.
            2. For request i, take left/right from q_offsets_cpu. Its queries are
               q[:,left:right,:]. Gather its history using block_tables[i] and
               kv_lengths_cpu[i], including the K/V just written.
            3. Compute scaled dot-product attention against that history. Let
               count=right-left and start=kv_length-count. Local query j sees
               keys 0 through start+j, inclusive. Mask later keys BEFORE softmax.
            4. Concatenate outputs along the token axis in packed request order.

        Returns:
            Attention vectors [H,T,D], same dtype/device as inputs. Output token t
            corresponds to query t. Head merging, output projection, RoPE and
            sampling are outside this method.

        Side effects:
            Updates only this layer's addressed K/V slots. Preserve input q/k/v,
            cached prefixes and other layers. No GPU-to-CPU metadata readbacks.

        Preconditions:
            Inputs and metadata are valid and consistent; no new validation or
            exception behavior is required. A Python per-request loop inside
            this reference attention method is allowed.
        """
        assert q.shape == k.shape == v.shape, f"gqa isn't supported yet"  # TODO support GQA
        n_q_heads = q.shape[0]
        packed_tokens = q.shape[1]
        v_head_dim = v.shape[-1]

        self.kv_store.write(
            layer_idx=self.layer_idx,
            slots=self.metadata.write_slots,
            k=k,
            v=v,
        )

        outputs: Float[Tensor, "n_q_heads packed_tokens v_head_dim"] = q.new_empty(
            size=(n_q_heads, packed_tokens, v_head_dim)
        )

        for request_idx, (left, right) in enumerate(
            zip(self.metadata.q_offsets_cpu[:-1], self.metadata.q_offsets_cpu[1:], strict=True)
        ):
            kv_cache: tuple = self.kv_store.gather(
                layer_idx=self.layer_idx,
                block_table=self.metadata.block_tables[request_idx],
                kv_len=self.metadata.kv_lengths_cpu[request_idx],
            )
            k_cache: Float[Tensor, "n_kv_heads kv_len qk_head_dim"] = kv_cache[0]
            v_cache: Float[Tensor, "n_kv_heads kv_len qk_head_dim"] = kv_cache[1]
            q_current: Float[Tensor, "n_q_heads q_seq_len qk_head_dim"] = q[:, left:right, :]

            # mask [q_seq_len, k_seq_len]
            q_seq_len = right - left
            k_seq_len = self.metadata.kv_lengths_cpu[request_idx]

            start = k_seq_len - q_seq_len
            q_range = torch.arange(q_seq_len, dtype=torch.int32, device=q_current.device) + start
            k_range = torch.arange(k_seq_len, dtype=torch.int32, device=k_cache.device)
            mask = q_range[:, None] >= k_range[None, :]

            attn_output: Float[Tensor, "n_q_heads q_seq_len qk_head_dim"] = scaled_dot_product_attention(
                q=q[:, left:right, :], k=k_cache, v=v_cache, mask=mask
            )
            outputs[:, left:right, :] = attn_output
        return outputs


@attrs.define(kw_only=True)
class PagedForwardContext:
    """Carry the shared store and this step's metadata through one model forward.

    Args:
        kv_store: Existing PagedKVStore containing every layer's pages.
        metadata: This step's packed request description, shared by all layers.

    This is a bundle of references. Construction allocates no KV pages and runs
    no attention. The model will obtain a layer context for each transformer layer.
    """

    kv_store: PagedKVStore = attrs.field(validator=attrs.validators.instance_of(PagedKVStore))
    metadata: PagedBatchMetadata = attrs.field(validator=attrs.validators.instance_of(PagedBatchMetadata))

    def for_layer(self, layer_idx: int) -> PagedLayerContext:
        """Return an attention context bound to one layer; perform no computation.

        Args:
            layer_idx: Valid CPU integer, 0 <= layer_idx < kv_store.num_layers.

        Returns:
            PagedLayerContext retaining this same store/metadata and the selected
            index. Its attend method will perform the work later. Layer contexts
            must retain independent indices, even when several coexist.

        Side effects:
            None on the store/metadata: do not write pages or copy K/V here.

        Preconditions:
            Caller supplies a valid index; no new exception policy is required.
        """
        return PagedLayerContext(
            # k_store=self.kv_store.k_pages[layer_idx],
            # v_store=self.kv_store.v_pages[layer_idx],
            kv_store=self.kv_store,
            metadata=self.metadata,
            layer_idx=layer_idx,
        )
