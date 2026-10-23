import attrs
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor

from cs336_basics.config._validation import _POSITIVE_INTEGER

from .requests import Batch


@attrs.define(kw_only=True)
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: Bool[Tensor, "batch_size"] | None = None

    @property
    def is_greedy(self):
        return self.temperatures is None


@attrs.define(kw_only=True)
class Sampler:
    vocab_size: int = attrs.field(validator=_POSITIVE_INTEGER)
    device: torch.device | str

    def prepare_batch_sampling_args(self, batch: Batch) -> BatchSamplingArgs:
        greedy_rows = [req.sampling_params.temperature is None for req in batch.reqs]

        if all(greedy_rows):
            return BatchSamplingArgs(temperatures=None)

        temperatures: list[float] = []
        top_ks: list[int] = []
        top_ps: list[float] = []

        for req in batch.reqs:
            temperature = (
                req.sampling_params.temperature
                if req.sampling_params.temperature is not None
                else 1
            )
            top_k = (
                req.sampling_params.top_k
                if req.sampling_params.top_k is not None
                else self.vocab_size
            )
            top_p = req.sampling_params.top_p

            temperatures.append(temperature)
            top_ks.append(top_k)
            top_ps.append(top_p)

        greedy_mask = (
            torch.tensor(greedy_rows, dtype=torch.bool, device=self.device)
            if any(greedy_rows)
            else None
        )

        return BatchSamplingArgs(
            temperatures=torch.tensor(temperatures, device=self.device).unsqueeze(-1),
            top_k=torch.tensor(top_ks, device=self.device).unsqueeze(-1),
            top_p=torch.tensor(top_ps, device=self.device).unsqueeze(-1),
            greedy_mask=greedy_mask,
        )

    def sample_impl(
        self,
        last_token_logits: Float[Tensor, "batch_size vocab_size"],
        sampling_params: BatchSamplingArgs,
    ) -> Int[Tensor, "batch_size"]:
        """return tensor at the same device - no cpu sync"""
        if last_token_logits.dim() != 2:
            raise ValueError(
                f"last_token_logits.shape must be [batch_size, voc_size], but got {last_token_logits.dim()=}, {last_token_logits.shape=}"
            )

        if sampling_params.is_greedy:
            return torch.argmax(last_token_logits, dim=-1)

        greedy_tokens = (
            torch.argmax(last_token_logits, dim=-1)
            if sampling_params.greedy_mask is not None
            else None
        )

        last_token_logits = last_token_logits / sampling_params.temperatures

        # top_k first
        if sampling_params.top_k is not None:
            if sampling_params.top_k.shape[:-1] != last_token_logits.shape[:-1]:
                raise ValueError(
                    f"Shapes are wrong, got {sampling_params.top_k.shape=}, {last_token_logits.shape=}"
                )

            # 2 additional tensors [B, V]
            sorted_logits, indices = torch.sort(
                last_token_logits, dim=1, descending=True
            )

            # additional tensor [V]
            range_tensor = torch.arange(
                sorted_logits.shape[1], device=last_token_logits.device
            )

            # additional tensor [B, V]
            mask = range_tensor >= sampling_params.top_k
            sorted_logits.masked_fill_(mask=mask, value=-float("inf"))
            last_token_logits.scatter_(dim=1, src=sorted_logits, index=indices)

        # top_p
        if sampling_params.top_p is not None:
            if sampling_params.top_p.shape[:-1] != last_token_logits.shape[:-1]:
                raise ValueError(
                    f"Shapes are wrong, got {sampling_params.top_p.shape=}, {last_token_logits.shape=}"
                )

            # additional sorting TODO optimization is reuse previous sorting in top_k
            sorted_logits, sorted_indices = torch.sort(
                last_token_logits, descending=True, dim=1
            )
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
            sorted_mask = cumulative_probs > sampling_params.top_p

            # shift the mask by 1 token to the right to account for the first token that is greater than sampling_params.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()

            # at least 1 token always remains in the distribution
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(
                dim=1, src=sorted_mask, index=sorted_indices
            )
            last_token_logits.masked_fill_(mask=indices_to_remove, value=-float("inf"))

        probs = F.softmax(last_token_logits, dim=1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if greedy_tokens is not None:
            if sampling_params.greedy_mask is None:
                raise RuntimeError(
                    "if greedy_tokens is not None -> greedy_mask should be not None"
                )

            sampled_tokens = torch.where(
                sampling_params.greedy_mask,
                greedy_tokens,
                sampled_tokens,
            )
        return sampled_tokens
