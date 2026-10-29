from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
    is_dense_head_weight,
)

logger = logging.getLogger(__name__)


def dflash_sts_temperature_grid() -> torch.Tensor:
    """The temperatures a collected shard carries its partition sums for.

    DFLASH's confidence is a softmax probability, so unlike DSpark's single
    confidence logit it cannot be recalibrated after the fact from one number
    per draft token: 1 / sum_v exp((logit_v - max) / T) needs the whole row.
    Collecting the sum on a fixed grid keeps the statistic exact and small,
    at the cost of fitting only to a temperature on the grid.
    """
    return torch.logspace(math.log10(0.1), math.log10(10.0), steps=41)


def partition_sums_over_grid(
    *,
    logits: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[n, vocab] logits -> ([n, len(grid)] sum_v exp((logit_v - max) / T),
    [n] row max). The max is returned because a sharded row has to be
    re-centered on the global max before the ranks are summed."""
    row_max = logits.amax(dim=-1).float()
    gaps = logits.float() - row_max.unsqueeze(-1)
    sums = [gaps.div(float(temperature)).exp().sum(dim=-1) for temperature in grid]
    return torch.stack(sums, dim=-1), row_max


def merge_shard_partition_sums(
    *,
    partitions: torch.Tensor,
    maxima: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Per-shard ([tp, n, len(grid)] sums, [tp, n] shard maxima) -> the global
    [n, len(grid)] sums. Each shard is re-centered on the global max at that
    grid point's own temperature; re-centering at a single temperature is what
    makes tp=1 agree and every sharded run disagree."""
    global_max = maxima.amax(dim=0)
    recentered = (maxima - global_max).unsqueeze(-1).div(grid.view(1, 1, -1))
    return (partitions * recentered.exp()).sum(dim=0)


def collect_dflash_partition_sums(
    *,
    hidden_states: torch.Tensor,
    lm_head,
    grid: torch.Tensor,
    tp_group=None,
) -> Optional[torch.Tensor]:
    """[n, hidden] drafted-row hidden states -> [n, len(grid)] partition sums,
    or None when the head's shape rules the derivation out (the same quantized
    and added-vocab cases draft_confidence_from_head refuses).

    1 / sums[:, k] is the confidence the runtime would publish at grid[k], so
    the column at T=1 reproduces the uncalibrated confidence exactly.
    """
    weight = getattr(lm_head, "weight", None)
    if not is_dense_head_weight(weight):
        return None
    shard = getattr(lm_head, "shard_indices", None)
    if shard is None:
        num_org, num_added = int(weight.shape[0]), 0
    else:
        num_org = int(shard.num_org_elements)
        num_added = int(shard.num_added_elements)
    if num_added > 0 or num_org <= 0:
        return None

    hs = hidden_states
    if hs.dtype != weight.dtype:
        hs = hs.to(weight.dtype)
    logits = torch.matmul(hs, weight[:num_org].T)
    grid = grid.to(device=logits.device, dtype=torch.float32)
    partitions, row_max = partition_sums_over_grid(logits=logits, grid=grid)

    tp_size = 1 if tp_group is None else int(tp_group.world_size)
    if tp_size == 1:
        return partitions

    n, num_grid = partitions.shape
    device = partitions.device
    gathered_partitions = torch.empty(
        (tp_size * n, num_grid), dtype=torch.float32, device=device
    )
    gathered_max = torch.empty((tp_size * n,), dtype=torch.float32, device=device)
    tp_group.all_gather_into_tensor(gathered_partitions, partitions)
    tp_group.all_gather_into_tensor(gathered_max, row_max)
    return merge_shard_partition_sums(
        partitions=gathered_partitions.view(tp_size, n, num_grid),
        maxima=gathered_max.view(tp_size, n),
        grid=grid,
    )


class DflashStsDataRecorder:
    """Buffers collected partition sums and writes them as .pt shards for
    sglang.benchmark.dflash_sts_fit."""

    def __init__(self, *, path_stem: str, gamma: int, flush_every: int) -> None:
        self.path_stem = path_stem
        self.gamma = int(gamma)
        self.flush_every = int(flush_every)
        self._partition_buffer: list[torch.Tensor] = []
        self._prefix_mask_buffer: list[torch.Tensor] = []
        self._grid: Optional[torch.Tensor] = None
        self._shard_ct = 0

    def record(
        self,
        *,
        partitions: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        grid: torch.Tensor,
    ) -> None:
        self._grid = grid.detach().to(device="cpu", dtype=torch.float32)
        positions = torch.arange(self.gamma).view(1, -1)
        counts = (
            num_correct_drafts.detach().to(device="cpu", dtype=torch.int64).view(-1, 1)
        )
        self._partition_buffer.append(
            partitions.detach().to(device="cpu", dtype=torch.float32)
        )
        self._prefix_mask_buffer.append((positions < counts).to(torch.float32))
        if len(self._partition_buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._partition_buffer:
            return
        shard_path = Path(f"{self.path_stem}.{self._shard_ct}.pt")
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "partitions": torch.cat(self._partition_buffer, dim=0),
                "prefix_mask": torch.cat(self._prefix_mask_buffer, dim=0),
                "temperature_grid": self._grid,
            },
            shard_path,
        )
        self._partition_buffer.clear()
        self._prefix_mask_buffer.clear()
        self._shard_ct += 1


class DflashStsCollector:
    """Per-step sink for DFLASH STS calibration data.

    The two halves of one sample arrive at different points of the verify
    step -- the partition sums with the draft, the correctness labels with
    the accept -- so the draft half is held until the accept half lands.
    """

    def __init__(
        self, *, path_stem: str, gamma: int, flush_every: int, is_writer: bool
    ) -> None:
        self._gamma = int(gamma)
        self._grid = dflash_sts_temperature_grid()
        # Every rank derives the sums -- the shard merge is a collective, so a
        # rank that skipped it would hang the others -- but they all hold the
        # same global values, so only one rank may write them.
        self._recorder = (
            DflashStsDataRecorder(
                path_stem=path_stem, gamma=int(gamma), flush_every=int(flush_every)
            )
            if is_writer
            else None
        )
        self._pending: Optional[torch.Tensor] = None
        self._warned_no_hidden = False
        self._warned_trimmed = False
        self._warned_drift = False

    def observe_draft(
        self, *, draft_hidden, bs: int, lm_head, tp_group, runtime_confidence
    ) -> None:
        self._pending = None
        if draft_hidden is None:
            if not self._warned_no_hidden:
                logger.warning(
                    "DFLASH STS collection needs the draft hidden states, which "
                    "this draft path does not publish; run with "
                    "SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1. Collecting nothing."
                )
                self._warned_no_hidden = True
            return
        drafted = draft_hidden.view(int(bs), -1, draft_hidden.shape[-1])[:, 1:, :]
        partitions = collect_dflash_partition_sums(
            hidden_states=drafted.reshape(-1, drafted.shape[-1]),
            lm_head=lm_head,
            grid=self._grid,
            tp_group=tp_group,
        )
        if partitions is not None:
            self._pending = partitions.view(int(bs), self._gamma, -1)
            self._check_identity_column(runtime_confidence)

    def _check_identity_column(self, runtime_confidence) -> None:
        """The T=1 column must be the confidence the runtime publishes.

        Only a sharded run can tell the two apart: with tp=1 both are one
        reduction over the same logits, while with tp>1 they are separate
        merges whose re-centering has to agree.
        """
        if runtime_confidence is None or self._warned_drift:
            return
        identity = int((self._grid - 1.0).abs().argmin())
        collected = self._pending[:, :, identity].reciprocal()
        drift = (collected - runtime_confidence.float()).abs() / collected.clamp_min(
            1e-12
        )
        worst = float(drift.max())
        if worst > 1e-3:
            logger.warning(
                "DFLASH STS collection disagrees with the published confidence "
                "by %.3e at T=1; the collected partitions and the runtime's "
                "merge have drifted apart.",
                worst,
            )
            self._warned_drift = True

    def observe_synthetic_step(self, *, partitions, accept_lens) -> None:
        """Record a step the synthetic draft drove. Its labels come from the
        draw, not from the target: what the target agreed with says nothing
        about a draft token this mode invented."""
        self._pending = None
        if self._recorder is None:
            return
        self._recorder.record(
            partitions=partitions, num_correct_drafts=accept_lens, grid=self._grid
        )

    def observe_verify(self, *, candidates, target_predict, verify_lens) -> None:
        pending, self._pending = self._pending, None
        if pending is None or target_predict is None or self._recorder is None:
            return
        if verify_lens is not None:
            # A trimmed window stops the accept early, so the drafts past it
            # are unlabeled rather than wrong -- recording them would teach the
            # fit that a high confidence precedes a rejection.
            if not self._warned_trimmed:
                logger.warning(
                    "DFLASH STS collection skipped a step whose verify window "
                    "was trimmed; collect with SGLANG_RAGGED_VERIFY_MODE=static."
                )
                self._warned_trimmed = True
            return
        num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self._recorder.record(
            partitions=pending,
            num_correct_drafts=num_correct_drafts,
            grid=self._grid,
        )


class DflashSyntheticDraft:
    """A stand-in draft whose confidence is calibrated at T=1 by construction.

    An untrained draft collects nothing a temperature can be fit to: it is
    never right, so every label is a reject, and its output is flat, so every
    confidence is 1/vocab. This replaces both halves with a draw that agrees
    with itself -- the confidence comes from synthetic logits, and a draft
    token is accepted with exactly that probability -- which makes 1.0 the
    answer a fit over the collected data has to return.

    The logits are drawn over a small synthetic vocabulary rather than the
    model's: nothing here touches the real head, and the partition sum's cost
    is linear in the vocabulary.
    """

    _VOCAB: int = 2048
    _SHARPNESS_SPAN: float = 8.0

    def __init__(
        self, *, gamma: int, target_accept: float, device, seed: int = 0
    ) -> None:
        self._gamma = int(gamma)
        self._device = device
        self._grid = dflash_sts_temperature_grid().to(device=device)
        self._identity = int((self._grid - 1.0).abs().argmin())
        self._generator = torch.Generator(device=device).manual_seed(int(seed))
        center = self._solve_sharpness(float(target_accept))
        self._sharpness_low, self._sharpness_high = self._span(center)
        # Measured here, not on demand: every rank seeds the same generator and
        # must consume it in lockstep, so a draw taken on one rank alone would
        # desynchronize the confidences and with them the verify budget.
        self.realized_accept = self._mean_accept_over(
            self._sharpness_low, self._sharpness_high
        )

    def _span(self, center: float) -> Tuple[float, float]:
        """The sharpness range a step draws from. It has to be a range, not one
        value: a draft whose every row is equally peaked publishes one
        confidence, and one confidence cannot tell two temperatures apart."""
        half = self._SHARPNESS_SPAN / 2
        return max(0.5, center - half), max(0.5, center - half) + self._SHARPNESS_SPAN

    def _mean_accept_over(self, low: float, high: float, rows: int = 512) -> float:
        sharpness = low + torch.rand(
            rows, 1, generator=self._generator, device=self._device
        ) * (high - low)
        logits = (
            torch.randn(
                rows, self._VOCAB, generator=self._generator, device=self._device
            )
            * sharpness
        )
        sums, _ = partition_sums_over_grid(logits=logits, grid=self._grid)
        return float(sums[:, self._identity].reciprocal().mean())

    def _solve_sharpness(self, target_accept: float) -> float:
        """The range center whose mean max-softmax probability is
        target_accept. Monotone in the center, so a bisection lands it; done
        once, on the synthetic vocabulary, and the realized rate is reported
        by the caller because the solve is over a sampled mean."""
        low, high = 0.5, 60.0
        for _ in range(24):
            mid = (low + high) / 2
            if self._mean_accept_over(*self._span(mid)) < target_accept:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def step(self, bs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step's ([bs, gamma, grid] partition sums, [bs, gamma] confidence,
        [bs] accepted draft counts)."""
        rows = int(bs) * self._gamma
        sharpness = self._sharpness_low + torch.rand(
            rows, 1, generator=self._generator, device=self._device
        ) * (self._sharpness_high - self._sharpness_low)
        logits = (
            torch.randn(
                rows, self._VOCAB, generator=self._generator, device=self._device
            )
            * sharpness
        )
        sums, _ = partition_sums_over_grid(logits=logits, grid=self._grid)
        confidence = sums[:, self._identity].reciprocal().view(int(bs), self._gamma)

        drawn = torch.rand(
            int(bs), self._gamma, generator=self._generator, device=self._device
        )
        # The chain stops at the first rejected draft, so the accepted count is
        # the length of the leading run of hits.
        miss = drawn >= confidence
        accept_lens = torch.where(
            miss.any(dim=1),
            miss.to(torch.int32).argmax(dim=1),
            torch.full((int(bs),), self._gamma, dtype=torch.int64, device=self._device),
        ).to(torch.int32)
        return sums.view(int(bs), self._gamma, -1), confidence, accept_lens
