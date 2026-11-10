from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    gather_strided_to_compact_into,
    scatter_compact_to_strided_into,
)


def ragged_verify_dense_scatter_indices(
    *,
    query_start_loc: torch.Tensor,
    seq_len: int,
    draft_token_num: int,
) -> torch.Tensor:
    """Dense [bs, draft_token_num] slot index per packed ragged-verify token.

    Rows never exceed draft_token_num under either layout variant (cap for
    graph replay, planner construction for eager -- see
    RaggedVerifyLayout.padded_to_bucket), so in-row offsets stay in-row;
    tokens past the layout's coverage collapse into one ghost row at index
    bs * draft_token_num.
    """
    batch_size = query_start_loc.shape[0] - 1
    token_pos = torch.arange(seq_len, device=query_start_loc.device, dtype=torch.int32)
    token_slots = torch.searchsorted(query_start_loc[1:], token_pos, right=True)
    return (
        token_slots * draft_token_num
        + (token_pos - query_start_loc[token_slots]).to(torch.int64)
    ).clamp_(max=batch_size * draft_token_num)


def scatter_ragged_to_dense(
    *,
    packed: torch.Tensor,
    dense_token_indices: torch.Tensor,
    num_dense_tokens: int,
) -> torch.Tensor:
    """Packed [total_tokens, D] -> dense [num_dense_tokens, D]; the trailing
    ghost slot absorbs tokens past the layout's coverage."""
    dense = packed.new_zeros(num_dense_tokens + 1, packed.shape[-1])
    dense.index_copy_(0, dense_token_indices, packed)
    return dense[:num_dense_tokens]


def gather_dense_to_ragged(
    *,
    dense_flat: torch.Tensor,
    dense_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Inverse of scatter_ragged_to_dense. The appended zero ghost row keeps
    uncovered tail tokens finite; their values are discarded downstream."""
    num_dense_tokens = dense_flat.shape[0]
    padded = dense_flat.new_zeros(num_dense_tokens + 1, dense_flat.shape[-1])
    padded[:num_dense_tokens] = dense_flat
    return padded[dense_token_indices]


def scatter_ragged_to_dense_into(
    *,
    packed: torch.Tensor,
    verify_lens: torch.Tensor,
    start: torch.Tensor,
    out: torch.Tensor,
    draft_token_num: int,
) -> torch.Tensor:
    """scatter_ragged_to_dense in one launch, into a caller-owned buffer.

    The kernel writes every row of `out` -- the covered ones from `packed`, the
    rest zero -- so a buffer reused across layers carries nothing stale. There
    is no ghost row: tokens past the layout's coverage are simply not read.
    """
    return scatter_compact_to_strided_into(
        compact=packed,
        verify_lens=verify_lens,
        start=start,
        out=out,
        stride=draft_token_num,
        fill_value=0,
    )


def gather_dense_to_ragged_into(
    *,
    dense_flat: torch.Tensor,
    dense_token_indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """gather_dense_to_ragged in one launch, into a caller-owned buffer.

    Writes every row of `out`; rows whose slot is past `dense_flat` are the
    uncovered tail and read back as zero, as the padded-ghost-row version did.
    """
    return gather_strided_to_compact_into(
        strided=dense_flat,
        slot_index=dense_token_indices,
        out=out,
    )
