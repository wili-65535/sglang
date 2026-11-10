"""Layout-independent commit invariant for DFLASH-family target verify."""

from typing import Optional

import torch


def check_commit_prefix(
    *,
    out_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    target_logits: torch.Tensor,
    verify_lens: Optional[torch.Tensor],
    stride: int,
    forward_ct: int,
) -> None:
    """Assert every committed token equals the target's own greedy output there.

    Holds whatever the verify budget trimmed each request to, so it separates a
    real commit bug from the tie-flips an end-to-end text diff degrades into once
    the budget changes the batch layout. Greedy only; the caller gates on that.
    """
    bs = commit_lens.shape[0]
    commit_lens = commit_lens.to(torch.int64)
    positions = torch.arange(stride, device=commit_lens.device).unsqueeze(0)
    committed = positions < commit_lens.unsqueeze(1)

    target_predict = target_logits.view(bs, stride, -1).argmax(dim=-1)
    content_bad = committed & (out_tokens[:, :stride] != target_predict)

    # A commit past the verify budget reads rows the forward never computed; in
    # compact mode those rows are padding, so catch it before the content check
    # blames the padding.
    budget_bad = (
        torch.zeros_like(commit_lens, dtype=torch.bool)
        if verify_lens is None
        else commit_lens > verify_lens.to(torch.int64)
    )
    if not bool(content_bad.any()) and not bool(budget_bad.any()):
        return

    rows = sorted(
        set(torch.nonzero(content_bad.any(dim=1)).flatten().tolist())
        | set(torch.nonzero(budget_bad).flatten().tolist())
    )
    details = []
    for i in rows[:4]:
        n = int(commit_lens[i])
        budget = "-" if verify_lens is None else int(verify_lens[i])
        details.append(
            f"req_index={i} commit_len={n} verify_len={budget} "
            f"committed={out_tokens[i, :n].tolist()} "
            f"target={target_predict[i, :n].tolist()}"
        )
    raise ValueError(
        f"Spec commit prefix check failed at forward_ct={forward_ct}: "
        f"{len(rows)}/{bs} requests. " + " | ".join(details)
    )
