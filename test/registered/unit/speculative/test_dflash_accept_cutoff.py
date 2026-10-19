"""Greedy accept confined to the compact verify window -- CPU-only.

Compact ragged verify runs only verify_lens[i] of each request's block, but
the draft proposed the whole block and the scatter back to [bs, block_size]
zero-fills the rows that were never run. Row j predicts candidate j+1, so the
chain can reach accept_len == verify_lens by matching off the last real row,
one past what the request's KV budget covers; one row further the target's
"prediction" is the argmax of a zero row.

Pinned here: the clamp is lossless. The candidate at the boundary comes back
as the bonus read from the last real row, so the step still commits it.
"""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import accept_within_verify_lens
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BLOCK = 8


def _accept(candidates, target_predict, verify_lens):
    return accept_within_verify_lens(
        candidates=torch.tensor(candidates, dtype=torch.int64),
        target_predict=torch.tensor(target_predict, dtype=torch.int64),
        verify_lens=torch.tensor(verify_lens, dtype=torch.int32),
    )


class TestAcceptWithinVerifyLens(unittest.TestCase):
    def test_clamp_turns_the_boundary_draft_into_the_bonus(self):
        # Row k-1 predicts candidate k, so an unclamped accept would reach k.
        # Clamped to k-1, the bonus is read from row k-1 and is that same
        # candidate k: the token still leaves the step, commit_lens is still k.
        k = 4
        candidates = [[10, 11, 12, 13, 14, 15, 16, 17]]
        target_predict = [[11, 12, 13, 14, 99, 99, 99, 99]]
        accept_len, bonus, out_tokens, commit_lens = _accept(
            candidates, target_predict, [k]
        )
        self.assertEqual(int(accept_len[0]), k - 1)
        self.assertEqual(int(bonus[0]), 14)
        self.assertEqual(int(commit_lens[0]), k)
        self.assertEqual(out_tokens[0, :k].tolist(), [11, 12, 13, 14])

    def test_a_short_chain_is_untouched_by_the_clamp(self):
        # Rejection before the boundary: the clamp must not move the bonus.
        accept_len, bonus, _, commit_lens = _accept(
            [[10, 11, 12, 13, 14, 15, 16, 17]],
            [[11, 99, 0, 0, 0, 0, 0, 0]],
            [6],
        )
        self.assertEqual(int(accept_len[0]), 1)
        self.assertEqual(int(bonus[0]), 99)
        self.assertEqual(int(commit_lens[0]), 2)

    def test_the_bonus_never_comes_from_a_padding_row(self):
        # The zero-filled rows past verify_lens are the failure this guards:
        # without the clamp the bonus would be read from row k and emitted.
        k = 3
        candidates = [[10, 11, 12, 13, 14, 15, 16, 17]]
        # Rows >= k are the scatter's zero fill, whose argmax is token 0.
        target_predict = [[11, 12, 13, 0, 0, 0, 0, 0]]
        accept_len, bonus, _, _ = _accept(candidates, target_predict, [k])
        self.assertEqual(int(accept_len[0]), k - 1)
        self.assertNotEqual(int(bonus[0]), 0)
        self.assertEqual(int(bonus[0]), 13)

    def test_each_request_is_clamped_by_its_own_budget(self):
        # The whole point of compact: budgets differ per request within a step.
        candidates = [[10, 11, 12, 13, 14, 15, 16, 17]] * 3
        target_predict = [[11, 12, 13, 14, 15, 16, 17, 18]] * 3
        accept_len, bonus, _, commit_lens = _accept(
            candidates, target_predict, [1, 5, 8]
        )
        self.assertEqual(accept_len.tolist(), [0, 4, 7])
        self.assertEqual(commit_lens.tolist(), [1, 5, 8])
        self.assertEqual(bonus.tolist(), [11, 15, 18])

    def test_a_single_row_budget_still_commits_the_anchor_step(self):
        # verify_lens == 1 leaves no draft to accept; the step must still make
        # progress through the bonus, or the request stalls forever.
        accept_len, bonus, _, commit_lens = _accept(
            [[10, 11, 12, 13, 14, 15, 16, 17]],
            [[11, 0, 0, 0, 0, 0, 0, 0]],
            [1],
        )
        self.assertEqual(int(accept_len[0]), 0)
        self.assertEqual(int(commit_lens[0]), 1)
        self.assertEqual(int(bonus[0]), 11)


if __name__ == "__main__":
    unittest.main()
