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

from sglang.srt.speculative.dflash_utils import apply_dflash_simulated_acceptance
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


class TestSimulatedAcceptanceWithinVerifyLens(unittest.TestCase):
    """SGLANG_SIMULATE_ACC_LEN forces the accept length past whatever the
    target actually agreed with, so the trimmed window has to cap it the same
    way the real accept is capped."""

    def _apply(self, *, bs, forced, verify_lens):
        candidates = torch.arange(10, 10 + bs * BLOCK, dtype=torch.int64).view(
            bs, BLOCK
        )
        target_predict = torch.arange(
            90, 90 + bs * BLOCK, dtype=torch.int64
        ).view(bs, BLOCK)
        accept_len = torch.zeros(bs, dtype=torch.int32)
        commit_lens = torch.zeros(bs, dtype=torch.int32)
        bonus = torch.zeros(bs, dtype=torch.int64)
        out_tokens = torch.zeros(bs, BLOCK, dtype=torch.int64)
        apply_dflash_simulated_acceptance(
            candidates=candidates,
            target_predict=target_predict,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            simulate_acc_len=float(forced),
            simulate_acc_method="match-expected",
            simulate_acc_token_mode="real-draft-token",
            verify_lens=(
                None
                if verify_lens is None
                else torch.tensor(verify_lens, dtype=torch.int32)
            ),
        )
        return accept_len, commit_lens, bonus, out_tokens, candidates, target_predict

    def test_the_forced_length_is_capped_per_request(self):
        # Beyond verify_lens the target never ran; compact scatters those rows
        # back as zeros, whose argmax is token 0, so an uncapped force commits
        # a token nothing verified.
        accept_len, commit_lens, bonus, out_tokens, candidates, target = self._apply(
            bs=3, forced=6, verify_lens=[8, 3, 5]
        )
        self.assertEqual(accept_len.tolist(), [5, 2, 4])
        self.assertEqual(commit_lens.tolist(), [6, 3, 5])
        # Each request's bonus is the target prediction at its own boundary.
        for i, k in enumerate(accept_len.tolist()):
            self.assertEqual(int(bonus[i]), int(target[i, k]))
            self.assertEqual(int(out_tokens[i, k]), int(target[i, k]))
            self.assertEqual(
                out_tokens[i, :k].tolist(), candidates[i, 1 : k + 1].tolist()
            )
            self.assertEqual(out_tokens[i, k + 1 :].tolist(), [0] * (BLOCK - k - 1))

    def test_per_request_forced_lengths_are_honored_and_still_capped(self):
        # The synthetic draft draws a length per request, not one per step, so
        # the knob has to carry a tensor through; the window still caps it.
        candidates = torch.arange(10, 10 + 3 * BLOCK, dtype=torch.int64).view(3, BLOCK)
        target_predict = torch.arange(90, 90 + 3 * BLOCK, dtype=torch.int64).view(
            3, BLOCK
        )
        accept_len = torch.zeros(3, dtype=torch.int32)
        commit_lens = torch.zeros(3, dtype=torch.int32)
        bonus = torch.zeros(3, dtype=torch.int64)
        out_tokens = torch.zeros(3, BLOCK, dtype=torch.int64)
        apply_dflash_simulated_acceptance(
            candidates=candidates,
            target_predict=target_predict,
            accept_len=accept_len,
            commit_lens=commit_lens,
            bonus=bonus,
            out_tokens=out_tokens,
            simulate_acc_len=1.0,
            simulate_acc_method="match-expected",
            simulate_acc_token_mode="real-draft-token",
            verify_lens=torch.tensor([8, 3, 8], dtype=torch.int32),
            forced_accept_lens=torch.tensor([7, 5, 0], dtype=torch.int32),
        )
        # Request 1 is capped to verify_lens - 1; the others keep their draw,
        # and simulate_acc_len is ignored entirely.
        self.assertEqual(accept_len.tolist(), [7, 2, 0])
        self.assertEqual(commit_lens.tolist(), [8, 3, 1])

    def test_an_untrimmed_window_keeps_the_uniform_behavior(self):
        # The benchmark knob's existing use passes no window; that path must
        # stay exactly what it was, or every published simulated number moves.
        for verify_lens in (None, [BLOCK] * 2):
            with self.subTest(verify_lens=verify_lens):
                (
                    accept_len,
                    commit_lens,
                    bonus,
                    out_tokens,
                    candidates,
                    target,
                ) = self._apply(bs=2, forced=4, verify_lens=verify_lens)
                self.assertEqual(accept_len.tolist(), [3, 3])
                self.assertEqual(commit_lens.tolist(), [4, 4])
                want = torch.zeros(2, BLOCK, dtype=torch.int64)
                want[:, :3] = candidates[:, 1:4]
                want[:, 3] = target[:, 3]
                self.assertTrue(torch.equal(out_tokens, want))


if __name__ == "__main__":
    unittest.main()
