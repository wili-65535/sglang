"""The PLE N-gram history a verify step commits, checked against the block.

The history is n - 1 tokens wide and ends at the last committed block row, so
after committing k rows it must equal block tokens [k - (n-1), k). Getting that
window off by one is invisible end to end -- the model keeps generating fluent
text from a slightly wrong N-gram context -- so the probe that pins it down is
itself worth pinning down.
"""

import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import check_ple_verify_history
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BLOCK = 8
WIDTH = 2  # ngram_size 3 -> history is 2 tokens


def _pool(history_rows):
    history = torch.tensor(history_rows, dtype=torch.int64)
    return SimpleNamespace(
        ngram_pool=SimpleNamespace(enabled=True),
        get_ngram_context=lambda indices: history[indices],
    )


def _run(*, blocks, commits, history_rows):
    return check_ple_verify_history(
        draft_tokens=torch.tensor(blocks, dtype=torch.int64),
        commit_lens=torch.tensor(commits, dtype=torch.int32),
        req_pool_indices=torch.arange(len(commits), dtype=torch.int64),
        req_to_token_pool=_pool(history_rows),
        forward_ct=7,
    )


class TestPleVerifyHistory(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(
            "sglang.srt.speculative.dflash_worker_v2"
        )

    def test_window_ends_at_the_last_committed_row(self):
        block = [[100, 101, 102, 103, 104, 105, 106, 107]]
        # commit 3 -> rows 0..2 committed -> history is rows 1 and 2.
        with self.assertNoLogs(self.logger, level="WARNING"):
            _run(blocks=block, commits=[3], history_rows=[[101, 102]])

    def test_off_by_one_window_is_reported(self):
        block = [[100, 101, 102, 103, 104, 105, 106, 107]]
        # Rows 2 and 3: one past the commit, the shape a stale roll-in makes.
        with self.assertLogs(self.logger, level="WARNING") as got:
            _run(blocks=block, commits=[3], history_rows=[[102, 103]])
        self.assertIn("forward_ct=7", got.output[0])

    def test_commit_shorter_than_the_history_is_skipped(self):
        # commit 1 leaves the older half of the history from before this block,
        # which the block cannot supply; guessing it would make noise, not a bug.
        block = [[100, 101, 102, 103, 104, 105, 106, 107]]
        with self.assertNoLogs(self.logger, level="WARNING"):
            _run(blocks=block, commits=[1], history_rows=[[55, 100]])

    def test_one_bad_request_does_not_hide_behind_a_good_one(self):
        blocks = [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [200, 201, 202, 203, 204, 205, 206, 207],
        ]
        with self.assertLogs(self.logger, level="WARNING") as got:
            _run(blocks=blocks, commits=[3, 5], history_rows=[[101, 102], [202, 203]])
        self.assertIn("req_index=1", got.output[0])
        self.assertIn("1/2 requests", got.output[0])

    def test_disabled_pool_is_a_no_op(self):
        pool = SimpleNamespace(ngram_pool=SimpleNamespace(enabled=False))
        with self.assertNoLogs(self.logger, level="WARNING"):
            check_ple_verify_history(
                draft_tokens=torch.zeros((1, BLOCK), dtype=torch.int64),
                commit_lens=torch.tensor([3], dtype=torch.int32),
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
                req_to_token_pool=pool,
                forward_ct=0,
            )


if __name__ == "__main__":
    unittest.main()
