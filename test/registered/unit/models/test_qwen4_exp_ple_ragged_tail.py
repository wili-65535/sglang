"""The PLE token layout when a ragged verify batch is padded to a graph tier.

A captured decode graph always processes its whole tier, but the ragged
verify layout staged into it covers only the live tokens, so the tail rows
between the two belong to no request. They are computed, not skipped: the
captured kernels recompute each token's row and offset from the refreshed
qo_indptr, so the tail lands past its request's stride. Writing it at that
offset indexes the dense [bs, stride] working buffer out of bounds, which on
CUDA is a device-side assert that poisons the context and makes the next
graph launch segfault.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen4_exp import _prepare_ple_batch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STRIDE = 8  # speculative_num_draft_tokens
TIER = 32  # captured graph tier
EOS = 999
NGRAM = 3


def _pool():
    return SimpleNamespace(
        ple_window_cache=None,
        get_mamba_indices=lambda idx: idx,
        get_ngram_context=lambda idx: torch.zeros(
            (idx.shape[0], NGRAM - 1), dtype=torch.int64
        ),
    )


def _batch(verify_lens):
    layout = RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=list(verify_lens),
        device="cpu",
        grid=[TIER],
    )
    bs = len(verify_lens)
    return layout, SimpleNamespace(
        tbo_parent_token_range=None,
        spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
        spec_info=SimpleNamespace(
            topk=1, draft_token_num=STRIDE, ragged_verify_layout=layout
        ),
        _original_forward_mode=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        # The replay view reports the padded tier, not the live token count.
        global_num_token_non_padded_cpu=TIER,
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        out_cache_loc=torch.ones(TIER, dtype=torch.int64),
    )


def _run(verify_lens):
    layout, fb = _batch(verify_lens)
    input_ids = torch.arange(1, TIER + 1, dtype=torch.int64)
    with patch("sglang.srt.models.qwen4_exp.get_req_to_token_pool", _pool):
        return _prepare_ple_batch(
            input_ids,
            fb,
            ngram_size=NGRAM,
            ngram_eos_token_id=EOS,
        )


class TestQwen4ExpPleRaggedTail(unittest.TestCase):
    def test_a_tail_past_the_layout_coverage_does_not_index_out_of_bounds(self):
        # 22 live tokens staged into a 32-token tier: rows 22..31 are the tail.
        _run([8, 6, 5, 3])

    def test_the_tail_does_not_overwrite_a_full_row(self):
        # A row at the full stride has no spare column, so a tail token
        # clamped into it would replace that row's last real token.
        batch = _run([8, 8, 4, 2])
        row_width = batch.row_width
        last_full = batch.ngram_context[1, -row_width:]
        self.assertEqual(last_full.tolist(), list(range(9, 17)))

    def test_live_rows_keep_their_own_tokens(self):
        batch = _run([8, 6, 5, 3])
        rows = batch.ngram_context[:, -batch.row_width :]
        self.assertEqual(rows[0].tolist(), list(range(1, 9)))
        self.assertEqual(rows[1][:6].tolist(), list(range(9, 15)))
        self.assertEqual(rows[2][:5].tolist(), list(range(15, 20)))
        self.assertEqual(rows[3][:3].tolist(), list(range(20, 23)))


if __name__ == "__main__":
    unittest.main()
