"""Unit tests for QSA's speculative row-to-request map — CPU-only.

Bug regression: with compact ragged verify each request gets its own verify
budget and the rows are packed end to end, so the packed row count is not a
multiple of the batch size. The mapping divided rows by batch and rejected the
batch outright -- "QSA speculative query rows cannot be mapped to requests:
rows=17, batch=4" -- taking the scheduler down on the first decode step of any
multi-request batch running a trimmed budget.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(verify_lens: list[int]) -> RaggedVerifyLayout:
    lens = torch.tensor(verify_lens, dtype=torch.int32)
    qo_indptr = torch.zeros(len(verify_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(lens, dim=0)
    return RaggedVerifyLayout(
        verify_lens=lens,
        graph_num_tokens=sum(verify_lens),
        extend_start_loc=qo_indptr[:-1].clone(),
        qo_indptr_device=qo_indptr,
        verify_lens_cpu=list(verify_lens),
        total_verify_tokens=sum(verify_lens),
    )


def _batch(bs: int, *, verify_lens=None, extend_seq_lens=None):
    spec_info = SimpleNamespace(
        ragged_verify_layout=None if verify_lens is None else _layout(verify_lens)
    )
    return SimpleNamespace(
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        spec_info=spec_info,
        extend_seq_lens=extend_seq_lens,
    )


class TestQsaRaggedRowToRequest(unittest.TestCase):
    def test_ragged_rows_not_divisible_by_batch(self):
        """The crashing shape: 4 requests, budgets summing to a prime count."""
        batch = _batch(4, verify_lens=[3, 3, 3, 8])
        out = QwenSparseAttnBackend._speculative_row_to_request(batch, 17)
        self.assertEqual(out.tolist(), [0] * 3 + [1] * 3 + [2] * 3 + [3] * 8)

    def test_padding_rows_alias_to_request_zero(self):
        """Rows past the real ones belong to no request; keep gathers in bounds."""
        batch = _batch(2, verify_lens=[2, 3])
        out = QwenSparseAttnBackend._speculative_row_to_request(batch, 8)
        self.assertEqual(out.tolist(), [0, 0, 1, 1, 1, 0, 0, 0])

    def test_uniform_rows_still_divide(self):
        """No layout means the fixed-stride path, which must stay as it was."""
        batch = _batch(3)
        out = QwenSparseAttnBackend._speculative_row_to_request(batch, 6)
        self.assertEqual(out.tolist(), [0, 0, 1, 1, 2, 2])

    def test_layout_outranks_extend_seq_lens(self):
        """Target verify carries both; the budgets are the row widths."""
        batch = _batch(
            2,
            verify_lens=[1, 4],
            extend_seq_lens=torch.tensor([3, 3], dtype=torch.int64),
        )
        out = QwenSparseAttnBackend._speculative_row_to_request(batch, 5)
        self.assertEqual(out.tolist(), [0, 1, 1, 1, 1])

    def test_rows_fewer_than_mapped_is_rejected(self):
        """A window shorter than the budgets is a layout bug, not padding."""
        batch = _batch(2, verify_lens=[4, 4])
        with self.assertRaises(ValueError):
            QwenSparseAttnBackend._speculative_row_to_request(batch, 5)


if __name__ == "__main__":
    unittest.main()
