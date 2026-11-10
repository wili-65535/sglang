"""Unit tests for ``ForwardBatch.compute_spec_mrope_positions`` — CPU-only.

Bug regression: with compact ragged verify the verify window is padded up to
``layout.graph_num_tokens`` so it can be replayed by a cuda graph, while the
per-token mrope delta was expanded from ``verify_lens`` alone. The two lengths
then disagree and the add raises ``RuntimeError: The size of tensor a (17) must
match the size of tensor b (3)``, taking the scheduler down on the first decode
step of any mrope model running a trimmed verify budget.

CPU-only: the method is index arithmetic over small tensors.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _layout(verify_lens: list[int], graph_num_tokens: int) -> RaggedVerifyLayout:
    lens = torch.tensor(verify_lens, dtype=torch.int32)
    qo_indptr = torch.zeros(len(verify_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(lens, dim=0)
    return RaggedVerifyLayout(
        verify_lens=lens,
        graph_num_tokens=graph_num_tokens,
        extend_start_loc=qo_indptr[:-1].clone(),
        qo_indptr_device=qo_indptr,
        verify_lens_cpu=list(verify_lens),
        total_verify_tokens=sum(verify_lens),
    )


def _run(verify_lens, graph_num_tokens, deltas):
    """Call the method against minimal stand-ins; returns mrope_positions."""
    bs = len(verify_lens)
    num_tokens = graph_num_tokens
    fb = SimpleNamespace(
        seq_lens=torch.zeros(bs, dtype=torch.int32),
        mrope_positions=None,
    )
    mm_inputs = [
        (
            None
            if d is None
            else SimpleNamespace(
                mrope_position_delta=torch.tensor([[d]], dtype=torch.int64)
            )
        )
        for d in deltas
    ]
    batch = SimpleNamespace(
        multimodal_inputs=mm_inputs,
        spec_info=SimpleNamespace(
            positions=torch.arange(num_tokens, dtype=torch.int64),
            ragged_verify_layout=_layout(verify_lens, graph_num_tokens),
        ),
    )
    ForwardBatch.compute_spec_mrope_positions(fb, SimpleNamespace(device="cpu"), batch)
    return fb.mrope_positions


class TestSpecMropeRaggedPadding(unittest.TestCase):
    def test_padded_window_does_not_raise(self):
        """A window padded past sum(verify_lens) is the crashing shape."""
        out = _run(verify_lens=[3], graph_num_tokens=17, deltas=[5])
        self.assertEqual(out.shape, (3, 17))
        # The three real tokens carry the request's delta.
        self.assertEqual(out[0, :3].tolist(), [5, 6, 7])
        # The padding tail carries none: it belongs to no request.
        self.assertEqual(out[0, 3:].tolist(), list(range(3, 17)))

    def test_exact_fit_is_unchanged(self):
        """Without padding the old expansion is already right; keep it so."""
        out = _run(verify_lens=[2, 3], graph_num_tokens=5, deltas=[10, 100])
        self.assertEqual(out[0].tolist(), [10, 11, 102, 103, 104])

    def test_delta_follows_the_row_not_the_stride(self):
        """Each token takes its own request's delta across ragged rows."""
        out = _run(verify_lens=[1, 4], graph_num_tokens=8, deltas=[7, 20])
        self.assertEqual(out[0, :5].tolist(), [7, 21, 22, 23, 24])
        self.assertEqual(out[0, 5:].tolist(), [5, 6, 7])

    def test_text_only_batch_has_no_delta(self):
        """mm_inputs all None is the text-only path; deltas are zero."""
        out = _run(verify_lens=[2], graph_num_tokens=6, deltas=[None])
        self.assertEqual(out[0].tolist(), list(range(6)))

    def test_all_three_rope_sections_match(self):
        """The three mrope sections are the same row repeated."""
        out = _run(verify_lens=[3], graph_num_tokens=9, deltas=[2])
        self.assertTrue(torch.equal(out[0], out[1]))
        self.assertTrue(torch.equal(out[1], out[2]))


if __name__ == "__main__":
    unittest.main()
