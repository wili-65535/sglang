"""The tapped target feature a Qwen4-Exp hands a DFLASH draft.

Layers hand each other every hyper-connection stream, so a capture is
hc_count * hidden_size wide. Whether the draft wants that whole thing or the
mean over the streams is a property of the draft checkpoint, not of the
target: Flash-Next's released DFlash2 draft carries an fc of 5 taps x 2560
and so wants the mean, while a post_layer_hc draft wants all four streams.
Getting this wrong is a silent numerical error when the widths happen to
match and a shape error when they do not.
"""

import unittest

import torch

from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC, HIDDEN, TAPS = 4, 2560, 5


def _model(width):
    m = object.__new__(Qwen4ExpModel)
    m.hc_count, m.hidden_size = HC, HIDDEN
    m.dflash_target_feature_width = width
    return m


def _captures(tokens=3):
    g = torch.Generator().manual_seed(0)
    return [
        torch.randn(tokens, HC * HIDDEN, generator=g) for _ in range(TAPS)
    ]


class TestQwen4ExpDflashCapture(unittest.TestCase):
    def test_mean_over_the_streams_when_the_draft_wants_hidden_size(self):
        caps = _captures()
        out = Qwen4ExpModel._narrow_dflash_captures(_model(HIDDEN), caps)
        self.assertEqual(len(out), TAPS)
        for got, src in zip(out, caps):
            self.assertEqual(got.shape[-1], HIDDEN)
            self.assertTrue(
                torch.equal(got, src.unflatten(-1, (HC, HIDDEN)).mean(dim=-2))
            )

    def test_full_streams_pass_through_untouched(self):
        caps = _captures()
        out = Qwen4ExpModel._narrow_dflash_captures(_model(HC * HIDDEN), caps)
        for got, src in zip(out, caps):
            self.assertIs(got, src)

    def test_unset_width_passes_through(self):
        # A target whose layers carry one stream has nothing to narrow, and a
        # draft config without the field must not silently get the mean.
        caps = _captures()
        out = Qwen4ExpModel._narrow_dflash_captures(_model(None), caps)
        for got, src in zip(out, caps):
            self.assertIs(got, src)

    def test_a_width_that_is_neither_is_refused(self):
        # Guessing here would hand the draft's fc a tensor of the right shape
        # built from the wrong features -- wrong numbers, no error.
        with self.assertRaises(ValueError) as e:
            Qwen4ExpModel._narrow_dflash_captures(_model(HIDDEN + 1), _captures())
        self.assertIn("hc_count", str(e.exception))


if __name__ == "__main__":
    unittest.main()
