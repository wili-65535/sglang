"""The confidence a DFlash2 selector draft publishes for the budget.

The lattice scores normalize over the K candidates the selector walks, so
they answer "best of these K" and run systematically high as a confidence.
The budget needs "right out of the whole vocabulary" -- the same quantity a
plain DFLASH draft publishes -- so the walked candidate's raw head logit is
taken against the full-vocabulary log-partition instead.
"""

import math
import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import selector_draft_confidence
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, GAMMA, K = 2, 3, 4


# Large on purpose: the within-K share and the vocabulary probability only
# part company when the tail holds many tokens just below the candidates,
# which is the regime a 248k-token vocabulary is always in.
VOCAB = 100_000


def _lattice(seed=0, gap=0.0):
    """A consistent fixture: top-k and the lse come from one vocabulary.

    Drawing the lse from unrelated logits lets exp(logit - lse) exceed 1,
    a state the real path cannot reach -- such a fixture tests nothing.
    """
    g = torch.Generator().manual_seed(seed)
    vocab_logits = torch.randn(BS, GAMMA, VOCAB, generator=g)
    logits, cols = vocab_logits.topk(K, dim=-1)
    if gap:
        # Push the runner-up candidates down so the within-K softmax saturates
        # on the top one, while the untouched tail keeps its vocabulary mass.
        logits = logits.clone()
        logits[:, :, 1:] -= gap
        vocab_logits.scatter_(-1, cols[:, :, 1:], logits[:, :, 1:])
    lse = torch.logsumexp(vocab_logits, dim=-1)
    return cols, logits, lse


class TestSelectorDraftConfidence(unittest.TestCase):
    def test_matches_the_vocabulary_softmax_of_the_walked_token(self):
        ids, logits, lse = _lattice()
        # Walk a different column per row so the lookup is actually exercised.
        cols = torch.tensor([[0, 2, 3], [1, 0, 2]])
        tokens = ids.gather(-1, cols.unsqueeze(-1)).squeeze(-1)
        got = selector_draft_confidence(
            candidate_ids=ids, raw_logits=logits, lse=lse, tokens=tokens
        )
        want = (logits.gather(-1, cols.unsqueeze(-1)).squeeze(-1) - lse).exp()
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_is_a_probability_not_the_within_candidate_share(self):
        # The within-K softmax of a peaked lattice is ~1; the vocabulary one is
        # not. Confusing them hands the budget a confidence that never drops.
        ids, logits, lse = _lattice(seed=3, gap=20.0)
        tokens = ids[:, :, 0]
        got = selector_draft_confidence(
            candidate_ids=ids, raw_logits=logits, lse=lse, tokens=tokens
        )
        within_k = torch.softmax(logits, dim=-1)[:, :, 0]
        self.assertTrue(bool((within_k > 0.99).all()))
        self.assertTrue(bool((got <= 1.0 + 1e-6).all()))
        # Saturated within K, but strictly below 1 against the vocabulary.
        self.assertTrue(bool((got < within_k).all()))

    def test_duplicate_free_lookup_picks_the_walked_column(self):
        # Top-k ids are distinct, so matching by id is unique; a row whose
        # walked token sits last must not read column 0.
        ids = torch.tensor([[[5, 6, 7, 8]]])
        logits = torch.tensor([[[0.0, 0.0, 0.0, 3.0]]])
        lse = torch.zeros(1, 1)
        got = selector_draft_confidence(
            candidate_ids=ids, raw_logits=logits, lse=lse, tokens=torch.tensor([[8]])
        )
        self.assertAlmostEqual(float(got[0, 0]), math.exp(3.0), places=4)


if __name__ == "__main__":
    unittest.main()
