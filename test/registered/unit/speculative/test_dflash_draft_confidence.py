"""DFLASH draft confidence derived from the draft logits -- CPU-only.

The adaptive verify budget needs a per-token confidence. A DFLASH draft has no
confidence head, so the chosen token's vocab-wide softmax probability is read
off the logits the greedy draft sampler already computes. Greedy picks the
argmax, so that probability is exactly 1/sum(exp(logit - max)) and the chosen
logit never has to be gathered back.

The derivations pinned here: the reciprocal-of-shifted-partition identity, the
per-shard re-centering that makes the tensor-parallel result independent of
which rank owns the global max, and the STS temperature that rides inside the
exponent of both.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_worker_v2 import (
    _DflashDraftSampler,
    _merge_shard_partitions,
    _shifted_exp_partition,
    draft_confidence_from_head,
    tile_sts_temperatures,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _reference_max_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1).amax(dim=-1)


class TestDflashDraftConfidence(unittest.TestCase):
    def test_partition_reciprocal_is_the_max_softmax_probability(self):
        torch.manual_seed(0)
        logits = torch.randn(16, 512) * 4.0
        got = _shifted_exp_partition(logits.clone()).reciprocal()
        torch.testing.assert_close(got, _reference_max_prob(logits))

    def test_peaked_and_flat_rows_land_at_the_ends_of_the_range(self):
        # A row that is one spike has probability ~1; a constant row has 1/vocab.
        vocab = 512
        peaked = torch.full((1, vocab), -30.0)
        peaked[0, 7] = 30.0
        flat = torch.zeros(1, vocab)
        logits = torch.cat([peaked, flat], dim=0)
        got = _shifted_exp_partition(logits.clone()).reciprocal()
        self.assertGreater(float(got[0]), 0.999)
        self.assertAlmostEqual(float(got[1]), 1.0 / vocab, places=6)

    def test_shard_merge_matches_the_unsharded_probability(self):
        torch.manual_seed(1)
        logits = torch.randn(8, 1024) * 4.0
        shards = logits.chunk(4, dim=-1)
        partitions = torch.stack([_shifted_exp_partition(s.clone()) for s in shards])
        maxima = torch.stack([s.amax(dim=-1).float() for s in shards])
        got = _merge_shard_partitions(partitions=partitions, maxima=maxima)
        torch.testing.assert_close(got, _reference_max_prob(logits))

    def test_shard_merge_is_independent_of_which_rank_owns_the_max(self):
        # Drop the re-centering and this is the case that breaks: every shard's
        # partition is relative to its own max, so they cannot just be summed.
        vocab, rows = 256, 4
        base = torch.randn(rows, vocab * 2) * 2.0
        for owner in (0, 1):
            logits = base.clone()
            logits[:, owner * vocab] += 25.0
            shards = logits.chunk(2, dim=-1)
            partitions = torch.stack(
                [_shifted_exp_partition(s.clone()) for s in shards]
            )
            maxima = torch.stack([s.amax(dim=-1).float() for s in shards])
            got = _merge_shard_partitions(partitions=partitions, maxima=maxima)
            torch.testing.assert_close(got, _reference_max_prob(logits))

    def test_bf16_logits_stay_close_to_the_fp32_probability(self):
        # Measured on this input: summing the bf16 exp in fp32 lands at 4.5e-4
        # relative, letting it accumulate in bf16 at 2.6e-3. The bound sits
        # between the two, so dropping the fp32 accumulation turns this red.
        # The error compounds through the budget's prefix product, so the
        # cheaper reduction is not good enough.
        torch.manual_seed(2)
        logits = (torch.randn(32, 2048) * 4.0).to(torch.bfloat16)
        got = _shifted_exp_partition(logits.clone()).reciprocal()
        want = _reference_max_prob(logits)
        self.assertLess(float((got - want).abs().max() / want.max()), 1e-3)

    def test_sampler_publishes_confidence_for_each_drafted_position(self):
        torch.manual_seed(3)
        vocab, hidden, block_size, bs = 64, 8, 8, 3
        weight = torch.randn(vocab, hidden, dtype=torch.float32) * 0.3
        sampler = _DflashDraftSampler(
            weight=weight,
            block_size=block_size,
            num_org=vocab,
            org_vocab_start=0,
            max_bs=bs,
            tp_group=None,
        )
        hidden_states = torch.randn(bs * block_size, hidden)
        sampler(hidden_states)

        # Block position 0 is the seeded bonus token; the drafted positions are 1:.
        drafted = hidden_states.view(bs, block_size, hidden)[:, 1:, :]
        logits = drafted.reshape(-1, hidden) @ weight.T
        torch.testing.assert_close(
            sampler.confidence[:bs].reshape(-1), _reference_max_prob(logits)
        )
        torch.testing.assert_close(
            sampler.out[: bs * (block_size - 1)], logits.argmax(dim=-1)
        )

    def test_sampler_confidence_rows_follow_the_request(self):
        # Confidence is [bs, gamma] while the tokens are flat; a wrong view here
        # would hand the budget another request's scores with no other symptom.
        torch.manual_seed(4)
        vocab, hidden, block_size = 64, 8, 8
        weight = torch.randn(vocab, hidden, dtype=torch.float32) * 0.3
        sampler = _DflashDraftSampler(
            weight=weight,
            block_size=block_size,
            num_org=vocab,
            org_vocab_start=0,
            max_bs=2,
            tp_group=None,
        )
        peaked = torch.zeros(block_size, hidden)
        peaked[:] = weight[5] * 40.0
        blurry = torch.zeros(block_size, hidden)
        sampler(torch.cat([peaked, blurry], dim=0))

        self.assertGreater(float(sampler.confidence[0].min()), 0.99)
        self.assertAlmostEqual(float(sampler.confidence[1].max()), 1.0 / vocab, places=5)


class TestEagerDraftConfidence(unittest.TestCase):
    """The eager draft path discards its logits, so confidence comes from a
    second pass over the head. It has to agree with the folded path's buffer
    and has to refuse the head shapes it cannot derive."""

    def _head(self, weight, shard=None):
        return SimpleNamespace(weight=weight, shard_indices=shard)

    def test_matches_the_folded_sampler_on_the_same_inputs(self):
        # The two paths are separate implementations of the same quantity; a
        # drift between them would show up only as a worse budget.
        torch.manual_seed(5)
        vocab, hidden, block_size, bs = 64, 8, 8, 3
        weight = torch.randn(vocab, hidden) * 0.3
        sampler = _DflashDraftSampler(
            weight=weight,
            block_size=block_size,
            num_org=vocab,
            org_vocab_start=0,
            max_bs=bs,
            tp_group=None,
        )
        hidden_states = torch.randn(bs * block_size, hidden)
        sampler(hidden_states)
        drafted = hidden_states.view(bs, block_size, hidden)[:, 1:, :]
        eager = draft_confidence_from_head(
            hidden_states=drafted.reshape(-1, hidden),
            lm_head=self._head(weight),
            bs=bs,
            gamma=block_size - 1,
        )
        torch.testing.assert_close(eager, sampler.confidence[:bs])

    def test_refuses_an_added_vocab_shard(self):
        # An added-vocab shard splits a row across two weight slices, which this
        # single-slice derivation would silently read as the whole vocabulary.
        weight = torch.randn(64, 8)
        shard = SimpleNamespace(
            num_org_elements=32, org_vocab_start_index=0, num_added_elements=4
        )
        self.assertIsNone(
            draft_confidence_from_head(
                hidden_states=torch.randn(7, 8),
                lm_head=self._head(weight, shard),
                bs=1,
                gamma=7,
            )
        )

    def test_refuses_a_head_without_a_dense_weight(self):
        self.assertIsNone(
            draft_confidence_from_head(
                hidden_states=torch.randn(7, 8),
                lm_head=SimpleNamespace(weight=None, shard_indices=None),
                bs=1,
                gamma=7,
            )
        )


class TestDflashStsTemperatures(unittest.TestCase):
    """Post-hoc calibration of the draft confidence.

    DFLASH has no confidence head, so its calibration is a softmax temperature
    applied inside the exponent: p_T = 1 / sum(exp((logit - max) / T_j)). It
    cannot be applied to the finished probability, and the tensor-parallel
    merge re-centers on the global max, so the same temperature has to divide
    the re-centering too.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.bs, self.gamma, self.vocab = 3, 7, 257
        self.logits = torch.randn(self.bs * self.gamma, self.vocab) * 4.0
        self.temperatures = torch.tensor([0.5, 0.8, 1.0, 1.3, 2.0, 3.0, 0.9])
        self.rows = tile_sts_temperatures(self.temperatures, self.bs)

    def _reference(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits / rows.unsqueeze(-1), dim=-1).amax(dim=-1)

    def test_identity_temperatures_leave_the_confidence_untouched(self):
        # Guards the off switch: a table of ones must not perturb a single bit,
        # or loading one would quietly move every budget decision.
        ones = tile_sts_temperatures(torch.ones(self.gamma), self.bs)
        plain = _shifted_exp_partition(self.logits.clone()).reciprocal_()
        calibrated = _shifted_exp_partition(
            self.logits.clone(), temperatures=ones
        ).reciprocal_()
        self.assertTrue(torch.equal(plain, calibrated))

    def test_temperature_matches_an_explicit_temperature_softmax(self):
        got = _shifted_exp_partition(
            self.logits.clone(), temperatures=self.rows
        ).reciprocal_()
        torch.testing.assert_close(got, self._reference(self.rows))

    def test_each_position_carries_its_own_temperature(self):
        # Guards the tiling: the logit rows are request-major, position-minor,
        # so a transposed tile would hand position j the wrong temperature and
        # still produce plausible numbers.
        got = _shifted_exp_partition(
            self.logits.clone(), temperatures=self.rows
        ).reciprocal_()
        per_row = torch.stack(
            [
                torch.softmax(self.logits[i] / self.temperatures[i % self.gamma], -1)
                .amax()
                for i in range(self.bs * self.gamma)
            ]
        )
        torch.testing.assert_close(got, per_row)

    def test_shard_merge_carries_the_temperature(self):
        # Guards the merge: with tp=1 the re-centering is a no-op, so dropping
        # the temperature there passes every single-rank test and only skews
        # the confidence once the vocabulary is actually sharded.
        tp = 4
        shards = self.logits.chunk(tp, dim=-1)
        partitions = torch.stack(
            [_shifted_exp_partition(s.clone(), temperatures=self.rows) for s in shards]
        )
        maxima = torch.stack([s.amax(dim=-1) for s in shards])
        merged = _merge_shard_partitions(
            partitions=partitions, maxima=maxima, temperatures=self.rows
        )
        torch.testing.assert_close(merged, self._reference(self.rows))


if __name__ == "__main__":
    unittest.main()
