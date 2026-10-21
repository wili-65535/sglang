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

from sglang.srt.speculative.dflash_sts import (
    DflashStsCollector,
    DflashSyntheticDraft,
    DflashStsDataRecorder,
    collect_dflash_partition_sums,
    dflash_sts_temperature_grid,
    merge_shard_partition_sums,
    partition_sums_over_grid,
)
from sglang.srt.speculative.dflash_worker_v2 import (
    _DflashDraftSampler,
    _merge_shard_partitions,
    _shifted_exp_partition,
    draft_confidence_from_head,
    tile_sts_temperatures,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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
            sts_temperatures=None,
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
                sts_temperatures=None,
            )
        )

    def test_refuses_a_head_without_a_dense_weight(self):
        self.assertIsNone(
            draft_confidence_from_head(
                hidden_states=torch.randn(7, 8),
                lm_head=SimpleNamespace(weight=None, shard_indices=None),
                bs=1,
                gamma=7,
                sts_temperatures=None,
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


class TestDflashSyntheticDraft(unittest.TestCase):
    """The stand-in draft exists so STS collection has something to calibrate
    before the draft has trained weights. Its whole claim is that the
    confidence it publishes is the probability its acceptance actually
    realizes, which makes T=1 the answer a fit must return."""

    def _collect(self, *, draft, steps, bs, gamma):
        partitions, accept_lens = [], []
        for _ in range(steps):
            sums, _confidence, lens = draft.step(bs)
            partitions.append(sums)
            accept_lens.append(lens)
        accept = torch.cat(accept_lens)
        mask = (torch.arange(gamma).view(1, -1) < accept.view(-1, 1)).float()
        return torch.cat(partitions), mask

    def test_the_acceptance_realizes_the_published_confidence(self):
        # If the draw and the confidence drift apart -- a different grid
        # column, a flipped comparison -- the collected data is calibrated to
        # a temperature nobody can name, and nothing else here would notice.
        gamma, bs = 4, 8
        draft = DflashSyntheticDraft(
            gamma=gamma, target_accept=0.7, device="cpu", seed=3
        )
        confidences, hits = [], []
        # 4800 draws: the gap is a binomial mean, so too few samples fail on a
        # fair draw rather than on a defect.
        for _ in range(600):
            _sums, confidence, accept_lens = draft.step(bs)
            confidences.append(confidence[:, 0])
            hits.append((accept_lens > 0).float())
        self.assertAlmostEqual(
            float(torch.cat(confidences).mean()),
            float(torch.cat(hits).mean()),
            delta=0.03,
        )

    def test_a_fit_over_its_output_returns_the_identity(self):
        gamma, bs = 4, 8
        draft = DflashSyntheticDraft(
            gamma=gamma, target_accept=0.85, device="cpu", seed=4
        )
        partitions, mask = self._collect(draft=draft, steps=250, bs=bs, gamma=gamma)
        from sglang.benchmark.dflash_sts_fit import fit_dflash_sts_temperatures

        grid = dflash_sts_temperature_grid()
        result = fit_dflash_sts_temperatures(
            partitions=partitions, prefix_mask=mask, grid=grid
        )
        # One grid step is a factor of 1.122; neighbouring temperatures are not
        # separable by a binned calibration error on a finite sample.
        for position, temperature in enumerate(result["temperatures"]):
            with self.subTest(position=position):
                self.assertLess(abs(torch.tensor(temperature).log()), 0.13)


class TestDflashStsCollect(unittest.TestCase):
    """The collect side records enough to re-derive the confidence at any
    temperature on the grid, which a single number per draft token cannot do:
    DFLASH's confidence is a softmax over the whole row, not a lone logit."""

    def _head(self, weight, shard=None):
        return SimpleNamespace(weight=weight, shard_indices=shard)

    def test_the_identity_column_is_the_uncalibrated_runtime_confidence(self):
        # The fit's "before" baseline is read off this column, so a collector
        # that drifted from the runtime's definition would silently calibrate
        # against a confidence nobody reports.
        torch.manual_seed(11)
        vocab, hidden, bs, gamma = 96, 8, 3, 7
        weight = torch.randn(vocab, hidden) * 0.3
        hidden_states = torch.randn(bs * gamma, hidden)
        grid = dflash_sts_temperature_grid()
        identity = int((grid - 1.0).abs().argmin())

        partitions = collect_dflash_partition_sums(
            hidden_states=hidden_states,
            lm_head=self._head(weight),
            grid=grid,
        )
        runtime = draft_confidence_from_head(
            hidden_states=hidden_states,
            lm_head=self._head(weight),
            bs=bs,
            gamma=gamma,
            sts_temperatures=None,
        )
        torch.testing.assert_close(
            partitions[:, identity].reciprocal().view(bs, gamma), runtime
        )

    def test_shard_merge_recenters_at_each_grid_temperature(self):
        # With tp=1 the re-centering is a no-op, so a merge that recentered at
        # one temperature for the whole grid passes every single-rank test and
        # skews every column but T=1 once the vocabulary is sharded.
        torch.manual_seed(12)
        logits = torch.randn(6, 256) * 4.0
        grid = dflash_sts_temperature_grid()
        want, _ = partition_sums_over_grid(logits=logits, grid=grid)

        shards = logits.chunk(4, dim=-1)
        per_shard = [partition_sums_over_grid(logits=s, grid=grid) for s in shards]
        got = merge_shard_partition_sums(
            partitions=torch.stack([p for p, _ in per_shard]),
            maxima=torch.stack([m for _, m in per_shard]),
            grid=grid,
        )
        torch.testing.assert_close(got, want)

    def test_recorder_writes_the_grid_and_the_correct_prefix_mask(self):
        # The grid has to travel with the data: a shard read against a
        # different grid maps every fitted temperature to the wrong column.
        import tempfile

        grid = torch.tensor([0.5, 1.0, 2.0])
        with tempfile.TemporaryDirectory() as tmp:
            stem = f"{tmp}/shard"
            recorder = DflashStsDataRecorder(path_stem=stem, gamma=3, flush_every=1)
            recorder.record(
                partitions=torch.ones(2, 3, 3),
                num_correct_drafts=torch.tensor([0, 2]),
                grid=grid,
            )
            shard = torch.load(f"{stem}.0.pt", map_location="cpu")

        torch.testing.assert_close(shard["temperature_grid"], grid)
        torch.testing.assert_close(
            shard["prefix_mask"], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        )

    def test_fit_picks_the_grid_point_whose_survival_is_calibrated(self):
        from sglang.benchmark.dflash_sts_fit import fit_dflash_sts_temperatures

        # Half the samples accept and half do not, so the calibrated column is
        # the one reading exactly 0.5; every other column is off by |p - 0.5|.
        grid = torch.tensor([0.5, 0.8, 1.0, 1.5, 2.0])
        probs = torch.tensor([0.2, 0.35, 0.8, 0.5, 0.95])
        num_samples = 64
        partitions = probs.reciprocal().view(1, 1, -1).expand(num_samples, 1, 5)
        prefix_mask = torch.zeros(num_samples, 1)
        prefix_mask[::2] = 1.0

        result = fit_dflash_sts_temperatures(
            partitions=partitions.contiguous(),
            prefix_mask=prefix_mask,
            grid=grid,
            num_bins=15,
        )
        self.assertEqual(result["temperatures"], [1.5])
        self.assertAlmostEqual(result["ece_after"][0], 0.0, places=9)
        # The T=1 column reads 0.8 against a 0.5 outcome rate.
        self.assertAlmostEqual(result["ece_before"][0], 0.3, places=9)

    def test_fit_leaves_an_undecidable_position_uncalibrated(self):
        from sglang.benchmark.dflash_sts_fit import fit_dflash_sts_temperatures

        # A draft that is never right collapses the survival, and from there
        # every grid point scores a perfect ECE. Breaking that tie toward the
        # end of the grid would publish a wild temperature fit to nothing.
        grid = torch.tensor([0.1, 0.5, 1.0, 2.0, 10.0])
        num_samples = 32
        partitions = torch.full((num_samples, 2, 5), 1e6)
        prefix_mask = torch.zeros(num_samples, 2)

        result = fit_dflash_sts_temperatures(
            partitions=partitions,
            prefix_mask=prefix_mask,
            grid=grid,
            num_bins=15,
        )
        self.assertEqual(result["temperatures"], [1.0, 1.0])

    def test_only_the_writer_rank_lays_down_shards(self):
        # Every rank has to derive the sums, because the shard merge is a
        # collective; they then all hold the same values, so a second writer
        # only races the first for the same filenames.
        import glob
        import tempfile

        torch.manual_seed(13)
        vocab, hidden, bs, block = 64, 8, 2, 8
        weight = torch.randn(vocab, hidden) * 0.3
        head = self._head(weight)
        draft_hidden = torch.randn(bs, block, hidden)
        candidates = torch.randint(0, vocab, (bs, block))

        written = {}
        for is_writer in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                collector = DflashStsCollector(
                    path_stem=f"{tmp}/sts",
                    gamma=block - 1,
                    flush_every=1,
                    is_writer=is_writer,
                )
                collector.observe_draft(
                    draft_hidden=draft_hidden,
                    bs=bs,
                    lm_head=head,
                    tp_group=None,
                    runtime_confidence=None,
                )
                collector.observe_verify(
                    candidates=candidates,
                    target_predict=candidates.clone(),
                    verify_lens=None,
                )
                written[is_writer] = len(glob.glob(f"{tmp}/sts.*.pt"))

        self.assertEqual(written[True], 1)
        self.assertEqual(written[False], 0)

    def test_the_closed_loop_recovers_a_planted_temperature(self):
        # Walks the whole chain the way a run does -- partition sums in,
        # several shards out, loader, fit -- against data whose calibrated
        # temperature is known. A column swapped against the grid, a shard
        # concatenated on the wrong axis or a survival that forgot to
        # accumulate all land here and nowhere else.
        import tempfile

        from sglang.benchmark.dflash_sts_fit import (
            fit_dflash_sts_temperatures,
            load_collected_shards,
            synthesize_dflash_sts_data,
        )

        planted = [0.5, 0.8, 1.26, 2.0]
        partitions, prefix_mask, grid = synthesize_dflash_sts_data(
            num_samples=2000, planted_temperatures=planted, seed=0
        )
        nearest = [float(grid[int((grid - t).abs().argmin())]) for t in planted]

        with tempfile.TemporaryDirectory() as tmp:
            stem = f"{tmp}/sts"
            recorder = DflashStsDataRecorder(
                path_stem=stem, gamma=len(planted), flush_every=4
            )
            # Several steps per shard and several shards, so a loader that
            # mixed the order would show up as a worse fit.
            for start in range(0, 2000, 200):
                recorder.record(
                    partitions=partitions[start : start + 200],
                    num_correct_drafts=prefix_mask[start : start + 200].sum(dim=1),
                    grid=grid,
                )
            recorder.flush()
            loaded, loaded_mask, loaded_grid = load_collected_shards(
                data_glob=f"{stem}.*.pt"
            )

        self.assertEqual(loaded.shape, partitions.shape)
        torch.testing.assert_close(loaded_grid, grid)
        result = fit_dflash_sts_temperatures(
            partitions=loaded, prefix_mask=loaded_mask, grid=loaded_grid
        )
        self.assertEqual(result["temperatures"], nearest)
        for position in range(len(planted)):
            self.assertLess(
                result["ece_after"][position], result["ece_before"][position]
            )

    def test_fit_refuses_shards_collected_on_different_grids(self):
        from sglang.benchmark.dflash_sts_fit import load_collected_shards

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for index, grid in enumerate(
                (torch.tensor([0.5, 1.0]), torch.tensor([0.5, 2.0]))
            ):
                torch.save(
                    {
                        "partitions": torch.ones(1, 1, 2),
                        "prefix_mask": torch.ones(1, 1),
                        "temperature_grid": grid,
                    },
                    f"{tmp}/shard.{index}.pt",
                )
            with self.assertRaises(ValueError):
                load_collected_shards(data_glob=f"{tmp}/shard.*.pt")


if __name__ == "__main__":
    unittest.main()
