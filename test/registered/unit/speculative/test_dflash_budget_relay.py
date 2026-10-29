"""The DFLASH verify budget comes off the confidence relay -- CPU-only.

The budget used to be computed inside the step from a blocking D2H on the
draft confidence. It now rides the same relay DSpark uses: the scheduler's
pre-forward hook resolves it from a lag-2 pinned ring, so the step reads an
int that is already on the host.

Two ways that goes silently wrong, both pinned here: the relay never gets
enabled for DFLASH (and the sync comes back), or the step reads the
precomputed field although nobody installed the hook to fill it -- which
reads None and degenerates to verify-all while the log still looks healthy.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import decide_needs_confidence_relay
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyPlanner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ExplodingIndices:
    """Stands in for the device-side req_pool_indices: any copy off it fails."""

    def to(self, *args, **kwargs):
        raise AssertionError("req_pool_indices was copied to the host")


def _planner(*, hook_installed: bool, budget_planner=None):
    planner = object.__new__(DSparkVerifyPlanner)
    planner._budget_hook_installed = hook_installed
    planner._budget_planner = budget_planner
    planner.model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_generation=torch.zeros(16, dtype=torch.int64)
        )
    )
    return planner


class _RecordingBudgetPlanner:
    def __init__(self, budget):
        self.budget = budget
        self.calls = []

    def compute_budget(self, **kwargs):
        self.calls.append(kwargs)
        return self.budget

    def note_non_decode_step(self):
        self.calls.append("non-decode")


class TestConfidenceRelayGate(unittest.TestCase):
    def _needs_relay(self, algorithm, mode):
        override = get_context().override_server_args(
            speculative_algorithm=algorithm
        )
        override.install()
        self.addCleanup(override.restore)
        with envs.SGLANG_RAGGED_VERIFY_MODE.override(mode):
            return decide_needs_confidence_relay()

    def test_dflash_gets_the_relay(self):
        # Narrowing this predicate back to DSpark puts the per-step D2H on the
        # draft confidence back on DFLASH's critical path.
        self.assertTrue(self._needs_relay("DFLASH", "compact"))

    def test_static_mode_needs_no_relay(self):
        # Nothing schedules a budget in static mode, so the ring is dead weight.
        self.assertFalse(self._needs_relay("DFLASH", "static"))

    def test_other_algorithms_are_untouched(self):
        self.assertFalse(self._needs_relay("EAGLE", "compact"))


class TestBudgetSourceGate(unittest.TestCase):
    def test_precomputed_budget_is_read_only_with_the_hook(self):
        budget_planner = _RecordingBudgetPlanner(budget=11)
        planner = _planner(hook_installed=True, budget_planner=budget_planner)
        draft_input = SimpleNamespace(verify_token_budget=23)

        budget = planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=torch.full((2, 7), 0.5),
            prefix_lens=torch.zeros(2, dtype=torch.int64),
            req_pool_indices=_ExplodingIndices(),
            req_pool_indices_cpu=torch.tensor([0, 1]),
        )

        self.assertEqual(budget, 23)
        self.assertEqual(budget_planner.calls, [])

    def test_without_the_hook_the_budget_is_computed_in_step(self):
        # The failure this guards: reading the field nobody filled returns
        # None, which is verify-all -- an adaptive run that never adapts.
        budget_planner = _RecordingBudgetPlanner(budget=11)
        planner = _planner(hook_installed=False, budget_planner=budget_planner)
        draft_input = SimpleNamespace(verify_token_budget=None)

        budget = planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=torch.full((2, 7), 0.5),
            prefix_lens=torch.zeros(2, dtype=torch.int64),
            req_pool_indices=_ExplodingIndices(),
            req_pool_indices_cpu=torch.tensor([0, 1]),
        )

        self.assertEqual(budget, 11)
        self.assertEqual(draft_input.verify_token_budget, 11)
        self.assertEqual(len(budget_planner.calls), 1)


class TestSyncBudgetUsesHostMirror(unittest.TestCase):
    def test_batch_mirror_replaces_the_second_d2h(self):
        budget_planner = _RecordingBudgetPlanner(budget=7)
        planner = _planner(hook_installed=False, budget_planner=budget_planner)

        budget = planner.compute_budget_sync(
            confidence=torch.full((3, 7), 0.25),
            prefix_lens=torch.zeros(3, dtype=torch.int64),
            req_pool_indices=_ExplodingIndices(),
            req_pool_indices_cpu=torch.tensor([2, 5, 9], dtype=torch.int32),
        )

        self.assertEqual(budget, 7)
        (call,) = budget_planner.calls
        # int64 regardless of what the batch carries: it indexes req_generation.
        self.assertEqual(call["req_pool_indices_cpu"].dtype, torch.int64)
        self.assertEqual(call["req_pool_indices_cpu"].tolist(), [2, 5, 9])

    def test_absent_mirror_still_falls_back_to_the_device_copy(self):
        budget_planner = _RecordingBudgetPlanner(budget=7)
        planner = _planner(hook_installed=False, budget_planner=budget_planner)

        planner.compute_budget_sync(
            confidence=torch.full((3, 7), 0.25),
            prefix_lens=torch.zeros(3, dtype=torch.int64),
            req_pool_indices=torch.tensor([2, 5, 9], dtype=torch.int32),
        )

        (call,) = budget_planner.calls
        self.assertEqual(call["req_pool_indices_cpu"].tolist(), [2, 5, 9])


class TestWorkersExposeTheHook(unittest.TestCase):
    def test_every_relayed_algorithm_offers_the_prepare_hook(self):
        # The scheduler calls this on the draft worker whenever the relay is
        # on; an algorithm admitted above without the method dies at startup.
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        for worker in (DFlashWorkerV2, DSparkWorkerV2):
            self.assertTrue(
                callable(worker.get_confidence_budget_prepare),
                f"{worker.__name__} has no confidence budget prepare hook",
            )


if __name__ == "__main__":
    unittest.main()
