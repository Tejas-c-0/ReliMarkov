import unittest
from types import SimpleNamespace

import pandas as pd

from kr_experiments_phase4 import (
    _cluster_bootstrap,
    assert_no_validation_cutoff_events,
    build_common_candidate_set,
    _record_metrics,
    rank_candidates,
    topk_rank_diagnostics,
)


class Phase4AdditionalDeterminismTests(unittest.TestCase):
    def test_rank_tie_break_is_deterministic(self):
        candidates = {4, 2, 9, 5}
        popularity = {2: 10, 4: 10, 5: 2, 9: 3}
        ranked = rank_candidates(candidates, lambda _: 1.0, popularity)
        self.assertEqual(ranked, [2, 4, 9, 5])

    def test_empty_candidate_set_is_allowed(self):
        self.assertEqual(build_common_candidate_set(set(), set(), set()), frozenset())

    def test_unknown_source_uses_global_top_100_union(self):
        self.assertEqual(
            build_common_candidate_set(set(), {7, 8, 9}, {1, 7, 8, 9}),
            frozenset({7, 8, 9}),
        )

    def test_out_of_catalogue_target_raises(self):
        engine = SimpleNamespace(
            catalogue={1, 2, 3},
            video_to_author={},
            author_catalogue=set(),
            author_ranking=lambda video_ranking: [],
        )
        row = SimpleNamespace(tgt_video_id=99, user_id=1)
        record = _record_metrics(
            engine,
            row,
            frozenset({1, 2, 3}),
            "synthetic-query",
            [1, 2, 3],
        )
        self.assertFalse(record["target_video_in_catalogue"])
        self.assertFalse(record["target_video_in_candidates"])
        self.assertEqual(record["video_metrics"]["hit@20"], 0.0)
        with self.assertRaises(ValueError):
            build_common_candidate_set({99}, set(), {1, 2, 3})
        with self.assertRaises(ValueError):
            build_common_candidate_set(set(), {99}, {1, 2, 3})

    def test_rank_displacement_and_position_agreement_are_reported(self):
        left = [10, 20, 30, 40]
        right = [20, 10, 40, 30]
        diagnostics = topk_rank_diagnostics(left, right, 4)
        self.assertAlmostEqual(diagnostics["position_agreement"], 0.0)
        self.assertAlmostEqual(diagnostics["mean_absolute_rank_displacement"], 1.0)
        self.assertAlmostEqual(diagnostics["overlap"], 1.0)
        self.assertAlmostEqual(diagnostics["jaccard"], 1.0)

    def test_repeat_run_reproducibility_on_synthetic_ranking(self):
        candidates = {7, 9, 11, 13}
        popularity = {7: 3, 9: 8, 11: 2, 13: 7}
        first = rank_candidates(candidates, lambda vid: (vid % 5), popularity)
        second = rank_candidates(candidates, lambda vid: (vid % 5), popularity)
        third = rank_candidates(candidates, lambda vid: (vid % 5), popularity)
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_user_cluster_bootstrap_reuses_resamples_for_aligned_records(self):
        baseline = [
            {"user_id": 1, "target_author": 10, "video_metrics": {"score": 0.0}},
            {"user_id": 1, "target_author": 10, "video_metrics": {"score": 0.4}},
            {"user_id": 2, "target_author": 20, "video_metrics": {"score": 0.2}},
            {"user_id": 3, "target_author": 30, "video_metrics": {"score": 0.8}},
        ]
        shifted = [
            {
                **record,
                "video_metrics": {"score": record["video_metrics"]["score"] + 0.1},
            }
            for record in baseline
        ]
        first = _cluster_bootstrap(
            baseline, ["score"], "next_video", rounds=100, seed=7
        )["score"]
        second = _cluster_bootstrap(
            shifted, ["score"], "next_video", rounds=100, seed=7
        )["score"]
        self.assertAlmostEqual(second[0] - first[0], 0.1)
        self.assertAlmostEqual(second[1] - first[1], 0.1)

    def test_guard_rejects_any_loaded_event_at_or_after_validation_end(self):
        good = pd.DataFrame({"time_ms": [1651593599999, 1651593599998]})
        bad = pd.DataFrame({"time_ms": [1651593599999, 1651593600000]})
        assert_no_validation_cutoff_events(good)
        with self.assertRaises(AssertionError):
            assert_no_validation_cutoff_events(bad)


if __name__ == "__main__":
    unittest.main()
