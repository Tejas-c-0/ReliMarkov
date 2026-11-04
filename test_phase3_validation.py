import unittest

import numpy as np

from kr_experiments_phase3 import (
    event_reliability,
    count_missing_metadata_durations,
    duration_treatment_supported,
    estimate_duration_medians,
    score_m4,
    score_m5a,
    score_m5b,
    score_m5c,
    topk_overlap,
)


class Phase3ScoringTests(unittest.TestCase):
    def test_m4_w1_does_not_depend_on_query_reliability(self):
        score_low = score_m4(4, 0.0, 0.2, 0.01, "W1", alpha=1.0)
        score_high = score_m4(4, 0.0, 1.4, 0.01, "W1", alpha=1.0)
        self.assertEqual(score_low, score_high)
        self.assertAlmostEqual(score_low, 4.01)

    def test_m4_weighted_schemes_use_query_reliability(self):
        score_low = score_m4(4, 2.0, 0.5, 0.01, "W4", alpha=1.0)
        score_high = score_m4(4, 2.0, 1.0, 0.01, "W4", alpha=1.0)
        self.assertAlmostEqual(score_low, 1.01)
        self.assertAlmostEqual(score_high, 2.01)

    def test_m5_b_normalizes_historical_weight_only(self):
        score = score_m5b(
            weighted_count=8.0,
            query_weight=0.4,
            mean_w4=0.5,
            alpha_eb=3.0,
            prior_mass=0.02,
        )
        self.assertAlmostEqual(score, 0.4 * (8.0 / 0.5) + 3.0 * 0.02)

    def test_m5_c_is_rank_scale_equivalent_to_corrected_m5_b(self):
        values = [(8.0, 0.4, 0.02), (1.0, 1.2, 0.4), (0.0, 0.8, 0.58)]
        mean_w4 = 0.5
        alpha_eb = 3.0
        b_scores = [
            score_m5b(weighted, query, mean_w4, alpha_eb, prior)
            for weighted, query, prior in values
        ]
        c_scores = [
            score_m5c(weighted, query, mean_w4, alpha_eb, prior)
            for weighted, query, prior in values
        ]
        self.assertEqual(np.argsort(b_scores).tolist(), np.argsort(c_scores).tolist())

    def test_m5a_keeps_raw_historical_scale(self):
        self.assertAlmostEqual(score_m5a(4.0, 0.5, 3.0, 0.02), 2.06)


class Phase3ReliabilityTests(unittest.TestCase):
    def test_zero_duration_treatments(self):
        event = {
            "duration_ms": 0,
            "play_time_ms": 0,
            "is_like": 0,
            "is_forward": 0,
            "is_hate": 0,
        }
        duration_medians = [0.2, 0.4, 0.6, 0.8]
        self.assertAlmostEqual(event_reliability(event, "D0", duration_medians), 0.1)
        self.assertAlmostEqual(event_reliability(event, "D1", duration_medians), 0.9016)
        self.assertAlmostEqual(
            event_reliability(event, "D2", duration_medians, metadata_duration_ms=10000),
            0.4,
        )
        self.assertAlmostEqual(
            event_reliability(
                event,
                "D2",
                duration_medians,
                metadata_duration_ms=None,
                global_duration_median=0.55,
            ),
            0.55,
        )

    def test_duration_medians_use_training_bins_with_global_empty_bin_fallback(self):
        global_median, bin_medians = estimate_duration_medians(
            [
                np.array([0.2, 0.4]),
                np.array([0.6]),
                np.array([]),
                np.array([0.8]),
            ]
        )
        self.assertAlmostEqual(global_median, 0.5)
        np.testing.assert_allclose(bin_medians, [0.3, 0.6, 0.5, 0.8])

    def test_topk_overlap_reports_overlap_and_jaccard(self):
        result = topk_overlap([1, 2, 3], [2, 3, 4], 3)
        self.assertAlmostEqual(result["overlap"], 2 / 3)
        self.assertAlmostEqual(result["jaccard"], 0.5)

    def test_metadata_lookup_counts_distinct_zero_duration_video_ids(self):
        zero_keys = {(1, 10, 100), (2, 10, 200), (3, 11, 300)}
        self.assertEqual(count_missing_metadata_durations(zero_keys, {10: 5000}), 1)
        self.assertFalse(duration_treatment_supported(zero_keys, {10: 5000}))
        self.assertTrue(duration_treatment_supported({(1, 10, 100)}, {10: 5000}))


if __name__ == "__main__":
    unittest.main()
