import unittest

from kr_experiments_phase4 import (
    _skip_line_numbers,
    aggregate_population,
    assert_candidate_set_equality,
    assert_query_alignment,
    build_common_candidate_set,
)


class Phase4CandidateTests(unittest.TestCase):
    def test_candidate_set_is_shared_training_neighbors_union_global_top_100(self):
        candidates = build_common_candidate_set(
            source_neighbors={1, 2, 3},
            global_top_100={3, 4, 5},
            training_catalogue={1, 2, 3, 4, 5},
        )
        self.assertEqual(candidates, frozenset({1, 2, 3, 4, 5}))

    def test_candidate_set_equality_checks_all_models(self):
        shared = frozenset({1, 2, 3})
        candidate_sets = {model: shared for model in ("M1", "M4_W1", "M5B")}
        self.assertEqual(
            assert_candidate_set_equality(
                candidate_sets,
                expected_models=("M1", "M4_W1", "M5B"),
            ),
            3,
        )
        candidate_sets["M5B"] = frozenset({1, 2, 4})
        with self.assertRaises(AssertionError):
            assert_candidate_set_equality(
                candidate_sets,
                expected_models=("M1", "M4_W1", "M5B"),
            )

    def test_row_filter_skips_unselected_data_lines_but_not_header(self):
        self.assertEqual(_skip_line_numbers(bytearray([1, 0, 1])).tolist(), [2])

    def test_query_alignment_rejects_reordered_or_missing_queries(self):
        assert_query_alignment(["q1", "q2"], ["q1", "q2"])
        with self.assertRaises(AssertionError):
            assert_query_alignment(["q1", "q2"], ["q2", "q1"])
        with self.assertRaises(AssertionError):
            assert_query_alignment(["q1", "q2"], ["q1"])


class Phase4EvaluationTests(unittest.TestCase):
    @staticmethod
    def _record(recommendations, hit20, target_author=10, author_recommendations=None):
        return {
            "video_recommendations": recommendations,
            "video_metrics": {
                "hit@10": 0.0,
                "ndcg@10": 0.0,
                "mrr@10": 0.0,
                "hit@20": hit20,
                "ndcg@20": hit20,
                "mrr@20": hit20,
            },
            "target_author": target_author,
            "author_recommendations": author_recommendations or [],
            "author_metrics": {
                "hit@10": 0.0,
                "ndcg@10": 0.0,
                "mrr@10": 0.0,
                "hit@20": hit20,
                "ndcg@20": hit20,
                "mrr@20": hit20,
            },
        }

    def test_subset_coverage_uses_only_selected_records(self):
        selected = [self._record([1, 2], 1.0)]
        aggregate = aggregate_population(
            selected,
            "next_video",
            video_catalogue_size=10,
            author_catalogue_size=5,
        )
        self.assertEqual(aggregate["n_population_queries"], 1)
        self.assertEqual(aggregate["n_cases"], 1)
        self.assertEqual(aggregate["unique_recommendations"], 2)
        self.assertAlmostEqual(aggregate["recommendation_coverage"], 0.2)

    def test_metric_denominator_and_hit_count_are_explicit(self):
        records = [
            self._record([1, 2], 1.0),
            self._record([3, 4], 0.0),
        ]
        aggregate = aggregate_population(
            records,
            "next_video",
            video_catalogue_size=10,
            author_catalogue_size=5,
        )
        self.assertEqual(aggregate["n_population_queries"], 2)
        self.assertEqual(aggregate["n_cases"], 2)
        self.assertEqual(aggregate["hit_count@20"], 1)
        self.assertAlmostEqual(aggregate["hit@20"], 0.5)

    def test_next_author_denominator_excludes_unknown_target_authors(self):
        records = [
            self._record([1, 2], 1.0, target_author=10, author_recommendations=[10]),
            self._record([3, 4], 0.0, target_author=None, author_recommendations=[]),
        ]
        aggregate = aggregate_population(
            records,
            "next_author",
            video_catalogue_size=10,
            author_catalogue_size=5,
        )
        self.assertEqual(aggregate["n_population_queries"], 2)
        self.assertEqual(aggregate["n_cases"], 1)
        self.assertEqual(aggregate["n_unscorable"], 1)
        self.assertEqual(aggregate["hit_count@20"], 1)


if __name__ == "__main__":
    unittest.main()
