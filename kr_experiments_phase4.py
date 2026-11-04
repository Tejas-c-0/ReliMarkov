"""Controlled-candidate, validation-only Phase 4 recommendation experiment."""

import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from kr_experiments_phase3 import (
    event_reliability,
    score_m4,
    score_m5a,
    score_m5b,
    score_m5c,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
EB_DIR = ROOT / "outputs" / "eb_params"
RESULTS_ROOT = ROOT / "outputs" / "phase4_validation"
LOG_FILES = (
    DATA_DIR / "log_standard_4_08_to_4_21_1k.csv",
    DATA_DIR / "log_standard_4_22_to_5_08_1k.csv",
)
GAPS = (30, 60)
TREATMENTS = ("D0", "D1")
TOP_KS = (10, 20)
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

TRAIN_START_MS = 1649347200000
TRAIN_END_MS = 1651161599999
VAL_END_MS = 1651593599999
CHUNK_SIZE = 200_000
TIME_CHUNK_SIZE = 500_000

MODEL_NAMES = (
    "M1",
    "M4_W1",
    "M4_W2",
    "M4_W3",
    "M4_W4",
    "M5A",
    "M5B",
    "M5C",
)
EVENT_COLUMNS = [
    "user_id",
    "video_id",
    "time_ms",
    "play_time_ms",
    "duration_ms",
    "is_like",
    "is_forward",
    "is_hate",
    "tab",
    "is_click",
]
EVENT_KEY_COLUMNS = ["user_id", "video_id", "time_ms"]


def event_key(user_id, video_id, time_ms):
    return int(user_id), int(video_id), int(time_ms)


def query_id(gap, row_number, row):
    return (
        f"{gap}min:{row_number}:{int(row.user_id)}:{int(row.src_video_id)}:"
        f"{int(row.src_time_ms)}:{int(row.tgt_video_id)}:{int(row.tgt_time_ms)}"
    )


def scan_training_validation_rows(log_path):
    """Build a row mask from timestamps only; never parse later event outcomes."""
    row_mask = bytearray()
    for frame in pd.read_csv(
        log_path,
        usecols=["time_ms"],
        chunksize=TIME_CHUNK_SIZE,
        skip_blank_lines=False,
    ):
        times = pd.to_numeric(frame["time_ms"], errors="coerce")
        selected = times.between(TRAIN_START_MS, VAL_END_MS, inclusive="both")
        row_mask.extend(selected.to_numpy(dtype=np.uint8).tobytes())
    return row_mask, sum(row_mask)


def _skip_line_numbers(row_mask):
    selected_rows = np.frombuffer(row_mask, dtype=np.uint8)
    return np.flatnonzero(selected_rows == 0) + 1


def _read_filtered_event_chunks(log_path, row_mask):
    return pd.read_csv(
        log_path,
        usecols=EVENT_COLUMNS,
        skiprows=_skip_line_numbers(row_mask),
        chunksize=CHUNK_SIZE,
        skip_blank_lines=False,
    )


def build_common_candidate_set(source_neighbors, global_top_100, training_catalogue):
    candidates = frozenset(source_neighbors).union(global_top_100)
    if not candidates.issubset(training_catalogue):
        raise ValueError("A common candidate is outside the training catalogue.")
    return candidates


def assert_candidate_set_equality(candidate_sets, expected_models=MODEL_NAMES):
    if set(candidate_sets) != set(expected_models):
        raise AssertionError("Candidate-set model keys differ from the expected model list.")
    reference = candidate_sets[expected_models[0]]
    for model in expected_models[1:]:
        if candidate_sets[model] != reference:
            raise AssertionError(f"Candidate set differs for {model}.")
    return len(reference)


def assert_query_alignment(left_query_ids, right_query_ids):
    if left_query_ids != right_query_ids:
        raise AssertionError("Prediction query IDs or their order differ.")


def assert_no_validation_cutoff_events(frame):
    if "time_ms" not in frame:
        raise KeyError("Loaded event frame is missing time_ms.")
    time_series = pd.to_numeric(frame["time_ms"], errors="coerce")
    post_cutoff = time_series[time_series >= (VAL_END_MS + 1)]
    if not post_cutoff.empty:
        raise AssertionError(
            "Loaded event rows fall on or after the validation cutoff: "
            f"{len(post_cutoff)} rows >= {VAL_END_MS + 1}."
        )


def rank_candidates(candidates, score_function, video_popularity):
    scores = {video_id: score_function(video_id) for video_id in candidates}
    return sorted(
        candidates,
        key=lambda video_id: (
            -scores[video_id],
            -video_popularity.get(video_id, 0),
            video_id,
        ),
    )


def topk_rank_diagnostics(left, right, k):
    if len(left) < k or len(right) < k:
        raise ValueError(f"Both rankings must contain at least {k} items.")
    left_top = left[:k]
    right_top = right[:k]
    left_set = set(left_top)
    right_set = set(right_top)
    common = left_set & right_set
    left_positions = {video_id: rank for rank, video_id in enumerate(left_top, start=1)}
    right_positions = {video_id: rank for rank, video_id in enumerate(right_top, start=1)}
    return {
        "overlap": len(common) / k,
        "jaccard": len(common) / len(left_set | right_set),
        "position_agreement": sum(a == b for a, b in zip(left_top, right_top)) / k,
        "mean_absolute_rank_displacement": (
            float(np.mean([abs(left_positions[v] - right_positions[v]) for v in common]))
            if common
            else float("nan")
        ),
    }


def _video_case_metrics(recommendations, target):
    rank = recommendations.index(target) + 1 if target in recommendations else None
    return _rank_metrics(rank)


def _author_case_metrics(recommendations, target_author):
    rank = recommendations.index(target_author) + 1 if target_author in recommendations else None
    return _rank_metrics(rank)


def _rank_metrics(rank):
    result = {}
    for k in TOP_KS:
        hit = float(rank is not None and rank <= k)
        result[f"hit@{k}"] = hit
        result[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        result[f"mrr@{k}"] = 1.0 / rank if hit else 0.0
    return result


def aggregate_population(records, metric_type, video_catalogue_size, author_catalogue_size):
    """Aggregate a preselected population; coverage uses only its scorable records."""
    if metric_type == "next_video":
        metric_records = records
        recommendation_key = "video_recommendations"
        catalogue_size = video_catalogue_size
        metrics_key = "video_metrics"
    elif metric_type == "next_author":
        metric_records = [record for record in records if record["target_author"] is not None]
        recommendation_key = "author_recommendations"
        catalogue_size = author_catalogue_size
        metrics_key = "author_metrics"
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    result = {
        "n_population_queries": len(records),
        "n_cases": len(metric_records),
        "n_unscorable": len(records) - len(metric_records),
    }
    for k in TOP_KS:
        for metric in (f"hit@{k}", f"ndcg@{k}", f"mrr@{k}"):
            values = [record[metrics_key][metric] for record in metric_records]
            result[metric] = float(np.mean(values)) if values else float("nan")
        result[f"hit_count@{k}"] = int(
            sum(record[metrics_key][f"hit@{k}"] for record in metric_records)
        )

    recommendations = {
        item
        for record in metric_records
        for item in record[recommendation_key]
    }
    result["unique_recommendations"] = len(recommendations)
    result["recommendation_catalogue_size"] = catalogue_size
    result["recommendation_coverage"] = (
        len(recommendations) / catalogue_size if catalogue_size else 0.0
    )
    return result


def _cluster_bootstrap(records, metrics, metric_type, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    metric_key = "video_metrics" if metric_type == "next_video" else "author_metrics"
    eligible = [
        record
        for record in records
        if metric_type == "next_video" or record["target_author"] is not None
    ]
    if not eligible:
        return {metric: (float("nan"), float("nan")) for metric in metrics}

    frame = pd.DataFrame(
        {
            "user_id": [record["user_id"] for record in eligible],
            **{
                metric: [record[metric_key][metric] for record in eligible]
                for metric in metrics
            },
        }
    )
    users = frame["user_id"].unique()
    user_index = pd.Index(users)
    row_user = user_index.get_indexer(frame["user_id"])
    user_counts = np.bincount(row_user, minlength=len(users)).astype(np.float64)
    sums = np.zeros((len(users), len(metrics)), dtype=np.float64)
    for metric_index, metric in enumerate(metrics):
        sums[:, metric_index] = np.bincount(
            row_user,
            weights=frame[metric].to_numpy(dtype=np.float64),
            minlength=len(users),
        )

    rng = np.random.RandomState(seed)
    bootstrap_values = np.empty((rounds, len(metrics)), dtype=np.float64)
    for round_index in range(rounds):
        sampled_users = rng.choice(len(users), size=len(users), replace=True)
        multiplicities = np.bincount(sampled_users, minlength=len(users)).astype(np.float64)
        denominator = float(np.dot(multiplicities, user_counts))
        bootstrap_values[round_index] = np.dot(multiplicities, sums) / denominator
    return {
        metric: (
            float(np.percentile(bootstrap_values[:, index], 2.5)),
            float(np.percentile(bootstrap_values[:, index], 97.5)),
        )
        for index, metric in enumerate(metrics)
    }


class Phase4ExperimentEngine:
    """Training-only tables and validation queries with no test-split reads."""

    def __init__(self, gap, data_dir=DATA_DIR):
        if gap not in GAPS:
            raise ValueError(f"Unsupported gap: {gap}")
        self.gap = gap
        self.tag = f"{gap}min"
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / "splits"
        self.eb_dir = self.data_dir.parent / "outputs" / "eb_params"
        self.log_files = (
            self.data_dir / "log_standard_4_08_to_4_21_1k.csv",
            self.data_dir / "log_standard_4_22_to_5_08_1k.csv",
        )
        self.train_df = pd.read_parquet(
            self.split_dir / f"train_transitions_{self.tag}.parquet"
        )
        self.val_df = pd.read_parquet(
            self.split_dir / f"val_transitions_{self.tag}.parquet"
        )
        self.catalogue = set(
            json.loads((self.split_dir / f"train_catalogue_{self.tag}.json").read_text())
        )
        self.video_popularity = {
            int(video_id): int(count)
            for video_id, count in json.loads(
                (self.split_dir / f"video_popularity_{self.tag}.json").read_text()
            ).items()
        }
        self.global_top_100 = tuple(
            sorted(
                self.video_popularity,
                key=lambda video_id: (-self.video_popularity[video_id], video_id),
            )[:100]
        )
        self.video_prior = {
            video_id: count / sum(self.video_popularity.values())
            for video_id, count in self.video_popularity.items()
        }
        author_video_map = json.loads(
            (self.split_dir / f"author_video_map_{self.tag}.json").read_text()
        )
        self.video_to_author = {
            int(video_id): int(author_id)
            for author_id, videos in author_video_map.items()
            for video_id in videos
        }
        self.author_catalogue = {
            self.video_to_author[video_id]
            for video_id in self.catalogue
            if video_id in self.video_to_author
        }
        eb_data = json.loads(
            (self.eb_dir / f"eb_hyperparams_{self.tag}.json").read_text()
        )
        self.alpha_eb_video = float(eb_data["video_eb"]["alpha0"])

        self.required_event_keys = self._required_event_keys()
        self.event_records, self.log_scan_diagnostics = self._load_relevant_events()
        self.tables = {}
        self.query_weights = {}
        self.mean_w4 = {}
        self.lookup_diagnostics = {}
        self._build_treatment_tables()

    def _required_event_keys(self):
        keys = set()
        for row in self.train_df.itertuples(index=False):
            keys.add(event_key(row.user_id, row.src_video_id, row.src_time_ms))
            keys.add(event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms))
        for row in self.val_df.itertuples(index=False):
            keys.add(event_key(row.user_id, row.src_video_id, row.src_time_ms))
        return keys

    def _load_relevant_events(self):
        required_index = pd.MultiIndex.from_tuples(
            sorted(self.required_event_keys),
            names=EVENT_KEY_COLUMNS,
        )
        event_records = {}
        duplicate_event_keys = 0
        file_diagnostics = []

        for log_path in self.log_files:
            row_mask, selected_row_count = scan_training_validation_rows(log_path)
            parsed_row_count = 0
            matched_row_count = 0
            for chunk in _read_filtered_event_chunks(log_path, row_mask):
                parsed_row_count += len(chunk)
                assert_no_validation_cutoff_events(chunk)
                times = pd.to_numeric(chunk["time_ms"], errors="coerce")
                dataset_a_click = chunk["tab"].eq(1) & chunk["is_click"].eq(1)
                eligible = chunk.loc[
                    times.between(TRAIN_START_MS, VAL_END_MS, inclusive="both")
                    & dataset_a_click
                ]
                if eligible.empty:
                    continue
                eligible_index = pd.MultiIndex.from_frame(eligible[EVENT_KEY_COLUMNS])
                matched = eligible.loc[eligible_index.isin(required_index)]
                matched_row_count += len(matched)
                for row in matched.itertuples(index=False):
                    key = event_key(row.user_id, row.video_id, row.time_ms)
                    if key in event_records:
                        duplicate_event_keys += 1
                        continue
                    event_records[key] = {
                        "duration_ms": row.duration_ms,
                        "play_time_ms": row.play_time_ms,
                        "is_like": row.is_like,
                        "is_forward": row.is_forward,
                        "is_hate": row.is_hate,
                    }
            if parsed_row_count != selected_row_count:
                raise RuntimeError(
                    f"Filtered row-count mismatch for {log_path.name}: "
                    f"expected {selected_row_count}, parsed {parsed_row_count}."
                )
            file_diagnostics.append(
                {
                    "file": log_path.name,
                    "timestamp_rows_selected": selected_row_count,
                    "event_rows_parsed": parsed_row_count,
                    "matched_event_rows": matched_row_count,
                    "event_columns_read_only_for_train_validation_window": True,
                }
            )
        return event_records, {
            "files": file_diagnostics,
            "duplicate_matched_event_keys": duplicate_event_keys,
            "maximum_loaded_event_time_ms": (
                max(key[2] for key in event_records) if event_records else None
            ),
        }

    def _event_weight(self, key, treatment):
        return event_reliability(
            self.event_records[key],
            treatment,
            duration_medians=(),
        )

    def _build_treatment_tables(self):
        for treatment in TREATMENTS:
            observed_training_weights = []
            for row in self.train_df.itertuples(index=False):
                for key in (
                    event_key(row.user_id, row.src_video_id, row.src_time_ms),
                    event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms),
                ):
                    if key in self.event_records:
                        observed_training_weights.append(self._event_weight(key, treatment))
            if not observed_training_weights:
                raise ValueError(f"No observed training endpoint reliabilities for {treatment}.")
            training_fallback = float(np.mean(observed_training_weights))

            video_counts = defaultdict(Counter)
            video_w2 = defaultdict(lambda: defaultdict(float))
            video_w3 = defaultdict(lambda: defaultdict(float))
            video_w4 = defaultdict(lambda: defaultdict(float))
            w4_by_transition = []
            missing_training_endpoints = 0
            for row in self.train_df.itertuples(index=False):
                source_key = event_key(row.user_id, row.src_video_id, row.src_time_ms)
                target_key = event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms)
                if source_key in self.event_records:
                    source_weight = self._event_weight(source_key, treatment)
                else:
                    source_weight = training_fallback
                    missing_training_endpoints += 1
                if target_key in self.event_records:
                    target_weight = self._event_weight(target_key, treatment)
                else:
                    target_weight = training_fallback
                    missing_training_endpoints += 1

                weighted_transition = math.sqrt(source_weight * target_weight)
                video_counts[row.src_video_id][row.tgt_video_id] += 1
                video_w2[row.src_video_id][row.tgt_video_id] += source_weight
                video_w3[row.src_video_id][row.tgt_video_id] += target_weight
                video_w4[row.src_video_id][row.tgt_video_id] += weighted_transition
                w4_by_transition.append(weighted_transition)

            validation_query_weights = {}
            missing_validation_sources = 0
            for row in self.val_df.itertuples(index=False):
                key = event_key(row.user_id, row.src_video_id, row.src_time_ms)
                if key in self.event_records:
                    validation_query_weights[key] = self._event_weight(key, treatment)
                else:
                    validation_query_weights[key] = training_fallback
                    missing_validation_sources += 1

            self.tables[treatment] = {
                "counts": video_counts,
                "W2": video_w2,
                "W3": video_w3,
                "W4": video_w4,
            }
            self.query_weights[treatment] = validation_query_weights
            self.mean_w4[treatment] = float(np.mean(w4_by_transition))
            self.lookup_diagnostics[treatment] = {
                "train_endpoint_references": 2 * len(self.train_df),
                "missing_train_endpoint_references": missing_training_endpoints,
                "validation_source_references": len(self.val_df),
                "missing_validation_source_references": missing_validation_sources,
                "missing_distinct_event_keys": len(
                    self.required_event_keys - set(self.event_records)
                ),
                "training_endpoint_fallback_mean": training_fallback,
                "training_mean_W4_per_transition": self.mean_w4[treatment],
            }

    def common_candidates(self, source):
        source_neighbors = self.tables["D0"]["counts"].get(source, Counter())
        return build_common_candidate_set(
            source_neighbors,
            self.global_top_100,
            self.catalogue,
        )

    def _score_function(self, treatment, model, source, candidate_set, query_weight):
        counts = self.tables[treatment]["counts"].get(source, Counter())
        if model == "M1":
            return lambda video_id: counts[video_id]
        if model.startswith("M4_"):
            scheme = model[3:]
            table = counts if scheme == "W1" else self.tables[treatment][scheme].get(source, {})
            return lambda video_id: score_m4(
                counts[video_id],
                table.get(video_id, 0.0) if scheme != "W1" else 0.0,
                query_weight,
                self.video_prior.get(video_id, 1e-9),
                scheme,
            )

        table = self.tables[treatment]["W4"].get(source, {})
        if model == "M5A":
            return lambda video_id: score_m5a(
                table.get(video_id, 0.0),
                query_weight,
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        if model == "M5B":
            return lambda video_id: score_m5b(
                table.get(video_id, 0.0),
                query_weight,
                self.mean_w4[treatment],
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        if model == "M5C":
            return lambda video_id: score_m5c(
                table.get(video_id, 0.0),
                query_weight,
                self.mean_w4[treatment],
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        raise ValueError(f"Unknown model: {model}")

    def author_ranking(self, video_ranking):
        authors = []
        seen = set()
        for video_id in video_ranking:
            author_id = self.video_to_author.get(video_id)
            if author_id is not None and author_id not in seen:
                seen.add(author_id)
                authors.append(author_id)
        return authors


def _record_metrics(engine, row, candidates, query_identifier, video_ranking):
    target_video = int(row.tgt_video_id)
    target_author = engine.video_to_author.get(target_video)
    candidate_authors = {
        engine.video_to_author[video_id]
        for video_id in candidates
        if video_id in engine.video_to_author
    }
    author_ranking = engine.author_ranking(video_ranking)
    return {
        "query_id": query_identifier,
        "user_id": int(row.user_id),
        "video_recommendations": video_ranking[: max(TOP_KS)],
        "author_recommendations": author_ranking[: max(TOP_KS)],
        "video_metrics": _video_case_metrics(video_ranking[: max(TOP_KS)], target_video),
        "author_metrics": _author_case_metrics(author_ranking[: max(TOP_KS)], target_author),
        "target_author": target_author,
        "target_video_in_catalogue": target_video in engine.catalogue,
        "target_video_in_candidates": target_video in candidates,
        "target_author_in_catalogue": (
            target_author in engine.author_catalogue if target_author is not None else False
        ),
        "target_author_in_candidate_authors": (
            target_author in candidate_authors if target_author is not None else False
        ),
        "candidate_pool_size": len(candidates),
    }


def _evaluate_queries(engine, process):
    predictions = {}
    records_by_model = {}
    equality_checks = 0
    candidate_sets_created = 0
    peak_working_set_mb = _peak_working_set_mb(process)
    for row_number, row in enumerate(engine.val_df.itertuples(index=False)):
        identifier = query_id(engine.gap, row_number, row)
        source = int(row.src_video_id)
        candidate_set = engine.common_candidates(source)
        candidate_sets_created += 1
        for treatment in TREATMENTS:
            model_candidates = {model: candidate_set for model in MODEL_NAMES}
            assert_candidate_set_equality(model_candidates)
            equality_checks += 1
            query_key = event_key(row.user_id, row.src_video_id, row.src_time_ms)
            query_weight = engine.query_weights[treatment].get(query_key)
            if query_weight is None:
                raise KeyError(f"Missing query reliability lookup for {identifier}/{treatment}.")
            for model in MODEL_NAMES:
                ranking = rank_candidates(
                    model_candidates[model],
                    engine._score_function(
                        treatment,
                        model,
                        source,
                        model_candidates[model],
                        query_weight,
                    ),
                    engine.video_popularity,
                )
                if len(ranking) < max(TOP_KS):
                    raise AssertionError(f"{model}/{treatment} returned fewer than Top-20.")
                if not set(ranking).issubset(engine.catalogue):
                    raise AssertionError(f"{model}/{treatment} recommended outside the catalogue.")
                key = (treatment, model)
                predictions.setdefault(key, []).append(
                    {"query_id": identifier, "ranking": ranking}
                )
                records_by_model.setdefault(key, []).append(
                    _record_metrics(engine, row, candidate_set, identifier, ranking)
                )
        peak_working_set_mb = max(
            peak_working_set_mb,
            _peak_working_set_mb(process),
        )
    return predictions, records_by_model, {
        "candidate_sets_created": candidate_sets_created,
        "candidate_set_equality_checks": equality_checks,
        "model_candidate_set_comparisons": equality_checks * len(MODEL_NAMES),
        "all_candidate_sets_identical": True,
        "peak_working_set_mb": peak_working_set_mb,
    }


def _overlap_rows(gap, predictions):
    rows = []
    same_treatment_pairs = (
        ("M1", "M4_W1"),
        ("M1", "M4_W4"),
        ("M1", "M5A"),
        ("M1", "M5B"),
        ("M1", "M5C"),
        ("M4_W4", "M5A"),
        ("M4_W4", "M5B"),
        ("M4_W4", "M5C"),
    )
    comparisons = []
    for treatment in TREATMENTS:
        comparisons.extend(
            (left, right, treatment, treatment)
            for left, right in same_treatment_pairs
        )
    for model in ("M4_W4", "M5A", "M5B", "M5C"):
        comparisons.append((model, model, "D0", "D1"))

    for left_model, right_model, left_treatment, right_treatment in comparisons:
        left_records = predictions[(left_treatment, left_model)]
        right_records = predictions[(right_treatment, right_model)]
        left_ids = [record["query_id"] for record in left_records]
        right_ids = [record["query_id"] for record in right_records]
        assert_query_alignment(left_ids, right_ids)
        for k in TOP_KS:
            diagnostics = [
                topk_rank_diagnostics(
                    left_record["ranking"],
                    right_record["ranking"],
                    k,
                )
                for left_record, right_record in zip(left_records, right_records)
            ]
            rows.append(
                {
                    "gap": gap,
                    "comparison": (
                        f"{left_model}_vs_{right_model}"
                        if left_treatment == right_treatment
                        else f"{left_model}_duration_sensitivity"
                    ),
                    "left_treatment": left_treatment,
                    "right_treatment": right_treatment,
                    "k": k,
                    "n_aligned_queries": len(left_ids),
                    "mean_overlap": float(np.mean([item["overlap"] for item in diagnostics])),
                    "mean_jaccard": float(np.mean([item["jaccard"] for item in diagnostics])),
                    "mean_position_agreement": float(
                        np.mean([item["position_agreement"] for item in diagnostics])
                    ),
                    "mean_absolute_rank_displacement": float(
                        np.nanmean(
                            [item["mean_absolute_rank_displacement"] for item in diagnostics]
                        )
                    ),
                }
            )
    return rows


def _peak_working_set_mb(process):
    memory = process.memory_info()
    return getattr(memory, "peak_wset", memory.rss) / 1_048_576


def _population_specs(metric_type):
    if metric_type == "next_video":
        return (
            ("all_validation_targets", lambda record: True),
            ("target_videos_in_training_catalogue", lambda record: record["target_video_in_catalogue"]),
            ("target_videos_in_query_candidate_set", lambda record: record["target_video_in_candidates"]),
        )
    return (
        ("all_validation_targets", lambda record: True),
        ("target_authors_in_training_catalogue", lambda record: record["target_author_in_catalogue"]),
        (
            "target_authors_in_query_candidate_authors",
            lambda record: record["target_author_in_candidate_authors"],
        ),
    )


def _metric_rows(gap, engine, records_by_model):
    rows = []
    for (treatment, model), records in records_by_model.items():
        for metric_type in ("next_video", "next_author"):
            for population, selector in _population_specs(metric_type):
                selected = [record for record in records if selector(record)]
                aggregate = aggregate_population(
                    selected,
                    metric_type,
                    video_catalogue_size=len(engine.catalogue),
                    author_catalogue_size=len(engine.author_catalogue),
                )
                bootstrap_metrics = [
                    f"hit@{k}" for k in TOP_KS
                ] + [f"ndcg@{k}" for k in TOP_KS] + [f"mrr@{k}" for k in TOP_KS]
                intervals = _cluster_bootstrap(selected, bootstrap_metrics, metric_type)
                result = {
                    "gap": gap,
                    "treatment": treatment,
                    "model": model,
                    "metric_type": metric_type,
                    "population": population,
                    "n_all_validation_queries": len(engine.val_df),
                    "n_population_queries": aggregate["n_population_queries"],
                    "n_cases": aggregate["n_cases"],
                    "n_unscorable": aggregate["n_unscorable"],
                    "candidate_pool_mean": (
                        float(np.mean([record["candidate_pool_size"] for record in selected]))
                        if selected
                        else float("nan")
                    ),
                    "unique_recommendations": aggregate["unique_recommendations"],
                    "recommendation_catalogue_size": aggregate["recommendation_catalogue_size"],
                    "recommendation_coverage": aggregate["recommendation_coverage"],
                }
                for k in TOP_KS:
                    for metric in (f"hit@{k}", f"ndcg@{k}", f"mrr@{k}"):
                        result[metric] = aggregate[metric]
                        result[f"{metric}_ci_lower"] = intervals[metric][0]
                        result[f"{metric}_ci_upper"] = intervals[metric][1]
                    result[f"hit_count@{k}"] = aggregate[f"hit_count@{k}"]
                rows.append(result)
    return rows


def run_gap(gap, run_dir, data_dir=DATA_DIR):
    started = time.perf_counter()
    process = psutil.Process()
    engine = Phase4ExperimentEngine(gap, data_dir=data_dir)
    predictions, records_by_model, equality_diagnostics = _evaluate_queries(engine, process)

    metrics_path = run_dir / f"validation_metrics_{gap}min.csv"
    pd.DataFrame(_metric_rows(gap, engine, records_by_model)).to_csv(metrics_path, index=False)
    overlap_path = run_dir / f"ranking_overlap_{gap}min.csv"
    overlap_rows = _overlap_rows(gap, predictions)
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
    diagnostic_path = run_dir / f"candidate_diagnostics_{gap}min.json"

    total_candidate_comparisons = len(engine.val_df) * len(TREATMENTS) * len(MODEL_NAMES)
    diagnostics = {
        "gap_minutes": gap,
        "candidate_policy": (
            "training source-neighbor videos union the shared global Top-100 "
            "training-popularity videos"
        ),
        "filtering_policy": "No source-item or previously-seen-item exclusion.",
        "candidate_generation_count": equality_diagnostics["candidate_sets_created"],
        "candidate_set_equality_checks": equality_diagnostics["candidate_set_equality_checks"],
        "model_candidate_set_comparisons": equality_diagnostics[
            "model_candidate_set_comparisons"
        ],
        "expected_model_candidate_set_comparisons": total_candidate_comparisons,
        "all_candidate_sets_identical": equality_diagnostics["all_candidate_sets_identical"]
        and equality_diagnostics["model_candidate_set_comparisons"]
        == total_candidate_comparisons,
        "query_alignment_assertions": len(overlap_rows) // len(TOP_KS),
        "validation_query_count": len(engine.val_df),
        "treatments": list(TREATMENTS),
        "models": list(MODEL_NAMES),
        "lookup_diagnostics": engine.lookup_diagnostics,
        "timestamp_only_log_scans": engine.log_scan_diagnostics,
        "maximum_loaded_event_time_ms": engine.log_scan_diagnostics[
            "maximum_loaded_event_time_ms"
        ],
        "standard_integer_count_eb_alpha0": engine.alpha_eb_video,
        "training_mean_W4": engine.mean_w4,
        "peak_working_set_mb": equality_diagnostics["peak_working_set_mb"],
    }
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2))
    elapsed = time.perf_counter() - started

    del engine, predictions, records_by_model
    gc.collect()
    return {
        "gap_minutes": gap,
        "elapsed_seconds": elapsed,
        "peak_working_set_mb": equality_diagnostics["peak_working_set_mb"],
        "validation_cases": diagnostics["validation_query_count"],
        "candidate_generation_count": diagnostics["candidate_generation_count"],
        "candidate_set_equality_checks": diagnostics["candidate_set_equality_checks"],
        "model_candidate_set_comparisons": diagnostics["model_candidate_set_comparisons"],
        "all_candidate_sets_identical": diagnostics["all_candidate_sets_identical"],
        "outputs": [metrics_path.name, overlap_path.name, diagnostic_path.name],
        "data_scope": "train transitions and validation transitions only; no test outcomes loaded",
    }


def run_phase4(data_dir=DATA_DIR, out_dir=RESULTS_ROOT):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    run_dir = out_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir()
    gap_results = [run_gap(gap, run_dir, data_dir=data_dir) for gap in GAPS]
    metadata = {
        "run_id": run_id,
        "configuration": {
            "gaps_minutes": list(GAPS),
            "models": list(MODEL_NAMES),
            "treatments": list(TREATMENTS),
            "candidate_policy": "training successors union global Top-100 by training popularity",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_ROUNDS,
        },
        "gaps": gap_results,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    console_text = json.dumps({"run_directory": run_dir.name, **metadata}, indent=2)
    (run_dir / "console.log").write_text(console_text + "\n", encoding="utf-8")
    print(console_text)
    return run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation-only Phase 4 evaluation.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    run_phase4(data_dir=args.data_dir, out_dir=args.out_dir)
