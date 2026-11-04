"""Phase 3 validation-only implementation; historical experiment code is untouched."""

import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
EB_DIR = ROOT / "outputs" / "eb_params"
RESULTS_ROOT = ROOT / "outputs" / "phase3_validation"
LOG_FILES = [
    DATA_DIR / "log_standard_4_08_to_4_21_1k.csv",
    DATA_DIR / "log_standard_4_22_to_5_08_1k.csv",
]
VIDEO_META = DATA_DIR / "video_features_basic_1k.csv"
GAPS = (30, 60)
TOP_KS = (10, 20)
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 42

TRAIN_START_MS = 1649347200000
TRAIN_END_MS = 1651161599999
VAL_END_MS = 1651593599999
DURATION_BIN_EDGES_MS = (7000, 18000, 60000)
BASE_TREATMENTS = {"D0": 0.0, "D1": 0.9016}

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


def _event_key(user_id, video_id, time_ms):
    return int(user_id), int(video_id), int(time_ms)


def _as_float(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    return float(value)


def _duration_bin(duration_ms):
    return int(np.digitize(float(duration_ms), DURATION_BIN_EDGES_MS, right=True))


def estimate_duration_medians(training_ratios_by_bin):
    """Estimate duration-bin and global watch-ratio medians from training events."""
    populated = [
        np.asarray(values, dtype=np.float64)
        for values in training_ratios_by_bin
        if len(values) > 0
    ]
    if not populated:
        raise ValueError("D2 requires positive-duration training interactions.")
    global_median = float(np.median(np.concatenate(populated)))
    medians = [
        float(np.median(values)) if len(values) > 0 else global_median
        for values in training_ratios_by_bin
    ]
    return global_median, medians


def event_reliability(
    event,
    treatment,
    duration_medians,
    metadata_duration_ms=None,
    global_duration_median=None,
):
    """Apply one shared reliability definition to every phase-3 treatment."""
    duration_ms = _as_float(event.get("duration_ms", 0))
    play_time_ms = _as_float(event.get("play_time_ms", 0))

    if duration_ms > 0:
        watch_ratio = float(np.clip(play_time_ms / duration_ms, 0.0, 1.0))
    elif treatment == "D0":
        watch_ratio = 0.0
    elif treatment == "D1":
        watch_ratio = 0.9016
    elif treatment == "D2":
        if metadata_duration_ms is not None and np.isfinite(metadata_duration_ms) and metadata_duration_ms > 0:
            watch_ratio = duration_medians[_duration_bin(metadata_duration_ms)]
        else:
            watch_ratio = (
                global_duration_median
                if global_duration_median is not None
                else duration_medians[-1]
            )
    else:
        raise ValueError(f"Unknown duration treatment: {treatment}")

    is_like = _as_float(event.get("is_like", 0))
    is_forward = _as_float(event.get("is_forward", 0))
    is_hate = _as_float(event.get("is_hate", 0))
    return float(np.clip(watch_ratio + 0.3 * is_like + 0.2 * is_forward - 0.5 * is_hate, 0.1, 1.5))


def score_m4(count, weighted_count, query_weight, prior_mass, scheme, alpha=1.0):
    """M4 scores; W1 is fully unweighted, W2-W4 use query reliability."""
    if scheme == "W1":
        evidence = count
    elif scheme in ("W2", "W3", "W4"):
        evidence = query_weight * weighted_count
    else:
        raise ValueError(f"Unknown M4 weighting scheme: {scheme}")
    return evidence + alpha * prior_mass


def score_m5a(weighted_count, query_weight, alpha_eb, prior_mass):
    return query_weight * weighted_count + alpha_eb * prior_mass


def score_m5b(weighted_count, query_weight, mean_w4, alpha_eb, prior_mass):
    if not np.isfinite(mean_w4) or mean_w4 <= 0:
        raise ValueError("M5-B requires a positive finite training mean W4.")
    return query_weight * (weighted_count / mean_w4) + alpha_eb * prior_mass


def score_m5c(weighted_count, query_weight, mean_w4, alpha_eb, prior_mass):
    if not np.isfinite(mean_w4) or mean_w4 <= 0:
        raise ValueError("M5-C requires a positive finite training mean W4.")
    return query_weight * weighted_count + (mean_w4 * alpha_eb) * prior_mass


def topk_overlap(left, right, k):
    left_set = set(left[:k])
    right_set = set(right[:k])
    union = left_set | right_set
    return {
        "overlap": len(left_set & right_set) / k,
        "jaccard": len(left_set & right_set) / len(union) if union else 1.0,
    }


def count_missing_metadata_durations(zero_duration_event_keys, metadata_duration):
    return sum(key[1] not in metadata_duration for key in zero_duration_event_keys)


def duration_treatment_supported(zero_duration_event_keys, metadata_duration):
    return count_missing_metadata_durations(zero_duration_event_keys, metadata_duration) == 0


class Phase3ExperimentEngine:
    """Build training-only transition statistics and validation query weights."""

    def __init__(self, gap):
        if gap not in GAPS:
            raise ValueError(f"Unsupported gap: {gap}")
        self.gap = gap
        self.tag = f"{gap}min"
        self.train_df = pd.read_parquet(SPLIT_DIR / f"train_transitions_{self.tag}.parquet")
        self.val_df = pd.read_parquet(SPLIT_DIR / f"val_transitions_{self.tag}.parquet")
        self.catalogue = set(json.loads((SPLIT_DIR / f"train_catalogue_{self.tag}.json").read_text()))
        self.video_pop = {
            int(key): int(value)
            for key, value in json.loads((SPLIT_DIR / f"video_popularity_{self.tag}.json").read_text()).items()
        }
        self.author_video_map = {
            int(key): [int(video_id) for video_id in value]
            for key, value in json.loads((SPLIT_DIR / f"author_video_map_{self.tag}.json").read_text()).items()
        }
        eb_data = json.loads((EB_DIR / f"eb_hyperparams_{self.tag}.json").read_text())
        self.alpha_eb_video = float(eb_data["video_eb"]["alpha0"])
        total_popularity = sum(self.video_pop.values())
        self.video_prior = {
            video_id: count / total_popularity
            for video_id, count in self.video_pop.items()
        }
        self.global_top_100 = sorted(self.video_pop, key=lambda video_id: (-self.video_pop[video_id], video_id))[:100]
        self.global_top_50 = self.global_top_100[:50]
        self.video_to_author = {
            video_id: author_id
            for author_id, videos in self.author_video_map.items()
            for video_id in videos
        }

        self._load_relevant_events_and_fit_d2()
        self.treatments = dict(BASE_TREATMENTS)
        self.d2_unavailable_reason = None
        if duration_treatment_supported(self.zero_duration_keys, self.metadata_duration):
            self.treatments["D2"] = None
        elif self.zero_duration_keys:
            missing = count_missing_metadata_durations(self.zero_duration_keys, self.metadata_duration)
            self.d2_unavailable_reason = (
                f"Not evaluated: video-duration metadata is missing for {missing} of "
                f"{len(self.zero_duration_keys)} distinct zero-duration source/transition events."
            )
        self.tables = {}
        self.query_weights = {}
        self.lookup_diagnostics = {}
        self.mean_w4 = {}
        self._build_treatment_tables()

    def _required_event_keys(self):
        keys = set()
        for row in self.train_df.itertuples(index=False):
            keys.add(_event_key(row.user_id, row.src_video_id, row.src_time_ms))
            keys.add(_event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms))
        for row in self.val_df.itertuples(index=False):
            keys.add(_event_key(row.user_id, row.src_video_id, row.src_time_ms))
        return keys

    def _load_relevant_events_and_fit_d2(self):
        required_keys = self._required_event_keys()
        required_index = pd.MultiIndex.from_tuples(sorted(required_keys), names=EVENT_KEY_COLUMNS)
        event_records = {}
        duplicate_event_keys = 0
        duration_ratio_parts = [[] for _ in range(len(DURATION_BIN_EDGES_MS) + 1)]

        for log_path in LOG_FILES:
            for chunk in pd.read_csv(log_path, usecols=EVENT_COLUMNS, chunksize=200_000):
                time_ms = pd.to_numeric(chunk["time_ms"], errors="coerce")
                through_validation = time_ms.le(VAL_END_MS)
                dataset_a = chunk["tab"].eq(1) & chunk["is_click"].eq(1)
                in_training = time_ms.ge(TRAIN_START_MS) & time_ms.le(TRAIN_END_MS)

                training_rows = chunk.loc[through_validation & dataset_a & in_training]
                train_duration = pd.to_numeric(training_rows["duration_ms"], errors="coerce")
                positive_duration = train_duration.gt(0)
                if positive_duration.any():
                    positive_rows = training_rows.loc[positive_duration]
                    durations = train_duration.loc[positive_duration].to_numpy(dtype=np.float64)
                    plays = pd.to_numeric(positive_rows["play_time_ms"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
                    ratios = np.clip(plays / durations, 0.0, 1.0)
                    bins = np.digitize(durations, DURATION_BIN_EDGES_MS, right=True)
                    for bin_id in range(len(duration_ratio_parts)):
                        values = ratios[bins == bin_id]
                        if values.size:
                            duration_ratio_parts[bin_id].append(values)

                eligible = chunk.loc[through_validation & dataset_a]
                if eligible.empty:
                    continue
                eligible_keys = pd.MultiIndex.from_frame(eligible[EVENT_KEY_COLUMNS])
                matched = eligible.loc[eligible_keys.isin(required_index)]
                for row in matched.itertuples(index=False):
                    key = _event_key(row.user_id, row.video_id, row.time_ms)
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

        training_ratios_by_bin = [
            np.concatenate(parts) if parts else np.array([], dtype=np.float64)
            for parts in duration_ratio_parts
        ]
        global_median, bin_medians = estimate_duration_medians(training_ratios_by_bin)
        self.duration_medians = bin_medians
        self.duration_global_median = global_median
        self.event_records = event_records
        self.duplicate_event_key_count = duplicate_event_keys
        self.required_event_keys = required_keys
        self.zero_duration_keys = {
            key
            for key, event in event_records.items()
            if _as_float(event["duration_ms"]) <= 0
        }

        zero_duration_video_ids = {
            key[1] for key in self.zero_duration_keys
        }
        self.metadata_duration = self._load_metadata_durations(zero_duration_video_ids)

    @staticmethod
    def _load_metadata_durations(video_ids):
        if not video_ids:
            return {}
        remaining = set(video_ids)
        durations = {}
        for chunk in pd.read_csv(
            VIDEO_META,
            usecols=["video_id", "video_duration"],
            chunksize=250_000,
        ):
            matched = chunk.loc[chunk["video_id"].isin(remaining)]
            for row in matched.itertuples(index=False):
                value = pd.to_numeric(pd.Series([row.video_duration]), errors="coerce").iloc[0]
                if pd.notna(value) and value > 0:
                    durations[int(row.video_id)] = float(value)
                    remaining.discard(int(row.video_id))
            if not remaining:
                break
        return durations

    def _event_weight(self, key, treatment):
        event = self.event_records[key]
        duration_value = _as_float(event["duration_ms"])
        metadata_duration = self.metadata_duration.get(key[1]) if duration_value <= 0 else None
        return event_reliability(
            event,
            treatment,
            self.duration_medians,
            metadata_duration,
            self.duration_global_median,
        )

    def _build_treatment_tables(self):
        for treatment in self.treatments:
            known_training_endpoint_weights = []
            for row in self.train_df.itertuples(index=False):
                for key in (
                    _event_key(row.user_id, row.src_video_id, row.src_time_ms),
                    _event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms),
                ):
                    if key in self.event_records:
                        known_training_endpoint_weights.append(self._event_weight(key, treatment))
            if not known_training_endpoint_weights:
                raise ValueError(f"No training endpoint reliabilities found for {treatment}.")
            training_fallback = float(np.mean(known_training_endpoint_weights))

            video_counts = defaultdict(Counter)
            video_w2 = defaultdict(lambda: defaultdict(float))
            video_w3 = defaultdict(lambda: defaultdict(float))
            video_w4 = defaultdict(lambda: defaultdict(float))
            source_weights = {}
            w4_by_transition = []
            missing_training_endpoints = 0

            for row in self.train_df.itertuples(index=False):
                src_key = _event_key(row.user_id, row.src_video_id, row.src_time_ms)
                tgt_key = _event_key(row.user_id, row.tgt_video_id, row.tgt_time_ms)
                if src_key in self.event_records:
                    w_src = self._event_weight(src_key, treatment)
                else:
                    w_src = training_fallback
                    missing_training_endpoints += 1
                if tgt_key in self.event_records:
                    w_tgt = self._event_weight(tgt_key, treatment)
                else:
                    w_tgt = training_fallback
                    missing_training_endpoints += 1

                source_weights[src_key] = w_src
                weighted_transition = math.sqrt(w_src * w_tgt)
                video_counts[row.src_video_id][row.tgt_video_id] += 1
                video_w2[row.src_video_id][row.tgt_video_id] += w_src
                video_w3[row.src_video_id][row.tgt_video_id] += w_tgt
                video_w4[row.src_video_id][row.tgt_video_id] += weighted_transition
                w4_by_transition.append(weighted_transition)

            missing_validation_sources = 0
            validation_query_weights = {}
            for row in self.val_df.itertuples(index=False):
                key = _event_key(row.user_id, row.src_video_id, row.src_time_ms)
                if key in self.event_records:
                    validation_query_weights[key] = self._event_weight(key, treatment)
                else:
                    validation_query_weights[key] = training_fallback
                    missing_validation_sources += 1

            missing_metadata_keys = count_missing_metadata_durations(
                self.zero_duration_keys,
                self.metadata_duration,
            )
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
                "missing_distinct_event_keys": len(self.required_event_keys - set(self.event_records)),
                "duplicate_matched_event_keys": self.duplicate_event_key_count,
                "zero_duration_distinct_keys": len(self.zero_duration_keys),
                "zero_duration_keys_without_video_metadata": missing_metadata_keys,
                "fallback_for_missing_lookups": "mean reliability over observed training transition endpoints",
                "training_endpoint_fallback_mean": training_fallback,
                "training_mean_W4_per_transition": self.mean_w4[treatment],
            }

    def _rank(self, candidates, score_function):
        scores = {video_id: score_function(video_id) for video_id in candidates}
        ranked = sorted(
            candidates,
            key=lambda video_id: (
                -scores[video_id],
                -self.video_pop.get(video_id, 0),
                video_id,
            ),
        )
        return ranked[: max(TOP_KS)]

    def recommend(self, row, treatment, model):
        source = int(row.src_video_id)
        query_key = _event_key(row.user_id, row.src_video_id, row.src_time_ms)
        query_weight = self.query_weights[treatment].get(query_key, 0.0)
        tables = self.tables[treatment]
        counts = tables["counts"].get(source, Counter())

        if model == "M1":
            candidates = set(counts) | set(self.global_top_100)
            return self._rank(candidates, lambda video_id: counts[video_id]), len(candidates)

        if model.startswith("M4_"):
            scheme = model[3:]
            table = counts if scheme == "W1" else tables[scheme].get(source, {})
            candidates = set(table) | set(self.global_top_50)
            return self._rank(
                candidates,
                lambda video_id: score_m4(
                    counts[video_id],
                    table.get(video_id, 0.0) if scheme != "W1" else 0.0,
                    query_weight,
                    self.video_prior.get(video_id, 1e-9),
                    scheme,
                ),
            ), len(candidates)

        table = tables["W4"].get(source, {})
        candidates = set(table) | set(self.global_top_50)
        if model == "M5A":
            scorer = lambda video_id: score_m5a(
                table.get(video_id, 0.0),
                query_weight,
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        elif model == "M5B":
            scorer = lambda video_id: score_m5b(
                table.get(video_id, 0.0),
                query_weight,
                self.mean_w4[treatment],
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        elif model == "M5C":
            scorer = lambda video_id: score_m5c(
                table.get(video_id, 0.0),
                query_weight,
                self.mean_w4[treatment],
                self.alpha_eb_video,
                self.video_prior.get(video_id, 1e-9),
            )
        else:
            raise ValueError(f"Unknown model: {model}")
        return self._rank(candidates, scorer), len(candidates)

    def author_ids(self, recommendations):
        return [self.video_to_author.get(video_id) for video_id in recommendations]


def _case_metrics(recommendations, target):
    metrics = {}
    rank = recommendations.index(target) + 1 if target in recommendations else None
    for k in TOP_KS:
        hit = float(rank is not None and rank <= k)
        metrics[f"hit@{k}"] = hit
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        metrics[f"mrr@{k}"] = 1.0 / rank if hit else 0.0
    return metrics


def _cluster_bootstrap(records, metric_names, rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
    if not records:
        return {name: (float("nan"), float("nan")) for name in metric_names}
    frame = pd.DataFrame(records)
    users = frame["user_id"].unique()
    user_index = pd.Index(users)
    row_user = user_index.get_indexer(frame["user_id"])
    user_counts = np.bincount(row_user, minlength=len(users)).astype(np.float64)
    sums = np.zeros((len(users), len(metric_names)), dtype=np.float64)
    for column_index, metric in enumerate(metric_names):
        sums[:, column_index] = np.bincount(
            row_user,
            weights=frame[metric].to_numpy(dtype=np.float64),
            minlength=len(users),
        )

    rng = np.random.RandomState(seed)
    boot = np.empty((rounds, len(metric_names)), dtype=np.float64)
    for round_index in range(rounds):
        sampled_users = rng.choice(len(users), size=len(users), replace=True)
        multiplicities = np.bincount(sampled_users, minlength=len(users)).astype(np.float64)
        denominator = float(np.dot(multiplicities, user_counts))
        boot[round_index] = np.dot(multiplicities, sums) / denominator
    return {
        metric: (
            float(np.percentile(boot[:, index], 2.5)),
            float(np.percentile(boot[:, index], 97.5)),
        )
        for index, metric in enumerate(metric_names)
    }


def _aggregate_metrics(records, all_case_records, engine, model, treatment, gap, subset):
    metrics = [f"hit@{k}" for k in TOP_KS]
    metrics += [f"ndcg@{k}" for k in TOP_KS]
    metrics += [f"mrr@{k}" for k in TOP_KS]
    confidence_intervals = _cluster_bootstrap(records, metrics)
    result = {
        "gap": gap,
        "treatment": treatment,
        "model": model,
        "subset": subset,
        "n_cases": len(records),
        "unique_recommendations": len(
            {video for record in all_case_records for video in record["recommendations"]}
        ),
        "candidate_catalogue_size": len(engine.catalogue),
        "recommendation_coverage": (
            len({video for record in all_case_records for video in record["recommendations"]})
            / len(engine.catalogue)
            if engine.catalogue
            else 0.0
        ),
    }
    for metric in metrics:
        result[metric] = float(np.mean([record[metric] for record in records])) if records else float("nan")
        result[f"{metric}_ci_lower"] = confidence_intervals[metric][0]
        result[f"{metric}_ci_upper"] = confidence_intervals[metric][1]

    pools = [record["candidate_pool_size"] for record in records]
    result["candidate_pool_mean"] = float(np.mean(pools)) if pools else float("nan")
    result["candidate_pool_median"] = float(np.median(pools)) if pools else float("nan")
    result["candidate_pool_p95"] = float(np.percentile(pools, 95)) if pools else float("nan")
    for k in TOP_KS:
        author_counts = [
            len({author for author in record["authors"][:k] if author is not None})
            for record in records
        ]
        result[f"mean_unique_authors@{k}"] = float(np.mean(author_counts)) if author_counts else float("nan")
    return result


def _validation_records(engine, process):
    models = ("M1", "M4_W1", "M4_W2", "M4_W3", "M4_W4", "M5A", "M5B", "M5C")
    predictions = {}
    records_by_model = {}
    peak_rss_mb = _peak_working_set_mb(process)
    for treatment in engine.treatments:
        for model in models:
            recommendations = []
            candidate_sizes = []
            case_records = []
            for row in engine.val_df.itertuples(index=False):
                recs, candidate_size = engine.recommend(row, treatment, model)
                if len(recs) != max(TOP_KS):
                    raise AssertionError(f"{model}/{treatment} returned {len(recs)} recommendations.")
                if not all(video_id in engine.catalogue for video_id in recs):
                    raise AssertionError(f"{model}/{treatment} recommended outside the training catalogue.")
                recommendations.append(recs)
                candidate_sizes.append(candidate_size)
                metrics = _case_metrics(recs, int(row.tgt_video_id))
                metrics.update(
                    {
                        "user_id": int(row.user_id),
                        "target_known": int(row.tgt_video_id in engine.catalogue),
                        "source_known": int(row.src_video_id in engine.tables[treatment]["counts"]),
                        "recommendations": recs,
                        "authors": engine.author_ids(recs),
                        "candidate_pool_size": candidate_size,
                    }
                )
                case_records.append(metrics)
            predictions[(treatment, model)] = recommendations
            records_by_model[(treatment, model)] = case_records
            peak_rss_mb = max(peak_rss_mb, _peak_working_set_mb(process))
    return predictions, records_by_model, peak_rss_mb


def _overlap_rows(gap, predictions):
    rows = []
    treatments = tuple(dict.fromkeys(treatment for treatment, _ in predictions))
    within_treatment_pairs = [
        ("M1", "M4_W1"),
        ("M1", "M4_W4"),
        ("M1", "M5A"),
        ("M1", "M5B"),
        ("M1", "M5C"),
        ("M4_W4", "M5A"),
        ("M4_W4", "M5B"),
        ("M4_W4", "M5C"),
    ]
    for treatment in treatments:
        for left_model, right_model in within_treatment_pairs:
            left = predictions[(treatment, left_model)]
            right = predictions[(treatment, right_model)]
            for k in TOP_KS:
                values = [topk_overlap(a, b, k) for a, b in zip(left, right)]
                rows.append(
                    {
                        "gap": gap,
                        "comparison": f"{left_model}_vs_{right_model}",
                        "left_treatment": treatment,
                        "right_treatment": treatment,
                        "k": k,
                        "mean_overlap": float(np.mean([value["overlap"] for value in values])),
                        "mean_jaccard": float(np.mean([value["jaccard"] for value in values])),
                    }
                )
    cross_treatment_models = ("M4_W4", "M5A", "M5B", "M5C")
    for model in cross_treatment_models:
        for left_index, left_treatment in enumerate(treatments):
            for right_treatment in treatments[left_index + 1 :]:
                left = predictions[(left_treatment, model)]
                right = predictions[(right_treatment, model)]
                for k in TOP_KS:
                    values = [topk_overlap(a, b, k) for a, b in zip(left, right)]
                    rows.append(
                        {
                            "gap": gap,
                            "comparison": f"{model}_duration_sensitivity",
                            "left_treatment": left_treatment,
                            "right_treatment": right_treatment,
                            "k": k,
                            "mean_overlap": float(np.mean([value["overlap"] for value in values])),
                            "mean_jaccard": float(np.mean([value["jaccard"] for value in values])),
                        }
                    )
    return rows


def run_gap(gap, run_dir):
    started = time.perf_counter()
    process = psutil.Process()
    engine = Phase3ExperimentEngine(gap)
    peak_rss_mb = _peak_working_set_mb(process)
    predictions, records_by_model, prediction_peak = _validation_records(engine, process)
    peak_rss_mb = max(peak_rss_mb, prediction_peak, _peak_working_set_mb(process))

    metrics_rows = []
    subset_names = ("all", "target_known", "source_known", "source_unknown")
    for (treatment, model), records in records_by_model.items():
        for subset in subset_names:
            if subset == "all":
                selected = records
            elif subset == "target_known":
                selected = [record for record in records if record["target_known"]]
            elif subset == "source_known":
                selected = [record for record in records if record["source_known"]]
            else:
                selected = [record for record in records if not record["source_known"]]
            metrics_rows.append(
                _aggregate_metrics(selected, records, engine, model, treatment, gap, subset)
            )

    metrics_path = run_dir / f"validation_metrics_{gap}min.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    overlap_path = run_dir / f"ranking_overlap_{gap}min.csv"
    pd.DataFrame(_overlap_rows(gap, predictions)).to_csv(overlap_path, index=False)

    lookup_path = run_dir / f"lookup_diagnostics_{gap}min.json"
    lookup_payload = {
        "gap": gap,
        "duration_bin_edges_ms": list(DURATION_BIN_EDGES_MS),
        "duration_bin_medians_training_only": engine.duration_medians,
        "global_watch_ratio_median_training_only": engine.duration_global_median,
        "D2_status": (
            "evaluated"
            if "D2" in engine.treatments
            else engine.d2_unavailable_reason or "Not evaluated: no zero-duration events."
        ),
        "treatments": engine.lookup_diagnostics,
    }
    lookup_path.write_text(json.dumps(lookup_payload, indent=2))

    elapsed = time.perf_counter() - started
    metadata_path = run_dir / f"run_metadata_{gap}min.json"
    metadata_path.write_text(
        json.dumps(
            {
                "gap_minutes": gap,
                "elapsed_seconds": elapsed,
                "peak_rss_mb": peak_rss_mb,
                "validation_cases": len(engine.val_df),
                "models_per_treatment": 8,
                "duration_treatments": list(lookup_payload["treatments"]),
                "bootstrap_rounds": BOOTSTRAP_ROUNDS,
                "outputs": [metrics_path.name, overlap_path.name, lookup_path.name],
                "data_scope": "train transitions and validation transitions only; no test evaluation",
            },
            indent=2,
        )
    )
    del engine, predictions, records_by_model
    gc.collect()
    return json.loads(metadata_path.read_text())


def _peak_working_set_mb(process):
    memory = process.memory_info()
    return getattr(memory, "peak_wset", memory.rss) / 1_048_576


def run_phase3():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    run_dir = RESULTS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir()
    run_metadata = [run_gap(gap, run_dir) for gap in GAPS]
    overall_path = run_dir / "run_metadata.json"
    overall_path.write_text(json.dumps({"run_id": run_id, "gaps": run_metadata}, indent=2))
    print(json.dumps({"run_directory": str(run_dir), "gaps": run_metadata}, indent=2))
    return run_dir


if __name__ == "__main__":
    run_phase3()
