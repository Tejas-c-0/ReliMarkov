"""Post-run integrity checks for validation-only Phase 4 outputs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from kr_experiments_phase4 import (
    BOOTSTRAP_ROUNDS,
    BOOTSTRAP_SEED,
    GAPS,
    MODEL_NAMES,
    TREATMENTS,
    VAL_END_MS,
    Phase4ExperimentEngine,
    assert_candidate_set_equality,
    query_id,
    rank_candidates,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metrics(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError(f"Metrics file is empty: {path.name}")
    for row in rows:
        population_queries = int(row["n_population_queries"])
        cases = int(row["n_cases"])
        unscorable = int(row["n_unscorable"])
        if population_queries != cases + unscorable:
            raise AssertionError("Population queries do not equal cases plus unscorable.")
        if row["metric_type"] == "next_video" and unscorable != 0:
            raise AssertionError("Next-video metrics unexpectedly contain unscorable cases.")
        for k in (10, 20):
            hits = int(row[f"hit_count@{k}"])
            if not 0 <= hits <= cases:
                raise AssertionError(f"Invalid Hit@{k} count.")
            rate = float(row[f"hit@{k}"])
            expected = hits / cases if cases else float("nan")
            if math.isnan(expected):
                if not math.isnan(rate):
                    raise AssertionError(f"Hit@{k} should be NaN for an empty denominator.")
            elif not math.isclose(rate, expected, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"Hit@{k} rate differs from hit count / denominator.")
    return len(rows)


def _sample_ranking_hash(engine, limit=1000):
    digest = hashlib.sha256()
    sample = engine.val_df.iloc[:limit]
    query_ids = []
    for row_number, row in enumerate(sample.itertuples(index=False)):
        identifier = query_id(engine.gap, row_number, row)
        query_ids.append(identifier)
        candidates = engine.common_candidates(int(row.src_video_id))
        candidate_sets = {model: candidates for model in MODEL_NAMES}
        assert_candidate_set_equality(candidate_sets)
        if len(candidates) < 20:
            raise AssertionError("A query candidate set has fewer than 20 items.")
        query_key = (int(row.user_id), int(row.src_video_id), int(row.src_time_ms))
        for treatment in TREATMENTS:
            query_weight = engine.query_weights[treatment].get(query_key)
            if query_weight is None:
                raise KeyError(f"Missing query reliability lookup for {identifier}/{treatment}.")
            for model in MODEL_NAMES:
                ranking = rank_candidates(
                    candidates,
                    engine._score_function(
                        treatment,
                        model,
                        int(row.src_video_id),
                        candidates,
                        query_weight,
                    ),
                    engine.video_popularity,
                )
                digest.update(
                    json.dumps(
                        [identifier, treatment, model, [int(video_id) for video_id in ranking]],
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
    if query_ids != sorted(query_ids, key=lambda value: int(value.split(":", 2)[1])):
        raise AssertionError("Subsample query IDs are not in validation row order.")
    return digest.hexdigest(), len(query_ids)


def validate_run(data_dir, run_dir):
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != run_dir.name:
        raise AssertionError("Run metadata ID does not match the output directory.")
    configuration = metadata.get("configuration", {})
    if configuration.get("gaps_minutes") != list(GAPS):
        raise AssertionError("The recorded gap configuration differs from the specification.")
    if configuration.get("models") != list(MODEL_NAMES):
        raise AssertionError("The recorded model list differs from the specification.")
    if configuration.get("treatments") != list(TREATMENTS):
        raise AssertionError("The recorded treatment list differs from the specification.")
    if configuration.get("candidate_policy") != (
        "training successors union global Top-100 by training popularity"
    ):
        raise AssertionError("The recorded candidate policy differs from the specification.")
    if configuration.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise AssertionError("The recorded bootstrap seed differs from the specified seed.")
    if configuration.get("bootstrap_resamples") != BOOTSTRAP_ROUNDS:
        raise AssertionError("The recorded bootstrap resample count differs from the specification.")

    inventory_path = run_dir.parent / "phase4_output_inventory_before.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if run_dir.name in inventory["entries"]:
        raise AssertionError("Run directory existed in the pre-run output inventory.")

    gap_metadata = {int(item["gap_minutes"]): item for item in metadata["gaps"]}
    if set(gap_metadata) != set(GAPS):
        raise AssertionError("Run metadata does not contain exactly both required gaps.")

    phase3_hashes_path = run_dir.parent / "phase3_hashes_before.json"
    phase3_hashes = json.loads(phase3_hashes_path.read_text(encoding="utf-8"))
    phase3_dir = run_dir.parent.parent / "phase3_validation" / "run_20261103_213050_674910100"
    for filename, expected_hash in phase3_hashes.items():
        path = phase3_dir / filename
        if _sha256(path) != expected_hash:
            raise AssertionError(f"Phase 3 output hash changed: {filename}")

    reports = {}
    for gap in GAPS:
        tag = f"{gap}min"
        diagnostics_path = run_dir / f"candidate_diagnostics_{tag}.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        query_count = int(diagnostics["validation_query_count"])
        expected_comparisons = query_count * len(TREATMENTS) * len(MODEL_NAMES)
        if not diagnostics["all_candidate_sets_identical"]:
            raise AssertionError(f"Candidate sets differ for {tag}.")
        if int(diagnostics["model_candidate_set_comparisons"]) != expected_comparisons:
            raise AssertionError(f"Candidate equality checks are incomplete for {tag}.")
        if int(diagnostics["expected_model_candidate_set_comparisons"]) != expected_comparisons:
            raise AssertionError(f"Expected candidate comparison count is wrong for {tag}.")
        if int(diagnostics["query_alignment_assertions"]) < 1:
            raise AssertionError(f"No query-ID alignment checks were recorded for {tag}.")
        max_time = diagnostics.get("maximum_loaded_event_time_ms")
        if max_time is None or int(max_time) >= VAL_END_MS + 1:
            raise AssertionError(f"A loaded event reached the validation cutoff for {tag}.")

        metrics_count = _check_metrics(run_dir / f"validation_metrics_{tag}.csv")
        overlap_path = run_dir / f"ranking_overlap_{tag}.csv"
        with overlap_path.open(newline="", encoding="utf-8") as stream:
            overlap_rows = list(csv.DictReader(stream))
        if not overlap_rows or any(
            int(row["n_aligned_queries"]) != query_count for row in overlap_rows
        ):
            raise AssertionError(f"Ranking overlap query counts differ for {tag}.")
        if int(diagnostics["query_alignment_assertions"]) != len(overlap_rows) // 2:
            raise AssertionError(f"Query-ID sequence checks are incomplete for {tag}.")

        engine = Phase4ExperimentEngine(gap, data_dir=data_dir)
        event_times = [key[2] for key in engine.event_records]
        transition_times = []
        for frame in (engine.train_df, engine.val_df):
            for col in ("src_time_ms", "tgt_time_ms"):
                if col in frame:
                    transition_times.extend(frame[col].dropna().astype("int64").tolist())
        all_loaded_times = event_times + transition_times
        if not all_loaded_times or max(all_loaded_times) >= VAL_END_MS + 1:
            raise AssertionError(f"A loaded event/transition reached the validation cutoff for {tag}.")
        first_hash, sample_count = _sample_ranking_hash(engine)
        second_hash, second_count = _sample_ranking_hash(engine)
        if first_hash != second_hash or sample_count != second_count:
            raise AssertionError(f"Repeated subsample ranking hashes differ for {tag}.")
        if int(diagnostics["standard_integer_count_eb_alpha0"]) <= 0:
            raise AssertionError(f"Invalid EB concentration in {tag} diagnostics.")
        reports[tag] = {
            "query_count": query_count,
            "metrics_rows_checked": metrics_count,
            "candidate_comparisons": expected_comparisons,
            "maximum_loaded_time_ms": max(all_loaded_times),
            "subsample_queries": sample_count,
            "subsample_ranking_sha256": first_hash,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_ROUNDS,
            "phase3_hashes_unchanged": True,
            "new_output_directory": True,
        }

    return {"run_id": run_dir.name, "gaps": reports, "valid": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Path to one timestamped Phase 4 run directory.",
    )
    args = parser.parse_args()
    report = validate_run(args.data_dir, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
