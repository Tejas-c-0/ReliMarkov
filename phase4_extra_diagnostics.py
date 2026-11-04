"""Compute descriptive, validation-only Phase 4 diagnostics."""

import argparse
import json
from pathlib import Path

import psutil

from kr_experiments_phase4 import (
    MODEL_NAMES,
    TREATMENTS,
    Phase4ExperimentEngine,
    _evaluate_queries,
    _population_specs,
    aggregate_population,
)


def _population_counts(engine, records_by_model):
    results = []
    for (treatment, model), records in records_by_model.items():
        for metric_type in ("next_video", "next_author"):
            for population, selector in _population_specs(metric_type):
                selected = [record for record in records if selector(record)]
                eligible = [
                    record
                    for record in selected
                    if metric_type == "next_video" or record["target_author"] is not None
                ]
                aggregate = aggregate_population(
                    selected,
                    metric_type,
                    video_catalogue_size=len(engine.catalogue),
                    author_catalogue_size=len(engine.author_catalogue),
                )
                results.append(
                    {
                        "treatment": treatment,
                        "model": model,
                        "metric_type": metric_type,
                        "population": population,
                        "n_population_queries": aggregate["n_population_queries"],
                        "n_cases": aggregate["n_cases"],
                        "n_unscorable": aggregate["n_unscorable"],
                        "n_distinct_users_in_bootstrap": len(
                            {record["user_id"] for record in eligible}
                        ),
                        "n_bootstrap_queries": len(eligible),
                        "hit_count@10": aggregate["hit_count@10"],
                        "hit_count@20": aggregate["hit_count@20"],
                    }
                )
        source_groups = {
            "source_known": [],
            "source_unknown": [],
        }
        source_key_known = {
            record["query_id"]: record["source_known"] for record in records
        }
        for name in source_groups:
            selected = [
                record
                for record in records
                if bool(source_key_known[record["query_id"]]) == (name == "source_known")
            ]
            video = aggregate_population(
                selected,
                "next_video",
                video_catalogue_size=len(engine.catalogue),
                author_catalogue_size=len(engine.author_catalogue),
            )
            author = aggregate_population(
                selected,
                "next_author",
                video_catalogue_size=len(engine.catalogue),
                author_catalogue_size=len(engine.author_catalogue),
            )
            results.append(
                {
                    "treatment": treatment,
                    "model": model,
                    "metric_type": "source_subset",
                    "population": name,
                    "n_population_queries": len(selected),
                    "n_distinct_users": len({record["user_id"] for record in selected}),
                    "video_hit_count@10": video["hit_count@10"],
                    "video_hit_count@20": video["hit_count@20"],
                    "author_hit_count@10": author["hit_count@10"],
                    "author_hit_count@20": author["hit_count@20"],
                }
            )
    return results


def _top20_slot_shares(engine, predictions):
    popular = set(engine.global_top_100)
    counters = {}
    for treatment in TREATMENTS:
        for model in MODEL_NAMES:
            groups = {
                "source_known": {"queries": 0, "slots": 0, "global_top_100": 0, "successor_only": 0, "both": 0},
                "source_unknown": {"queries": 0, "slots": 0, "global_top_100": 0, "successor_only": 0, "both": 0},
            }
            rows = engine.val_df.itertuples(index=False)
            for row, prediction in zip(rows, predictions[(treatment, model)]):
                source = int(row.src_video_id)
                known = source in engine.tables[treatment]["counts"]
                group = groups["source_known" if known else "source_unknown"]
                neighbors = set(engine.tables[treatment]["counts"].get(source, {}))
                top20 = prediction["ranking"][:20]
                group["queries"] += 1
                group["slots"] += len(top20)
                group["global_top_100"] += sum(video_id in popular for video_id in top20)
                group["successor_only"] += sum(
                    video_id in neighbors and video_id not in popular for video_id in top20
                )
                group["both"] += sum(
                    video_id in neighbors and video_id in popular for video_id in top20
                )
            counters[f"{treatment}/{model}"] = {
                group_name: {
                    **counts,
                    "share_global_top_100": (
                        counts["global_top_100"] / counts["slots"] if counts["slots"] else 0.0
                    ),
                    "share_successor_only": (
                        counts["successor_only"] / counts["slots"] if counts["slots"] else 0.0
                    ),
                    "share_both": (
                        counts["both"] / counts["slots"] if counts["slots"] else 0.0
                    ),
                }
                for group_name, counts in groups.items()
            }
    return counters


def diagnose_gap(gap, data_dir, out_dir):
    engine = Phase4ExperimentEngine(gap, data_dir=data_dir)
    predictions, records_by_model, _ = _evaluate_queries(engine, psutil.Process())
    records_with_source = {}
    for key, records in records_by_model.items():
        records_with_source[key] = [
            {
                **record,
                "source_known": int(
                    int(engine.val_df.iloc[index]["src_video_id"])
                    in engine.tables[key[0]]["counts"]
                ),
            }
            for index, record in enumerate(records)
        ]

    total_popularity = sum(engine.video_popularity.values())
    alpha_mass = {
        video_id: engine.alpha_eb_video * count / total_popularity
        for video_id, count in engine.video_popularity.items()
    }
    diagnostics = {
        "gap_minutes": gap,
        "scope": "training and validation only",
        "video_eb_alpha0": engine.alpha_eb_video,
        "videos_alpha_times_normalized_popularity_at_least_one": sum(
            value >= 1.0 for value in alpha_mass.values()
        ),
        "largest_alpha_times_normalized_popularity": max(alpha_mass.values()),
        "top20_slot_shares": _top20_slot_shares(engine, predictions),
        "top20_hits_and_bootstrap_population_sizes": _population_counts(
            engine, records_with_source
        ),
        "distinct_users_and_queries_are_counted_after_author_scoring_exclusions": True,
    }
    output_path = Path(out_dir) / f"phase4_extra_diagnostics_{gap}min.json"
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gap", type=int, choices=(30, 60), required=True)
    args = parser.parse_args()
    report = diagnose_gap(args.gap, args.data_dir, args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
