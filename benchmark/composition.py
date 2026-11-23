"""Create the benchmark's source/donor composition for one condition."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import stable_seed
from .conditions import Condition, all_conditions


def _composition(
    condition: Condition,
    source: pd.DataFrame,
    donor: pd.DataFrame,
    full_s: pd.DataFrame,
    strata: pd.DataFrame,
    replicate_seed: int,
) -> tuple[list[str], list[str], list[int], list[int], float]:
    clean_ids = source.source_id.astype(str).tolist()
    output_ids = clean_ids.copy()
    donor_ids = [""] * len(source)
    injected = [0] * len(source)
    duplicate = [0] * len(source)
    actual_unique = 1.0
    parameter = condition.parameters
    if condition.family == "diversity_collapse":
        requested = float(parameter["unique_fraction"])
        unique_count = max(1, int(round(requested * len(source))))
        rng = np.random.default_rng(stable_seed(condition.dataset, "diversity", replicate_seed))
        priority = rng.permutation(len(source))
        retained_positions = priority[:unique_count]
        retained_ids = [clean_ids[int(index)] for index in retained_positions]
        output_ids = retained_ids.copy()
        fill = len(source) - unique_count
        if fill:
            draw = rng.choice(unique_count, size=fill, replace=True)
            output_ids.extend(retained_ids[int(index)] for index in draw)
            duplicate = [0] * unique_count + [1] * fill
        actual_unique = len(set(output_ids)) / len(output_ids)
    elif condition.family == "memorisation_injection":
        count = int(round(float(parameter["injected_fraction"]) * len(source)))
        rng = np.random.default_rng(stable_seed(condition.dataset, "memorisation", replicate_seed))
        positions = rng.permutation(len(source))[:count]
        donor_order = rng.permutation(len(donor))[:count]
        for position, donor_index in zip(positions, donor_order, strict=True):
            donor_id = str(donor.iloc[int(donor_index)].source_id)
            output_ids[int(position)] = donor_id
            donor_ids[int(position)] = donor_id
            injected[int(position)] = 1
    elif condition.family == "support_loss":
        stratum = strata.set_index("source_id")
        target_column = "common_target" if condition.subtype == "common_stratum" else "rare_target"
        target_positions = [index for index, value in enumerate(clean_ids) if bool(stratum.loc[value, target_column])]
        remove_count = int(round(float(parameter["removed_fraction"]) * len(target_positions)))
        rng = np.random.default_rng(stable_seed(condition.dataset, condition.subtype, replicate_seed))
        remove_positions = rng.permutation(target_positions)[:remove_count]
        selected_set = set(clean_ids)
        candidates = [
            value
            for value in full_s.source_id.astype(str)
            if value not in selected_set and not bool(stratum.loc[value, target_column])
        ]
        candidates = [candidates[int(index)] for index in rng.permutation(len(candidates))]
        if remove_count > len(candidates):
            raise RuntimeError(
                f"Insufficient unique support-loss replacements: {condition.dataset}/{condition.subtype} "
                f"needs {remove_count}, has {len(candidates)}"
            )
        for position, replacement in zip(remove_positions, candidates[:remove_count], strict=True):
            output_ids[int(position)] = replacement
        if len(set(output_ids)) != len(output_ids):
            raise RuntimeError("Support-loss construction introduced duplicates")
    return output_ids, donor_ids, injected, duplicate, actual_unique


def _read_ids(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"source_id": str})
    if "source_id" not in frame or frame.empty or frame.source_id.isna().any() or frame.source_id.duplicated().any():
        raise ValueError(f"{path}: source_id column must be nonempty and unique")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="cohort-composition condition ID")
    parser.add_argument("--source-csv", type=Path, required=True, help="ordered replicate S IDs")
    parser.add_argument("--donor-csv", type=Path, required=True, help="ordered replicate D IDs")
    parser.add_argument("--pool-csv", type=Path, required=True, help="all eligible source-pool IDs")
    parser.add_argument("--strata-csv", type=Path, help="source_id,common_target,rare_target for support loss")
    parser.add_argument("--replicate-seed", type=int, default=41001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    condition = next((c for c in all_conditions() if c.condition_id == args.condition), None)
    if condition is None or condition.condition_kind != "cohort_composition":
        parser.error("condition must be a cohort-composition ID; use python -m benchmark.run --list")
    if args.output.exists():
        parser.error("output already exists")
    source, donor, full_s = (_read_ids(path) for path in (args.source_csv, args.donor_csv, args.pool_csv))
    if not set(source.source_id).issubset(set(full_s.source_id)) or set(source.source_id) & set(donor.source_id):
        parser.error("source IDs must be in the source pool and disjoint from donor IDs")
    if condition.family == "memorisation_injection" and round(float(condition.parameters["injected_fraction"]) * len(source)) > len(donor):
        parser.error("not enough donor IDs for the requested injection fraction")
    strata = pd.DataFrame(columns=["source_id", "common_target", "rare_target"])
    if condition.family == "support_loss":
        if args.strata_csv is None:
            parser.error("--strata-csv is required for support loss")
        strata = _read_ids(args.strata_csv)
        if not {"common_target", "rare_target"}.issubset(strata.columns) or not set(full_s.source_id).issubset(set(strata.source_id)):
            parser.error("strata must include common_target and rare_target for all pool IDs")
        for column in ("common_target", "rare_target"):
            values = strata[column].astype(str).str.strip().str.lower()
            if not values.isin(["0", "1", "true", "false"]).all():
                parser.error(f"{column} must contain true/false or 1/0")
            strata[column] = values.isin(["1", "true"])
    ids, donor_ids, injected, duplicate, unique = _composition(condition, source, donor, full_s, strata, args.replicate_seed)
    result = pd.DataFrame({"source_id": ids, "clean_source_id": source.source_id.tolist(), "donor_id": donor_ids,
                           "true_injected": injected, "is_duplicate_position": duplicate, "actual_unique_fraction": unique})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    print(f"Wrote {len(result)} positions to {args.output}")


if __name__ == "__main__":
    main()
