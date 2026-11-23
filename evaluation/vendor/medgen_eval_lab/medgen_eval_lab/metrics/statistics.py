"""Shared statistics engine for metric uncertainty and significance.

This module consolidates the bootstrap-confidence-interval logic that was
previously duplicated in ``metrics/distribution/kernels.py`` and
``metrics/radiomics/frd.py`` into one deterministic, pure-Python implementation
and adds null-baseline significance testing (a bootstrap/permutation p-value vs a
stated null) plus a standardized effect size.

Design rules (see the locked decisions in the refinement plan):

* Bootstrap CIs are appropriate for **any** headline scalar.
* A p-value is only ever reported against an explicit null baseline that exists
  for the metric (e.g. a real-vs-real or train-vs-val split). Metrics without a
  meaningful null get a CI only -- this module never fabricates a p-value.
* The effect size is standardized: how many null standard deviations the observed
  value sits from the null mean. No absolute quality thresholds are invented.

Everything is stdlib-only (``math`` + ``random``) so it runs in the CPU-light
lane with no NumPy dependency, and is seeded for reproducibility.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from statistics import NormalDist
from typing import Any

from medgen_eval_lab.schemas.results.metric_result import (
    ConfidenceInterval,
    SignificanceResult,
)

Sample = Any
Group = Sequence[Sample]
Alternative = str

__all__ = [
    "Alternative",
    "Group",
    "Sample",
    "bootstrap_confidence_intervals",
    "bootstrap_statistics",
    "null_baseline_significance",
    "percentile_confidence_interval",
    "permutation_test",
    "quantile",
    "seeded_subset",
    "standardized_mean_difference",
    "wilson_proportion_interval",
]


def _stable_sample_key(value: Sample) -> str:
    """Return an order-independent key for seeded sampling/permutation inputs."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def seeded_subset(group: Group, size: int, *, seed: int) -> list[Sample]:
    """Select a deterministic randomized subset independent of caller row order.

    Sorting by a canonical representation before shuffling makes a fixed seed
    reproducible even when discovery returns the same samples in another order.
    """

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("subset size must be a non-negative integer")
    if size > len(group):
        raise ValueError("subset size cannot exceed group size")
    population = sorted(group, key=_stable_sample_key)
    random.Random(seed).shuffle(population)
    return population[:size]


def wilson_proportion_interval(
    successes: int,
    total: int,
    *,
    level: float = 0.95,
) -> ConfidenceInterval | None:
    """Wilson score confidence interval for a binomial proportion ``successes/total``.

    Wilson is the right CI for a rate (it stays inside [0, 1] and behaves well at
    small N and extreme proportions), so headline metrics that are proportions --
    e.g. memorisation copy rates -- get an honest interval without resampling.
    Returns ``None`` for an empty denominator.
    """

    if total <= 0 or successes < 0 or successes > total:
        return None
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total))
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return ConfidenceInterval(
        low=low,
        high=high,
        level=level,
        method=f"wilson_score/n={total}",
    )


def _resampled_groups(
    groups: Sequence[Group],
    *,
    repetitions: int,
    seed: int,
) -> Iterator[list[list[Sample]]]:
    """Yield ``repetitions`` with-replacement resamples of ``groups`` (sizes kept).

    A single RNG seeded by ``seed`` drives all draws, with groups resampled in
    order, so callers that consume statistics in the same order get identical
    deterministic results.
    """

    rng = random.Random(seed)
    for _ in range(repetitions):
        yield [
            [group[rng.randrange(len(group))] for _ in range(len(group))]
            for group in groups
        ]


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile over an already-sorted sequence.

    Matches NumPy's default ``linear`` method and the historical
    ``kernels._quantile`` behaviour so existing CIs are unchanged.
    """

    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bootstrap_statistics(
    groups: Sequence[Group],
    statistic: Callable[..., float],
    *,
    repetitions: int,
    seed: int,
) -> list[float]:
    """Resample each group with replacement and recompute ``statistic``.

    Each group is resampled to its own size with a single RNG seeded by ``seed``
    (groups resampled in order), so results are deterministic given the same
    inputs. ``statistic`` is called with the resampled groups as positional
    arguments. Repetitions that raise ``ValueError`` are skipped; the caller
    decides how to treat non-finite values.
    """

    if repetitions <= 0:
        return []
    values: list[float] = []
    for resampled in _resampled_groups(groups, repetitions=repetitions, seed=seed):
        try:
            values.append(float(statistic(*resampled)))
        except ValueError:
            continue
    return values


def bootstrap_confidence_intervals(
    groups: Sequence[Group],
    statistics_fn: Callable[..., Mapping[str, float]],
    *,
    repetitions: int,
    seed: int,
    level: float,
) -> dict[str, ConfidenceInterval]:
    """Percentile bootstrap CIs for several named statistics in one resampling pass.

    ``statistics_fn`` is called with the resampled groups and returns a mapping of
    statistic name to value; each name accumulates its own bootstrap sample and a
    CI is emitted per name (names with no finite values are dropped). Computing
    every headline scalar from a single resample avoids paying the O(N^2) cost
    once per scalar.
    """

    if repetitions <= 0:
        return {}
    collected: dict[str, list[float]] = {}
    for resampled in _resampled_groups(groups, repetitions=repetitions, seed=seed):
        try:
            sample = statistics_fn(*resampled)
        except ValueError:
            continue
        for key, value in sample.items():
            collected.setdefault(key, []).append(float(value))
    intervals: dict[str, ConfidenceInterval] = {}
    for key, values in collected.items():
        interval = percentile_confidence_interval(values, level=level)
        if interval is not None:
            intervals[key] = interval
    return intervals


def percentile_confidence_interval(
    values: Sequence[float],
    *,
    level: float,
    method: str | None = None,
) -> ConfidenceInterval | None:
    """Percentile bootstrap CI from a sample of statistic values.

    Non-finite values are dropped. Returns ``None`` when no finite values remain
    so callers can simply skip emitting a CI.
    """

    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return None
    sorted_values = sorted(finite)
    alpha = 1.0 - level
    low = quantile(sorted_values, alpha / 2.0)
    high = quantile(sorted_values, 1.0 - alpha / 2.0)
    return ConfidenceInterval(
        low=low,
        high=high,
        level=level,
        method=method or f"bootstrap_percentile/{len(finite)}_repetitions",
    )


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Cohen's d: mean difference divided by the pooled standard deviation.

    Returns ``0.0`` when the pooled standard deviation is zero (degenerate but
    well-defined: identical, constant groups have no standardized separation).
    """

    left_finite = [float(value) for value in left if math.isfinite(value)]
    right_finite = [float(value) for value in right if math.isfinite(value)]
    if len(left_finite) < 2 or len(right_finite) < 2:
        raise ValueError("standardized mean difference needs >=2 finite values per group")
    mean_left = sum(left_finite) / len(left_finite)
    mean_right = sum(right_finite) / len(right_finite)
    var_left = sum((value - mean_left) ** 2 for value in left_finite) / (len(left_finite) - 1)
    var_right = sum((value - mean_right) ** 2 for value in right_finite) / (len(right_finite) - 1)
    n_left = len(left_finite)
    n_right = len(right_finite)
    pooled_n = n_left + n_right - 2
    pooled_var = ((n_left - 1) * var_left + (n_right - 1) * var_right) / pooled_n
    pooled_std = math.sqrt(pooled_var)
    if pooled_std <= 0.0:
        return 0.0
    return (mean_left - mean_right) / pooled_std


def null_baseline_significance(
    observed: float,
    null_values: Sequence[float],
    *,
    null_baseline: str,
    alternative: Alternative = "two-sided",
    method: str | None = None,
) -> SignificanceResult | None:
    """Bootstrap/permutation p-value + standardized effect size vs a null sample.

    ``null_values`` is the distribution of the same statistic under the null
    (e.g. real-vs-real resamples). The p-value uses add-one smoothing so it is
    never exactly zero, and the effect size is standardized as
    ``(observed - null_mean) / null_std``. Returns ``None`` when the inputs are
    too degenerate to support an honest test.
    """

    if not math.isfinite(observed):
        return None
    finite = [float(value) for value in null_values if math.isfinite(value)]
    if len(finite) < 2:
        return None
    count = len(finite)
    mean = sum(finite) / count
    variance = sum((value - mean) ** 2 for value in finite) / (count - 1)
    std = math.sqrt(variance)
    if alternative == "greater":
        extreme = sum(1 for value in finite if value >= observed)
    elif alternative == "less":
        extreme = sum(1 for value in finite if value <= observed)
    elif alternative == "two-sided":
        deviation = abs(observed - mean)
        extreme = sum(1 for value in finite if abs(value - mean) >= deviation)
    else:
        raise ValueError("alternative must be one of: two-sided, greater, less")
    p_value = min(1.0, (extreme + 1) / (count + 1))
    effect_size = (observed - mean) / std if std > 0.0 else 0.0
    return SignificanceResult(
        p_value=p_value,
        null_baseline=null_baseline,
        method=method or f"bootstrap_null/{count}_repetitions",
        alternative=alternative,  # type: ignore[arg-type]
        effect_size=effect_size,
        repetitions=count,
        observed=observed,
        null_mean=mean,
        null_std=std,
    )


def permutation_test(
    left: Group,
    right: Group,
    statistic: Callable[[Group, Group], float],
    *,
    repetitions: int,
    seed: int,
    null_baseline: str,
    alternative: Alternative = "two-sided",
    method: str | None = None,
) -> SignificanceResult | None:
    """Two-sample permutation test for a statistic of two groups.

    The observed statistic is compared against its distribution under random
    relabelings of the pooled samples. Delegates to
    :func:`null_baseline_significance` for the p-value and effect size. Returns
    ``None`` when the observed statistic is undefined or too few permutations
    succeed.
    """

    if repetitions <= 0:
        return None
    try:
        observed = float(statistic(left, right))
    except ValueError:
        return None
    if not math.isfinite(observed):
        return None
    # Canonicalize before shuffling so permutation draws do not depend on the
    # incidental order in which equivalent cohorts were discovered.
    pooled = sorted([*left, *right], key=_stable_sample_key)
    if len(pooled) < 2:
        return None
    n_left = len(left)
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(repetitions):
        rng.shuffle(pooled)
        try:
            value = float(statistic(pooled[:n_left], pooled[n_left:]))
        except ValueError:
            continue
        if math.isfinite(value):
            null_values.append(value)
    return null_baseline_significance(
        observed,
        null_values,
        null_baseline=null_baseline,
        alternative=alternative,
        method=method or f"permutation/{len(null_values)}_repetitions",
    )
