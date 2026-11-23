"""FRDv1 aggregation of real and generated feature matrices.

Uses frd-score by Richard Osuala, Nick Konz and Preeti Verma,
https://github.com/RichardObi/frd-score, revision
bf05c71fc44e461ab123bd273e58390b90f9cf54, under Apache-2.0; see
LICENSE-APACHE-2.0 in this directory. This study wrapper filters invalid rows
and columns and reports the paper-log formulation with diagnostic counts.
"""

import math
import time
import numpy as np
from frd_score import frd


def prepare_v1_rows(features: np.ndarray) -> tuple[np.ndarray, int]:
    nan_rows = np.isnan(features).any(axis=1)
    retained = np.asarray(features[~nan_rows], dtype=np.float32)
    return retained, int(nan_rows.sum())


def compute_pair(
    real: np.ndarray,
    generated: np.ndarray,
    feature_names: list[str],
) -> dict[str, object]:
    started = time.perf_counter()
    real_rows, real_dropped = prepare_v1_rows(real)
    generated_rows, generated_dropped = prepare_v1_rows(generated)
    if real_rows.shape[0] < 2 or generated_rows.shape[0] < 2:
        raise RuntimeError("fewer than two retained images after v1 NaN-row filtering")

    real_normalized = frd.z_score_normalize(
        real_rows,
        new_min=0,
        new_max=1,
        replace_nan=False,
        feature_names=feature_names,
        base_distribution=real_rows,
        handle_std_zero=False,
    )
    generated_normalized = frd.z_score_normalize(
        generated_rows,
        new_min=0,
        new_max=1,
        replace_nan=False,
        feature_names=feature_names,
        base_distribution=real_rows,
        handle_std_zero=False,
    )
    invalid_columns = (
        np.isnan(real_normalized).any(axis=0)
        | np.isinf(real_normalized).any(axis=0)
        | np.isnan(generated_normalized).any(axis=0)
        | np.isinf(generated_normalized).any(axis=0)
    )
    retained_columns = ~invalid_columns
    real_normalized = real_normalized[:, retained_columns]
    generated_normalized = generated_normalized[:, retained_columns]
    if real_normalized.shape[1] == 0:
        raise RuntimeError("all feature columns were removed after D1 normalization")

    mu_real = np.mean(real_normalized, axis=0)
    mu_generated = np.mean(generated_normalized, axis=0)
    sigma_real = np.cov(real_normalized, rowvar=False)
    sigma_generated = np.cov(generated_normalized, rowvar=False)
    distance_squared = float(
        frd.calculate_frechet_distance(
            mu_real, sigma_real, mu_generated, sigma_generated
        )
    )
    if not math.isfinite(distance_squared) or distance_squared <= 0:
        raise RuntimeError(f"invalid squared Frechet distance: {distance_squared}")
    return {
        "real_retained_count": int(real_rows.shape[0]),
        "real_nan_row_drop_count": real_dropped,
        "generated_retained_count": int(generated_rows.shape[0]),
        "generated_nan_row_drop_count": generated_dropped,
        "raw_feature_count": int(real.shape[1]),
        "retained_feature_count": int(retained_columns.sum()),
        "dropped_nonfinite_normalized_feature_count": int(invalid_columns.sum()),
        "frechet_distance_squared": distance_squared,
        "frd_paper_log_distance": 0.5 * math.log(distance_squared),
        "frd_cli_default_log_squared_distance": math.log(distance_squared),
        "aggregation_seconds": time.perf_counter() - started,
    }
