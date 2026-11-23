"""Descriptive chronological residual intervals. No conformal guarantee."""

import numpy as np


def residual_interval_audit(prediction, reference, level=0.9, calibration_fraction=0.5):
    p = np.asarray(prediction)
    r = np.asarray(reference)
    if (
        p.shape != r.shape
        or p.ndim != 3
        or len(p) < 4
        or not np.isfinite(p).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError(
            "Need matching finite (time,x,y) arrays and at least four times"
        )
    if not 0 < level < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("Fractions must lie strictly between zero and one")
    cut = int(len(p) * calibration_fraction)
    if cut < 2 or len(p) - cut < 2:
        raise ValueError("Need at least two calibration and evaluation times")
    errors = np.abs(p - r)
    q = np.quantile(errors[:cut], level, axis=0)
    coverage = float(np.mean(errors[cut:] <= q))
    return {
        "nominal_level": level,
        "empirical_coverage": coverage,
        "coverage_gap": coverage - level,
        "nominal_coverage_met_on_evaluation": bool(coverage >= level),
        "guaranteed_coverage": False,
        "half_width": q,
        "calibration_time_indices": list(range(cut)),
        "test_time_indices": list(range(cut, len(p))),
        "interpretation": "Descriptive residual intervals against numerical reference; dependent chronological samples do not establish exchangeability or distribution-free coverage.",
    }


def conformal_uq_from_reference(*args, **kwargs):
    raise RuntimeError(
        "Use residual_interval_audit. The old routine incorrectly implied conformal guarantees for dependent time slices."
    )
