import numpy as np


def compute_accumulation(frames, times):
    """Time-integrated exposure (kg s/m²), not retained or deposited mass."""
    frames = np.asarray(frames)
    times = np.asarray(times)
    if (
        frames.ndim != 3
        or len(frames) != len(times)
        or len(times) < 2
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Need matching fields and increasing times")
    if not np.isfinite(frames).all() or not np.isfinite(times).all():
        raise ValueError("Nonfinite values")
    return (0.5 * (frames[1:] + frames[:-1]) * np.diff(times)[:, None, None]).sum(
        axis=0
    )
