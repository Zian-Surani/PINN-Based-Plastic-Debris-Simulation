"""Inference refuses legacy checkpoints, changed physics and extrapolation."""

from pathlib import Path
import numpy as np
import torch
from config import SCHEMA_VERSION
from core.pinn_model import PINN
from core.training import inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs/checkpoints/v5/hybrid_2026_final.pt"


def load_pinn(path=DEFAULT_CHECKPOINT):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Train a schema-{SCHEMA_VERSION} checkpoint first: {path}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("metadata", {}).get("schema") != SCHEMA_VERSION
    ):
        raise ValueError(
            "Legacy/incompatible checkpoint: retraining is required; no fallback permitted."
        )
    meta = payload["metadata"]
    model = PINN(meta["width"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.metadata = meta
    return model


def validate_experiment(model, exp, research=False):
    if research:
        # Explicit audit-only interpolation; production inference remains guarded.
        candidates = [
            e for e in model.metadata["experiments"] if e["terrain"] == exp.terrain
        ]
        fixed = ("length", "width", "horizon", "initial_density", "boundary", "initial")
        if not candidates or any(getattr(exp, k) != candidates[0][k] for k in fixed):
            raise ValueError(
                "Research audit permits changed hydraulics only, not changed IC/domain/horizon"
            )
        if any(
            not min(e[k] for e in candidates)
            <= getattr(exp, k)
            <= max(e[k] for e in candidates)
            for k in ("u", "diffusion")
        ):
            raise ValueError(
                "Research hydraulics must remain inside trained parameter bounds"
            )
        return
    if exp.signature() not in model.metadata["signatures"]:
        raise ValueError(
            "Experiment physics/domain do not match a supported training configuration."
        )


def predict_points(model, exp, x, y, t, *, research=False):
    validate_experiment(model, exp, research=research)
    x, y, t = np.broadcast_arrays(x, y, t)
    for a, upper in ((x, exp.length), (y, exp.width), (t, exp.horizon)):
        if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > upper + 1e-10):
            raise ValueError("Prediction outside trained physical domain/time range")
    z = inputs(x, y, t, exp)
    chunks = []
    with torch.no_grad():
        for part in np.array_split(z, max(1, int(np.ceil(len(z) / 8192)))):
            chunks.append(model(torch.from_numpy(part)).numpy().ravel())
    return np.concatenate(chunks).reshape(x.shape) * exp.initial_density


def predict_frames(model, exp, times=None, *, research=False):
    times = exp.times() if times is None else np.asarray(times, dtype=float)
    x, y = exp.grid()
    tt, xx, yy = np.meshgrid(times, x, y, indexing="ij")
    return predict_points(model, exp, xx, yy, tt, research=research)
