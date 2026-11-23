"""Conservative explicit finite-volume reference: Ct=-div(u C-D grad C).

Zero tracer concentration at inlet, zero flux at walls; zero diffusive
gradient and advective outflow at outlet. Upstream diffusive export is
included in the mass budget. No clipping or mass renormalisation.
"""

import numpy as np
from config import initial_field


def timestep_limit(exp, safety=0.8):
    if not 0 < safety <= 1:
        raise ValueError("safety must lie in (0,1]")
    rate = exp.u / exp.dx + exp.diffusion * (3 / exp.dx**2 + 2 / exp.dy**2)
    return np.inf if rate == 0 else safety / rate


def fluxes(c, exp):
    if c.shape != (exp.nx, exp.ny) or not np.isfinite(c).all():
        raise ValueError("Invalid concentration field")
    jx = np.zeros((exp.nx + 1, exp.ny))
    jy = np.zeros((exp.nx, exp.ny + 1))
    jx[1:-1] = exp.u * c[:-1] - exp.diffusion * np.diff(c, axis=0) / exp.dx
    jx[0] = -2 * exp.diffusion * c[0] / exp.dx  # C=0 at inlet face, half-cell distance
    jx[-1] = exp.u * c[-1]
    jy[:, 1:-1] = -exp.diffusion * np.diff(c, axis=1) / exp.dy
    return jx, jy


def solve(exp, times=None, initial=None, safety=0.8):
    times = np.asarray(exp.times() if times is None else times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or times[-1] > exp.horizon + 1e-10
    ):
        raise ValueError(
            "Times must start at zero, strictly increase, and stay within the horizon"
        )
    c = (
        initial_field(exp)
        if initial is None
        else np.array(initial, dtype=float, copy=True)
    )
    if c.shape != (exp.nx, exp.ny) or not np.isfinite(c).all() or c.min() < 0:
        raise ValueError(
            "Initial field must be finite, nonnegative, and match the grid"
        )
    frames = [c.copy()]
    escaped = [0.0]
    total_out = 0.0
    largest_cfl = 0.0
    dtmax = timestep_limit(exp, safety)
    for interval in np.diff(times):
        n = 1 if np.isinf(dtmax) else max(1, int(np.ceil(interval / dtmax)))
        dt = interval / n
        rate = exp.u / exp.dx + exp.diffusion * (3 / exp.dx**2 + 2 / exp.dy**2)
        largest_cfl = max(largest_cfl, dt * rate)
        for _ in range(n):
            jx, jy = fluxes(c, exp)
            total_out += dt * (jx[-1].sum() - jx[0].sum()) * exp.dy
            c = c - dt * (np.diff(jx, axis=0) / exp.dx + np.diff(jy, axis=1) / exp.dy)
            if not np.isfinite(c).all() or c.min() < -1e-12 * exp.initial_density:
                raise FloatingPointError("FVM positivity/finiteness failure")
        frames.append(c.copy())
        escaped.append(total_out)
    frames = np.asarray(frames)
    mass = frames.sum(axis=(1, 2)) * exp.dx * exp.dy
    escaped = np.asarray(escaped)
    balance = mass + escaped - mass[0]
    return {
        "frames": frames,
        "times": times,
        "mass_kg": mass,
        "escaped_mass_kg": escaped,
        "balance_error_kg": balance,
        "relative_balance_error": float(np.max(np.abs(balance)) / max(mass[0], 1e-30)),
        "combined_cfl_max": largest_cfl,
        "minimum_density": float(frames.min()),
        "method": "upwind finite volume with central diffusion",
    }
