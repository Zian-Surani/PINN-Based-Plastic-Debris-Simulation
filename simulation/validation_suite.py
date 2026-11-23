"""Independent numerical comparison, SI mass budgets and measured runtime."""

from time import perf_counter
import numpy as np
import torch
from config import initial_field
from core.pinn_model import derivative, residual
from core.training import inputs
from core.pinn_inference import predict_frames, validate_experiment
from simulation.finite_volume import solve


def compute_error_timeseries(pred, reference):
    p = np.asarray(pred, dtype=float)
    r = np.asarray(reference, dtype=float)
    if (
        p.shape != r.shape
        or p.ndim != 3
        or not np.isfinite(p).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("Fields must match and be finite (time,x,y) arrays")
    diff = p - r
    mse = np.mean(diff**2, axis=(1, 2))
    den = np.linalg.norm(r.reshape(len(r), -1), axis=1)
    num = np.linalg.norm(diff.reshape(len(r), -1), axis=1)
    rel = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "relative_l2": rel,
        "rmse_final": float(np.sqrt(mse[-1])),
        "relative_l2_final": float(rel[-1]),
        "relative_l2_spacetime": float(np.linalg.norm(diff) / np.linalg.norm(r))
        if np.linalg.norm(r) > 0
        else None,
    }


def boundary_flux(model, exp, times, *, research=False):
    """Actual learned total flux, including any soft-boundary violations."""
    validate_experiment(model, exp, research=research)
    y = (np.arange(64) + 0.5) * exp.width / 64
    tt, yy = np.meshgrid(times, y, indexing="ij")
    net = np.zeros(len(times))
    inlet_errors = []
    outlet_grad = []
    for side in (0.0, exp.length):
        z = torch.tensor(
            inputs(np.full_like(yy, side), yy, tt, exp), requires_grad=True
        )
        c = model(z)
        g = derivative(c, z)
        flux = (
            exp.u * c - exp.diffusion * g[:, 0:1] / exp.length
        ).detach().numpy().reshape(len(times), 64) * exp.initial_density
        net += (1 if side else -1) * flux.mean(1) * exp.width
        if side == 0:
            inlet_errors = flux
        else:
            outlet_grad = g[:, 0].detach().numpy()
    return net, float(np.max(np.abs(inlet_errors))), float(np.max(np.abs(outlet_grad)))


def compute_mass_metrics(frames, exp, times, outward_flux):
    frames = np.asarray(frames, dtype=float)
    times = np.asarray(times, dtype=float)
    flux = np.asarray(outward_flux, dtype=float)
    if (
        frames.shape != (len(times), exp.nx, exp.ny)
        or flux.shape != times.shape
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Mass budget inputs have incompatible grid/time axes")
    if not np.isfinite(frames).all() or not np.isfinite(flux).all():
        raise ValueError("Nonfinite mass budget")
    mass = frames.sum(axis=(1, 2)) * exp.dx * exp.dy
    escaped = np.r_[0.0, np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(times))]
    prescribed = float(initial_field(exp).sum() * exp.dx * exp.dy)
    error = mass + escaped - prescribed
    return {
        "mass_kg": mass,
        "integrated_outward_flux_kg": escaped,
        "balance_error_kg": error,
        "max_relative_balance_error": float(np.max(np.abs(error)) / prescribed),
        "relative_initial_mass_error": float(abs(mass[0] - prescribed) / prescribed),
        "maximum_mass_increase_relative": float(
            max(0.0, np.diff(mass).max()) / prescribed
        ),
        "mass_monotonicity_definition": "maximum positive increment in domain-integrated mass between reported times, divided by prescribed initial mass; zero means nonincreasing at those times, not spatial pointwise monotonicity",
        "quadrature": "cell midpoint space; trapezoidal output-time flux; not an exact network constraint",
    }


def evaluate(model, exp, *, research=False):
    validate_experiment(model, exp, research=research)
    pred = predict_frames(model, exp, research=research)
    fine = solve(exp.refined(4 * exp.nx, 4 * exp.ny))
    f = fine["frames"]
    ref = f.reshape(len(f), exp.nx, 4, exp.ny, 4).mean(axis=(2, 4))
    err = compute_error_timeseries(pred, ref)
    flux, inlet, outlet = boundary_flux(model, exp, exp.times(), research=research)
    mass = compute_mass_metrics(pred, exp, exp.times(), flux)
    ic = initial_field(exp)
    initial_l2 = float(np.linalg.norm(pred[0] - ic) / np.linalg.norm(ic))
    rng = np.random.default_rng(91031)
    z = rng.random((2048, 6)).astype("float32")
    z[:, 3:] = exp.coefficients
    pde_rms = float(
        torch.sqrt(residual(model, torch.from_numpy(z)).square().mean()).detach()
    )
    metrics = {
        "evaluation_scope": "unseen hydraulic interpolation audit"
        if research
        else "supported training configuration",
        "terrain": exp.terrain,
        "season": exp.season,
        "signature": exp.signature(),
        "minimum_density_kg_m2": float(pred.min()),
        "relative_l2_spacetime": err["relative_l2_spacetime"],
        "rmse_final_kg_m2": err["rmse_final"],
        "initial_relative_l2": initial_l2,
        "mass_balance_relative_error": mass["max_relative_balance_error"],
        "initial_mass_relative_error": mass["relative_initial_mass_error"],
        "maximum_mass_increase_relative": mass["maximum_mass_increase_relative"],
        "pde_residual_rms_dimensionless": pde_rms,
        "inlet_flux_max_kg_m_s": inlet,
        "outlet_gradient_max_dimensionless": outlet,
        "fvm_relative_balance_error": fine["relative_balance_error"],
        "fvm_combined_cfl": fine["combined_cfl_max"],
    }
    # Declared numerical acceptance criteria; passing is not field validation.
    metrics["acceptance"] = {
        "nonnegative": bool(pred.min() >= 0),
        "accuracy_l2_under_10pct": err["relative_l2_spacetime"] < 0.10,
        "initial_l2_under_5pct": initial_l2 < 0.05,
        "budget_error_under_2pct": mass["max_relative_balance_error"] < 0.02,
        "reference_conservative": fine["relative_balance_error"] < 1e-10,
    }
    metrics["accepted_numerically"] = all(metrics["acceptance"].values())
    return {
        "metrics": metrics,
        "prediction": pred,
        "reference": ref,
        "error": err,
        "mass": mass,
    }


def benchmark_runtime(model, exp, repeats=10):
    if repeats < 2:
        raise ValueError("At least two timing repetitions required")
    predict_frames(model, exp)
    solve(exp)  # untimed warm-up
    records = []
    for i in range(repeats):
        row = {}
        methods = [
            ("pinn", lambda: predict_frames(model, exp)),
            ("finite_volume", lambda: solve(exp)),
        ]
        if i % 2:
            methods.reverse()
        for name, fn in methods:
            start = perf_counter()
            fn()
            row[name] = perf_counter() - start
        records.append(row)
    summary = {
        name: {
            "mean_s": float(np.mean([r[name] for r in records])),
            "sample_sd_s": float(np.std([r[name] for r in records], ddof=1)),
            "median_s": float(np.median([r[name] for r in records])),
        }
        for name in records[0]
    }
    return {
        "repeats": repeats,
        "warmup_pairs": 1,
        "ordering": "alternating",
        "raw_seconds": records,
        "summary": summary,
        "scope": "CPU, model already loaded; both evaluate same cell grid and output times; training excluded",
    }


def grid_refinement(exp):
    runs = [
        solve(exp.refined(exp.nx * k, exp.ny * k), times=[0, exp.horizon])
        for k in (1, 2, 4)
    ]
    fine = runs[-1]["frames"][-1]
    rows = []
    for k, r in zip((1, 2, 4), runs):
        ratio = 4 // k
        ref = fine.reshape(exp.nx * k, ratio, exp.ny * k, ratio).mean(axis=(1, 3))
        field = r["frames"][-1]
        rows.append(
            {
                "nx": exp.nx * k,
                "ny": exp.ny * k,
                "rmse_vs_restricted_finest": float(
                    np.sqrt(np.mean((field - ref) ** 2))
                ),
                "mass_balance_relative_error": r["relative_balance_error"],
            }
        )
    return {
        "rows": rows,
        "note": "Finest self-error is zero by construction, not proof of zero discretization error. No convergence order inferred from two nonzero comparisons.",
    }
