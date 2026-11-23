"""Frozen-checkpoint hydraulic holdout, analytical and resolution audits."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch
from config import TERRAIN, SEEDS, make_experiment
from core.pinn_inference import load_pinn, predict_frames
from simulation.finite_volume import solve
from simulation.validation_suite import evaluate
from utils.provenance import provenance


def unseen_experiments():
    cases = []
    for terrain, (width, q0, roughness, slope) in TERRAIN.items():
        for multiplier in (0.85, 1.15):
            q = q0 * multiplier
            depth = (q * roughness / (width * math.sqrt(slope))) ** 0.6
            u = q / (width * depth)
            cases.append(
                replace(
                    make_experiment(terrain),
                    season=f"holdout_Q{multiplier}",
                    u=u,
                    diffusion=0.08 * u * depth,
                )
            )
    return cases


def analytic_case(nx, safety=0.8):
    # Exact homogeneous ADE eigenmode satisfying the same mixed boundaries.
    exp = make_experiment(
        length=10.0,
        width=5.0,
        u=0.2,
        diffusion=0.5,
        horizon=2.0,
        nx=nx,
        ny=nx // 2,
        n_times=2,
    )
    alpha = exp.u / (2 * exp.diffusion)
    k = brentq(
        lambda k: k * math.cos(k * exp.length) + alpha * math.sin(k * exp.length),
        math.pi / (2 * exp.length),
        math.pi / exp.length,
    )
    x, y = exp.grid()
    longitudinal = np.exp(alpha * x[:, None]) * np.sin(k * x[:, None])
    lateral = np.cos(2 * np.pi * y[None, :] / exp.width)
    initial = exp.initial_density * longitudinal * (1 + 0.3 * lateral)
    exact = (
        exp.initial_density
        * longitudinal
        * np.exp(
            -(exp.diffusion * k * k + exp.u * exp.u / (4 * exp.diffusion)) * exp.horizon
        )
        * (
            1
            + 0.3
            * np.exp(-exp.diffusion * (2 * np.pi / exp.width) ** 2 * exp.horizon)
            * lateral
        )
    )
    result = solve(exp, initial=initial, safety=safety)
    return dict(
        nx=nx,
        ny=nx // 2,
        safety=safety,
        relative_l2=float(
            np.linalg.norm(result["frames"][-1] - exact) / np.linalg.norm(exact)
        ),
        relative_mass_balance=result["relative_balance_error"],
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cases = unseen_experiments()
    checkpoints = [
        Path(f"outputs/checkpoints/v5/{v}_{s}_final.pt")
        for v in ("hybrid", "data_only")
        for s in SEEDS
    ]
    plan = dict(
        experiments=[asdict(e) for e in cases],
        seeds=list(SEEDS),
        checkpoint_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints
        },
        protocol="Frozen checkpoints, no retraining or selection on new cases. Eight unseen discharge multipliers, same IC/domain/horizon. Not unseen-IC or extrapolation validation.",
        analytical="C=exp(u*x/(2D))*sin(k*x)*exp(-(D*k^2+u^2/(4D))*t)*(1+0.3*cos(2*pi*y/W)*exp(-D*(2*pi/W)^2*t)); k*cos(kL)+u/(2D)*sin(kL)=0",
        provenance=provenance(),
    )
    (output / "protocol.json").write_text(json.dumps(plan, indent=2))
    analytical = [analytic_case(n) for n in (20, 40, 80, 160)]
    temporal = [analytic_case(80, s) for s in (0.8, 0.4, 0.2, 0.1)]
    (output / "analytical.json").write_text(
        json.dumps(
            dict(
                spatial=analytical,
                temporal=temporal,
                note="Analytical eigenmode is a solver benchmark with its own prescribed initial field; the trained PINN is not evaluated on this different IC. Fixed-grid temporal rows approach a spatial-error floor.",
            ),
            indent=2,
        )
    )
    print("Analytical benchmark", analytical, flush=True)
    exp = make_experiment()
    reference = []
    for nx, ny in ((400, 46), (800, 92), (1600, 184)):
        e = exp.refined(nx, ny)
        r = solve(e, times=[0, exp.horizon])
        ratio = nx // exp.nx
        field = r["frames"][-1].reshape(exp.nx, ratio, exp.ny, ratio).mean((1, 3))
        reference.append(field)
    model = load_pinn(checkpoints[0])
    pred = predict_frames(model, exp, times=[exp.horizon])[0]
    refinement = []
    for nx, field in zip((400, 800, 1600), reference):
        refinement.append(
            dict(
                nx=nx,
                final_relative_l2_vs_1600=float(
                    np.linalg.norm(field - reference[-1])
                    / np.linalg.norm(reference[-1])
                ),
                neural_final_relative_l2=float(
                    np.linalg.norm(pred - field) / np.linalg.norm(field)
                ),
            )
        )
    long = solve(replace(exp, horizon=600.0, n_times=121))
    (output / "reference_audit.json").write_text(
        json.dumps(
            dict(
                refinement=refinement,
                longer_horizon_reference=dict(
                    horizon_s=600,
                    relative_mass_balance=long["relative_balance_error"],
                    minimum_density=float(long["frames"].min()),
                ),
                neural_horizon="300 seconds only; 600-second reference does not establish neural long-horizon performance",
            ),
            indent=2,
        )
    )
    rows = []
    for path in checkpoints:
        model = load_pinn(path)
        for e in cases:
            r = evaluate(model, e, research=True)["metrics"]
            r.update(seed=model.metadata["seed"], variant=model.metadata["variant"])
            rows.append(r)
            print(
                r["variant"],
                r["seed"],
                e.terrain,
                e.season,
                "L2",
                r["relative_l2_spacetime"],
                "mass",
                r["mass_balance_relative_error"],
                "passed",
                r["accepted_numerically"],
                flush=True,
            )
        (output / "holdout_metrics.json").write_text(
            json.dumps(rows, indent=2, allow_nan=False)
        )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for p, h in plan["checkpoint_sha256"].items()
    )
    summary = {
        v: dict(
            cases=len(a),
            passed=sum(r["accepted_numerically"] for r in a),
            max_l2=max(r["relative_l2_spacetime"] for r in a),
            max_mass_error=max(r["mass_balance_relative_error"] for r in a),
        )
        for v in ("hybrid", "data_only")
        if (a := [r for r in rows if r["variant"] == v])
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
