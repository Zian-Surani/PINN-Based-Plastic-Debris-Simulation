"""Reproducible hybrid training. Numerical labels and physics losses are explicit."""

from dataclasses import asdict
from pathlib import Path
import hashlib, json, platform, random, time
import numpy as np
import torch
from config import all_experiments, initial_normalized, SCHEMA_VERSION
from core.pinn_model import PINN, derivative, residual
from simulation.finite_volume import solve


def inputs(x, y, t, exp):
    x, y, t = np.broadcast_arrays(x, y, t)
    return np.column_stack(
        (
            x.ravel() / exp.length,
            y.ravel() / exp.width,
            t.ravel() / exp.horizon,
            np.tile(exp.coefficients, (x.size, 1)),
        )
    ).astype("float32")


def build_dataset(experiments):
    features = []
    targets = []
    label_rng = np.random.default_rng(71713)
    for exp in experiments:
        coarse = exp.refined(400, 48)
        result = solve(coarse)
        x, y = coarse.grid()
        flat = label_rng.choice(result["frames"].size, size=60000, replace=False)
        it, ix, iy = np.unravel_index(flat, result["frames"].shape)
        features.append(inputs(x[ix], y[iy], coarse.times()[it], coarse))
        targets.append(
            (result["frames"][it, ix, iy] / exp.initial_density)
            .reshape(-1, 1)
            .astype("float32")
        )
    return np.concatenate(features), np.concatenate(targets)


def sample(rng, n, experiments):
    z = rng.random((n, 6)).astype("float32")
    idx = rng.integers(len(experiments), size=n)
    z[:, 3:] = np.asarray([experiments[i].coefficients for i in idx])
    return z


def physics_losses(model, z, ic, bc, quad):
    lpde = residual(model, z).square().mean()
    target = torch.tensor(
        initial_normalized(ic[:, 0].numpy(), ic[:, 1].numpy()), dtype=torch.float32
    ).reshape(-1, 1)
    lic = (model(ic) - target).square().mean()
    bc = bc.detach().requires_grad_(True)
    c = model(bc)
    g = derivative(c, bc)
    n = len(bc) // 2
    inlet = c[:n]
    lbc = inlet.square().mean() + g[n:, 0:1].square().mean()
    # Midpoint quadrature of the integral conservation law at 4 scenario/time pairs.
    quad = quad.detach().requires_grad_(True)
    cq = model(quad)
    gq = derivative(cq, quad)
    q = quad.reshape(-1, 1024, 6)
    nq = len(q)
    out = q[:, :16].clone()
    out[:, :, 0] = 1.0
    out[:, :, 1] = (torch.arange(16) + 0.5) / 16
    outc = model(out.reshape(-1, 6)).reshape(nq, 16)
    left = out.detach().clone()
    left[:, :, 0] = 0
    left = left.reshape(-1, 6).requires_grad_(True)
    leftgrad = derivative(model(left), left)[:, 0].reshape(nq, 16)
    mass = (
        (
            gq[:, 2].reshape(nq, 1024).mean(1)
            + q[:, 0, 3] * outc.mean(1)
            + q[:, 0, 4] * leftgrad.mean(1)
        )
        .square()
        .mean()
    )
    return lpde, lic, lbc, mass


def train_checkpoint(output, seed=2026, iterations=6000, variant="hybrid", threads=2, pde_weight=4.0):
    if variant not in ("hybrid", "data_only"):
        raise ValueError("Unknown training variant")
    if not np.isfinite(pde_weight) or pde_weight < 0:
        raise ValueError("PDE weight must be finite and nonnegative")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    experiments = all_experiments()
    print("Generating numerical training labels...", flush=True)
    X, Y = build_dataset(experiments)
    split_rng = np.random.default_rng(seed + 100000)
    order = split_rng.permutation(len(X))
    cut = int(0.9 * len(X))
    train_ids = order[:cut]
    val_ids = order[cut : cut + 8192]
    xv = torch.from_numpy(X[val_ids])
    yv = torch.from_numpy(Y[val_ids])
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    start = time.perf_counter()
    qx, qy = np.meshgrid(
        (np.arange(64) + 0.5) / 64, (np.arange(16) + 0.5) / 16, indexing="ij"
    )
    for step in range(1, iterations + 1):
        ids = rng.choice(train_ids, size=2048, replace=True)
        z = sample(rng, 512, experiments)
        ic = sample(rng, 512, experiments)
        ic[:, 2] = 0
        bc = sample(rng, 256, experiments)
        bc[:128, 0] = 0
        bc[128:, 0] = 1
        pairs = sample(rng, 4, experiments)
        quad = np.repeat(pairs, 1024, axis=0)
        quad[:, 0] = np.tile(qx.ravel(), 4)
        quad[:, 1] = np.tile(qy.ravel(), 4)
        opt.zero_grad(set_to_none=True)
        pred = model(torch.from_numpy(X[ids]))
        target = torch.from_numpy(Y[ids])
        ldata = (pred - target).square().mean() + 0.0001 * (
            torch.log(pred + 1e-4) - torch.log(target + 1e-4)
        ).square().mean()
        ramp = min(1.0, max(0.0, (step - 200) / 500))
        if variant == "hybrid":
            lpde, lic, lbc, lmass = physics_losses(
                model, *[torch.from_numpy(a) for a in (z, ic, bc, quad)]
            )
            loss = 10 * ldata + 20 * lic + ramp * (pde_weight * lpde + lbc + 20 * lmass)
        else:
            lpde = lic = lbc = lmass = torch.tensor(0.0)
            loss = 10 * ldata
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        for group in opt.param_groups:
            group["lr"] = 0.001 * (
                0.15 + 0.85 * 0.5 * (1 + np.cos(np.pi * step / iterations))
            )
        if step % 100 == 0 or step == iterations:
            with torch.no_grad():
                val = float((model(xv) - yv).square().mean())
            row = dict(
                step=step,
                total=float(loss.detach()),
                data=float(ldata.detach()),
                pde=float(lpde.detach()),
                ic=float(lic.detach()),
                bc=float(lbc.detach()),
                mass=float(lmass.detach()),
                validation_mse=val,
                physics_ramp=ramp,
            )
            history.append(row)
            print(
                f"{variant} seed={seed} step={step}/{iterations} val={val:.6g} loss={row['total']:.5g}",
                flush=True,
            )
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    metadata = {
        "schema": SCHEMA_VERSION,
        "variant": variant,
        "seed": seed,
        "iterations": iterations,
        "selected_step": iterations,
        "selection": "final fixed-budget iterate; reported reference never used for checkpoint selection",
        "training_seconds": time.perf_counter() - start,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "lr_schedule": "cosine decay to 0.00015",
        "weights": {"data": 10.0, "ic": 20.0, "pde": float(pde_weight), "bc": 1.0, "mass": 20.0},
        "physics_ramp": "clip((step-200)/500,0,1)",
        "counts_per_iteration": {
            "data": 2048,
            "interior": 512,
            "initial": 512,
            "inlet": 128,
            "outlet": 128,
            "mass_quadrature": 4096,
        },
        "data_provenance": "400 x 48 cell FVM, 61 output times, 12 synthetic scenarios; 60000 labels/scenario with fixed seed 71713; 90% seeded train split",
        "data_loss": "MSE + 0.0001*MSE(log(pred+0.0001),log(target+0.0001))",
        "signatures": [e.signature() for e in experiments],
        "experiments": [asdict(e) for e in experiments],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "threads": threads,
        "architecture": "7-feature 3x64 Tanh, Softplus output, hard zero-concentration inlet/outlet/wall transforms v5",
        "width": 64,
        "limitations": "Positivity and boundary conditions are architectural (finite-precision tolerances apply); initial/PDE/mass residuals are soft constraints, not exact guarantees.",
    }
    torch.save({"metadata": metadata, "state_dict": best_state}, output)
    metadata["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return metadata


def refine_checkpoint(source, output, max_iter=200, mass_snapshot_weight=20.0):
    """A recorded, fixed-sample L-BFGS stage; never overwrite Adam results."""
    from core.pinn_inference import load_pinn
    from utils.provenance import provenance

    if not np.isfinite(mass_snapshot_weight) or mass_snapshot_weight < 0:
        raise ValueError("Mass snapshot weight must be finite and nonnegative")

    run_provenance = provenance()

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = load_pinn(source)
    meta = dict(model.metadata)
    seed = meta["seed"]
    variant = meta["variant"]
    pde_weight = float(meta["weights"]["pde"])
    rng = np.random.default_rng(seed + 400000)
    experiments = all_experiments()
    X, Y = build_dataset(experiments)
    split_rng = np.random.default_rng(seed + 100000)
    order = split_rng.permutation(len(X))
    ids = rng.choice(order[: int(0.9 * len(X))], 8192, replace=False)
    xd = torch.from_numpy(X[ids])
    yd = torch.from_numpy(Y[ids])
    z = sample(rng, 4096, experiments)
    ic = sample(rng, 2048, experiments)
    ic[:, 2] = 0
    bc = sample(rng, 512, experiments)
    bc[:256, 0] = 0
    bc[256:, 0] = 1
    # Every scenario contributes all six mass-snapshot times, including endpoints.
    # Targets are explicitly numerical supervision, not a hidden post-hoc correction.
    pairs = []
    mass_targets = []
    mass_scale = []
    for exp in experiments:
        teacher = solve(exp.refined(400, 48))
        for index in (0, 12, 24, 36, 48, 60):
            pairs.append([0.0, 0.0, index / 60, *exp.coefficients])
            mass_targets.append(
                teacher["mass_kg"][index]
                / (exp.length * exp.width * exp.initial_density)
            )
            mass_scale.append(
                teacher["mass_kg"][0] / (exp.length * exp.width * exp.initial_density)
            )
    pairs = np.asarray(pairs, dtype="float32")
    quad = np.repeat(pairs, 1024, axis=0)
    qx, qy = np.meshgrid(
        (np.arange(64) + 0.5) / 64, (np.arange(16) + 0.5) / 16, indexing="ij"
    )
    quad[:, 0] = np.tile(qx.ravel(), len(pairs))
    quad[:, 1] = np.tile(qy.ravel(), len(pairs))
    target_mass = torch.tensor(mass_targets, dtype=torch.float32)
    scale_mass = torch.tensor(mass_scale, dtype=torch.float32)
    tensors = [torch.from_numpy(a) for a in (z, ic, bc, quad)]
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=30,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    history = []
    start = time.perf_counter()

    def closure():
        opt.zero_grad(set_to_none=True)
        pred = model(xd)
        data = (pred - yd).square().mean() + 0.0001 * (
            torch.log(pred + 1e-4) - torch.log(yd + 1e-4)
        ).square().mean()
        if variant == "hybrid":
            p, i, b, m = physics_losses(model, *tensors)
            fitted_mass = model(tensors[-1]).reshape(len(pairs), 1024).mean(1)
            mass_data = ((fitted_mass - target_mass) / scale_mass).square().mean()
            loss = 10 * data + 20 * i + pde_weight * p + b + 20 * m + mass_snapshot_weight * mass_data
        else:
            loss = 10 * data
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite refinement loss")
        loss.backward()
        history.append(float(loss.detach()))
        if len(history) % 20 == 0:
            print(
                f"LBFGS {variant} {seed}: evaluation {len(history)}, loss {history[-1]:.7g}",
                flush=True,
            )
        return loss

    opt.step(closure)
    meta["parent_checkpoint_sha256"] = hashlib.sha256(
        Path(source).read_bytes()
    ).hexdigest()
    meta["refinement"] = {
        "provenance": run_provenance,
        "optimizer": "LBFGS",
        "max_iterations": max_iter,
        "closure_evaluations": len(history),
        "history_size": 30,
        "line_search": "strong_wolfe",
        "sampling_seed": seed + 400000,
        "fixed_counts": {
            "data": 8192,
            "interior": 4096,
            "initial": 2048,
            "boundary": 512,
            "mass_quadrature": len(quad),
        },
        "pde_weight": pde_weight if variant == "hybrid" else 0.0,
        "mass_snapshot_weight": float(mass_snapshot_weight) if variant == "hybrid" else 0.0,
        "mass_snapshot_supervision": f"hybrid only: 6 times x 12 scenarios, same numerical teacher; relative mass MSE weight {mass_snapshot_weight:g}; no inference renormalisation",
        "seconds": time.perf_counter() - start,
        "loss_history": history,
    }
    torch.save({"metadata": meta, "state_dict": model.state_dict()}, output)
    meta["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta
