"""Generate new results without editing manuscript files or archived tables."""

import argparse, json, hashlib, platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import all_experiments, make_experiment, SEEDS, SCHEMA_VERSION
from core.pinn_inference import load_pinn
from simulation.validation_suite import evaluate, benchmark_runtime, grid_refinement
from simulation.uncertainty_quantification import residual_interval_audit
from utils.provenance import provenance


def run(output, seeds=SEEDS):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh result directory to preserve provenance")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    rows = []
    timings = {}
    intervals = []
    checkpoints = {}
    for variant in ("hybrid", "data_only"):
        for seed in seeds:
            path = Path(f"outputs/checkpoints/v5/{variant}_{seed}_final.pt")
            model = load_pinn(path)
            checkpoints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for exp in all_experiments():
                result = evaluate(model, exp)
                m = result["metrics"]
                m.update(seed=seed, variant=variant)
                rows.append(m)
                key = f"{variant}_{seed}_{exp.terrain}_{exp.season}"
                print(
                    key,
                    "L2",
                    round(m["relative_l2_spacetime"], 4),
                    "balance",
                    round(m["mass_balance_relative_error"], 4),
                    "accepted",
                    m["accepted_numerically"],
                    flush=True,
                )
                uq = residual_interval_audit(result["prediction"], result["reference"])
                uq.pop("half_width")
                uq["case"] = key
                intervals.append(uq)
                if seed == seeds[0]:
                    timings[key] = benchmark_runtime(model, exp, 10)
                if (
                    variant == "hybrid"
                    and seed == seeds[0]
                    and exp.terrain == "plain"
                    and exp.season == "normal"
                ):
                    np.savez_compressed(
                        output / "baseline_fields.npz",
                        prediction=result["prediction"],
                        reference=result["reference"],
                        times=exp.times(),
                        x=exp.grid()[0],
                        y=exp.grid()[1],
                    )
                    x, y = exp.grid()
                    fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
                    for ax, field, title in zip(
                        axs,
                        (
                            result["prediction"][-1],
                            result["reference"][-1],
                            result["prediction"][-1] - result["reference"][-1],
                        ),
                        ("Hybrid PINN", "Independent finer reference", "Signed error"),
                    ):
                        im = ax.pcolormesh(x, y, field.T, shading="auto")
                        ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
                        fig.colorbar(im, ax=ax, label="kg/m²")
                    fig.tight_layout()
                    fig.savefig(output / "baseline_fields.png", dpi=180)
                    plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "acceptance"} for r in rows]
    ).to_csv(output / "metrics.csv", index=False)
    (output / "timings.json").write_text(json.dumps(timings, indent=2))
    (output / "residual_intervals.json").write_text(json.dumps(intervals, indent=2))
    (output / "grid_refinement.json").write_text(
        json.dumps(grid_refinement(make_experiment()), indent=2)
    )
    manifest = {
        **provenance(),
        "schema": SCHEMA_VERSION,
        "experiments": [asdict(e) for e in all_experiments()],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "seeds": list(seeds),
        "checkpoint_sha256": checkpoints,
        "training_grid": [400, 48],
        "evaluation_grid": [200, 23],
        "independent_reference_grid": [800, 92],
        "notes": "All physical parameters, BCs and ICs shared. Grids intentionally differ for resolution validation. Numerical test scenarios are within supported training configurations; no unseen-parameter or field-validation claim.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    from scripts.summarize_results import summarize

    summarize(output)
    print("Saved", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args()
    run(a.output, tuple(a.seeds))
