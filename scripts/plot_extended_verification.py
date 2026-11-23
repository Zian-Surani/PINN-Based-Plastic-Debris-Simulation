"""Export audit figures from saved results; no model rerun or selection."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(folder):
    folder = Path(folder)
    analytic = json.loads((folder / "analytical.json").read_text())["spatial"]
    rows = json.loads((folder / "holdout_metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    nx = np.array([r["nx"] for r in analytic])
    errors = 100 * np.array([r["relative_l2"] for r in analytic])
    axes[0].loglog(nx, errors, "o-", label="Analytical ADE benchmark")
    axes[0].loglog(nx, errors[0] * nx[0] / nx, "--", label="First-order guide")
    axes[0].set(
        xlabel="Longitudinal cell count",
        ylabel="Final relative L2 error (%)",
        title="Finite-volume spatial refinement",
    )
    axes[0].legend()
    for variant, label, marker in [
        ("hybrid", "Hybrid", "o"),
        ("data_only", "Data-only", "s"),
    ]:
        selected = [r for r in rows if r["variant"] == variant]
        axes[1].scatter(
            [100 * r["relative_l2_spacetime"] for r in selected],
            [100 * r["mass_balance_relative_error"] for r in selected],
            label=label,
            marker=marker,
            alpha=0.8,
        )
    axes[1].set(
        xlabel="Space-time relative L2 error (%)",
        ylabel="Maximum mass-budget error (%)",
        title="Unseen hydraulic cases: 8 cases × 3 seeds",
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "extended_verification.png", dpi=300)
    fig.savefig(folder / "extended_verification.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    run(p.parse_args().folder)
