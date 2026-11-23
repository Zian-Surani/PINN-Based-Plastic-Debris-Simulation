"""Matched-seed full hybrid versus data-only training on identical labels.

Both variants have the same architecture and initial weights per seed; they
draw identical supervised minibatches. Training metadata records all losses.
"""

from pathlib import Path
from core.training import train_checkpoint, refine_checkpoint


def run_ablation_experiment(output_dir, seeds=(2026, 2027, 2028), iterations=6000):
    result = []
    for seed in seeds:
        for variant in ("hybrid", "data_only"):
            path = Path(output_dir) / f"{variant}_{seed}.pt"
            if path.exists():
                raise FileExistsError(
                    "Use a fresh directory for a new experiment; do not mix archived runs."
                )
            train_checkpoint(path, seed, iterations, variant)
            final = path.with_name(path.stem + "_final.pt")
            result.append(refine_checkpoint(path, final, max_iter=300))
    return result
