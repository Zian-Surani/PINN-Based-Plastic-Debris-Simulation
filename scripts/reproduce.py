"""Resume a declared full study; isolate RNG state in separate processes."""

import argparse, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from config import SEEDS, SCHEMA_VERSION, all_experiments


def job(seed, variant):
    root = Path("outputs/checkpoints/v5")
    root.mkdir(parents=True, exist_ok=True)
    adam = root / f"{variant}_{seed}.pt"
    final = root / f"{variant}_{seed}_final.pt"

    def compatible(path):
        m = json.loads(path.with_suffix(".json").read_text())
        return (
            m["schema"] == SCHEMA_VERSION
            and m["seed"] == seed
            and m["variant"] == variant
            and m["iterations"] == 6000
            and m["signatures"] == [e.signature() for e in all_experiments()]
        )

    with (root / f"{variant}_{seed}.run.log").open("a", encoding="utf-8") as log:
        if adam.exists():
            if not compatible(adam):
                raise ValueError(f"Incompatible existing run: {adam}")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "scripts.train",
                    "--seed",
                    str(seed),
                    "--iterations",
                    "6000",
                    "--variant",
                    variant,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if final.exists():
            if (
                not compatible(final)
                or json.loads(final.with_suffix(".json").read_text())["refinement"][
                    "max_iterations"
                ]
                != 300
            ):
                raise ValueError(f"Incompatible refinement: {final}")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "scripts.refine",
                    str(adam),
                    str(final),
                    "--iterations",
                    "300",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print("Completed", variant, seed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(
            pool.map(
                lambda c: job(*c),
                [(s, v) for s in SEEDS for v in ("hybrid", "data_only")],
            )
        )
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scripts.run_validation",
            "--output",
            str(a.output),
        ],
        check=True,
    )
