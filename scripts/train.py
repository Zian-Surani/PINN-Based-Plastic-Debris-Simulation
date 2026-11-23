"""python -m scripts.train --seed 2026 --iterations 6000"""

import argparse
from pathlib import Path
from core.training import train_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--variant", choices=["hybrid", "data_only"], default="hybrid")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    train_checkpoint(
        a.output or Path(f"outputs/checkpoints/v5/{a.variant}_{a.seed}.pt"),
        a.seed,
        a.iterations,
        a.variant,
        a.threads,
    )


if __name__ == "__main__":
    main()
