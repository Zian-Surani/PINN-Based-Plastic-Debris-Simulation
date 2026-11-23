import argparse
from pathlib import Path
from core.training import refine_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--iterations", type=int, default=200)
    a = p.parse_args()
    refine_checkpoint(a.source, a.output, a.iterations)
