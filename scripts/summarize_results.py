"""Summarize completed validation artifacts without selecting favorable cases."""

import argparse
import json
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    rows = json.loads((folder / "metrics.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    timings = json.loads((folder / "timings.json").read_text())
    lines = [
        "# Corrected numerical verification results",
        "",
        "These results supersede the historical manuscript numbers for the revised code. "
        "They describe synthetic numerical verification, not field validation.",
        "",
        f"CPU: {manifest['cpu']}. Device: CPU; {manifest['torch_threads']} PyTorch threads. "
        f"Seeds: {manifest['seeds']}. All scenarios run to 300 seconds.",
        "",
        "| Model | Cases | Passed declared checks | Space-time L2 range | Mass-budget error range |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("hybrid", "data_only"):
        subset = [r for r in rows if r["variant"] == variant]
        l2 = [100 * r["relative_l2_spacetime"] for r in subset]
        mass = [100 * r["mass_balance_relative_error"] for r in subset]
        lines.append(
            f"| {variant} | {len(subset)} | {sum(r['accepted_numerically'] for r in subset)} | "
            f"{min(l2):.3f}–{max(l2):.3f}% | {min(mass):.3f}–{max(mass):.3f}% |"
        )
    lines += [
        "",
        "## Baseline: plain / normal",
        "",
        "| Model | Seed | Space-time L2 | Maximum mass-budget error | Initial field L2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["terrain"] == "plain" and r["season"] == "normal":
            lines.append(
                f"| {r['variant']} | {r['seed']} | {100 * r['relative_l2_spacetime']:.3f}% | "
                f"{100 * r['mass_balance_relative_error']:.3f}% | {100 * r['initial_relative_l2']:.3f}% |"
            )
    key = f"hybrid_{manifest['seeds'][0]}_plain_normal"
    timing = timings[key]
    lines += [
        "",
        "## Baseline timing",
        "",
        f"{timing['repeats']} repetitions, one warm-up, alternating method order. "
        "Both methods produce the same output grid and time samples. Training is excluded.",
        "",
    ]
    for name, stat in timing["summary"].items():
        lines.append(
            f"- {name}: {stat['mean_s']:.6f} ± {stat['sample_sd_s']:.6f} s (mean ± sample SD)."
        )
    failure = [
        f"{r['variant']}/{r['seed']}/{r['terrain']}/{r['season']}"
        for r in rows
        if not r["accepted_numerically"]
    ]
    lines += [
        "",
        "## Interpretation and reproducibility",
        "",
        "- Nonnegative output and boundary conditions are constructed into the network. Neural conservation remains approximate; errors are measured without clipping or mass rescaling.",
        "- Acceptance criteria: space-time L2 <10%, initial-field L2 <5%, mass-budget error <2%, nonnegative density, and conservative-reference balance <1e-10. These are declared numerical criteria, not journal standards.",
        "- Full results, including every failure, are in metrics.csv and metrics.json. Raw timing repetitions are in timings.json; source/checkpoint hashes and hardware are in manifest.json.",
        "- The finer reference was used during development diagnostics. This is not a blinded test or an unseen-parameter generalization study.",
        "- Numerical residual intervals have no guaranteed coverage. Read residual_intervals.json before making uncertainty claims.",
        "- Report generation does not edit manuscript tables. A manuscript must identify the exact experiment and reference used; legacy five-second results are not interchangeable with this 300-second study.",
        f"- Failed cases: {', '.join(failure) if failure else 'none under the declared numerical checks'}.",
        "",
    ]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
