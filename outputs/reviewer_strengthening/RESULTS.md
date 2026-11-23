# Reviewer-strengthening results

Protocol: `protocol.json`. All original checkpoints remain frozen; nine new final checkpoints complete a three-seed 2×2 factorial. No evaluation-based checkpoint selection was performed.

## Independent reference

Baseline neural cell-average error is 2.8591% in space–time and 4.9738% at final time against the refined reference. These replace any interpretation of 1.2318% as a PDE-accuracy claim; that value remains agreement with the original upwind reference.

The separable centered-flux Crank–Nicolson reference is specific to constant coefficients and the prescribed separable initial condition. It is not an exact solution. The finest self-difference zero is not evidence of exactness. Tiny negative values are reported floating-point effects, not clipped outputs.

## Controlled comparison

All factorial cells retain initial-condition, boundary and physical integral-mass penalties. The factors are the pointwise PDE residual and additional numerical mass labels.

| Scope | PDE weight | Mass-label weight | Mean refined L2 (%) | Max refined L2 (%) | Max budget (%) |

|---|---:|---:|---:|---:|---:|

| supported | 0 | 0 | 5.0709 | 6.3270 | 0.4983 |

| supported | 0 | 20 | 5.2158 | 6.5211 | 0.5275 |

| supported | 4 | 0 | 2.6277 | 3.3846 | 0.8069 |

| supported | 4 | 20 | 2.9165 | 3.7558 | 0.8448 |

| holdout | 0 | 0 | 5.0626 | 6.2022 | 0.4191 |

| holdout | 0 | 20 | 5.2062 | 6.3804 | 0.4514 |

| holdout | 4 | 0 | 2.6030 | 3.2270 | 0.6390 |

| holdout | 4 | 20 | 2.8923 | 3.5183 | 0.7343 |

The pointwise PDE term improves refined-reference error for every paired seed at both mass-label levels. Additional numerical mass labels worsen refined-reference error in this experiment, even where they improve agreement with the upwind reference. This is evidence of a reference-dependent ranking, not proof that mass supervision is generally harmful. Scenario rows are correlated; the three seeds are the replication units. No significance claim is made.

## Practical sensitivity demonstration

For the frozen full hybrid, retained-mass derivative errors at discharge multipliers 0.85, 1.0 and 1.15 are 3.93%, 3.91% and 2.20%. The demonstration is restricted to the fixed plain-terrain Manning path and initial pulse; it is not a field-control recommendation.

## Interpretation and novelty

The contribution is an experimentally supported verification protocol that distinguishes agreement with a discretized teacher, accuracy against a converged independent method, physical mass budgets, and derivative utility. Hard boundaries, physics losses, mass penalties and numerical-bias concerns are established ideas; no first-of-kind algorithm claim is made.

## Reproduction

Run `python -m scripts.reviewer_ablation --seed SEED` for each of 2026/2027/2028; then `python -m scripts.reviewer_reference_audit`, `python -m scripts.reviewer_evaluate` and `python -m scripts.plot_reviewer_results`. Existing checkpoints/results are reused only where explicitly documented.