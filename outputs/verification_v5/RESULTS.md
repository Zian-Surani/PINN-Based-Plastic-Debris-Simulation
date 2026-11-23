# Corrected numerical verification results

These results supersede the historical manuscript numbers for the revised code. They describe synthetic numerical verification, not field validation.

CPU: AMD Ryzen 9 8940HX with Radeon Graphics. Device: CPU; 2 PyTorch threads. Seeds: [2026, 2027, 2028]. All scenarios run to 300 seconds.

| Model | Cases | Passed declared checks | Space-time L2 range | Mass-budget error range |
|---|---:|---:|---:|---:|
| hybrid | 36 | 36 | 0.699–1.968% | 0.079–0.845% |
| data_only | 36 | 36 | 1.369–2.374% | 0.217–1.773% |

## Baseline: plain / normal

| Model | Seed | Space-time L2 | Maximum mass-budget error | Initial field L2 |
|---|---:|---:|---:|---:|
| hybrid | 2026 | 1.232% | 0.448% | 1.456% |
| hybrid | 2027 | 1.360% | 0.610% | 1.419% |
| hybrid | 2028 | 1.326% | 0.480% | 1.239% |
| data_only | 2026 | 1.783% | 0.600% | 1.202% |
| data_only | 2027 | 1.724% | 0.585% | 1.072% |
| data_only | 2028 | 1.768% | 0.799% | 1.114% |

## Baseline timing

10 repetitions, one warm-up, alternating method order. Both methods produce the same output grid and time samples. Training is excluded.

- pinn: 0.101137 ± 0.001587 s (mean ± sample SD).
- finite_volume: 0.015980 ± 0.000226 s (mean ± sample SD).

## Interpretation and reproducibility

- Nonnegative output and boundary conditions are constructed into the network. Neural conservation remains approximate; errors are measured without clipping or mass rescaling.
- Acceptance criteria: space-time L2 <10%, initial-field L2 <5%, mass-budget error <2%, nonnegative density, and conservative-reference balance <1e-10. These are declared numerical criteria, not journal standards.
- Full results, including every failure, are in metrics.csv and metrics.json. Raw timing repetitions are in timings.json; source/checkpoint hashes and hardware are in manifest.json.
- The finer reference was used during development diagnostics. This is not a blinded test or an unseen-parameter generalization study.
- Numerical residual intervals have no guaranteed coverage. Read residual_intervals.json before making uncertainty claims.
- The original manuscript is not synchronized with this new 300-second experiment. Its old values, training claims and figures must not be reused as current results.
- Failed cases: none under the declared numerical checks.
