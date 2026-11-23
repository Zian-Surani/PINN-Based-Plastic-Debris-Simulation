# Additional verification completed

The six previously trained checkpoints were frozen and their hashes verified unchanged. The 48 new evaluations comprise eight unseen discharge settings for each of three seeds and two models. All 24 hybrid and all 24 data-only evaluations passed the declared numerical criteria. Hybrid maximum space-time L2 was 1.6701%; maximum mass-budget error was 0.7343%. See summary.json and holdout_metrics.json.

The independent analytical transport benchmark decreases from 0.4547% to 0.05929% error with refinement, consistent with the solver's first-order upwind advection. Fixed-grid temporal tests approach a spatial-error floor.

The 800-cell reference differs by 2.5142% from the restricted 1600-cell reference at the final time. This is a measured reference-resolution limitation, not an exact-error bound. Do not equate neural errors against the reference with error against the exact PDE solution.

A 600-second finite-volume rollout remained nonnegative and conservative (relative budget error about 5e-16). The neural horizon is still 300 seconds. New initial conditions and field data remain outside this validation.

The app explicitly warns about residual-interval undercoverage. This audit does not establish a calibrated uncertainty method or a speed advantage. The original baseline benchmark remains unchanged and favors the finite-volume solver.

Verification: 19 scientific tests passed, static checks passed, app action and undercoverage warning passed, original baseline metrics unchanged, and all six checkpoint hashes unchanged. The final code is archived in source_snapshot.zip; protocol.json records the pre-evaluation plan. The old verification_v5 report remains a historical artifact with its original source hashes.

Use ../../docs/MANUSCRIPT_REVISION_GUIDE.docx or the matching Markdown file for the 16-section manuscript revision list. Original manuscript files were not edited. extended_verification.png and .pdf provide the new benchmark/holdout figure.
