# Discretization-aware verification and discharge sensitivity

This extension tests a specific scientific question: can agreement with a numerical training/reference scheme conceal PDE approximation error and change the apparent ranking of hybrid losses?

The executed experiment combines an independently refined reference, a matched-seed 2×2 factorial, explicit open-system mass budgets and a verified derivative use case. It does not claim that hard boundaries, mass penalties, PINNs or numerical-discretization concerns are new inventions.

## Evidence

- The full hybrid's baseline space–time error is 1.2318% against the original upwind reference but 2.8591% when neural cell averages are compared with an independently refined centered-flux/Crank–Nicolson reference. Final-time error in the latter comparison is 4.9738%.
- Across supported cases and three seeds, the pointwise PDE penalty reduces mean refined-reference error from 5.0709% to 2.6277% without numerical mass labels, and from 5.2158% to 2.9165% with them. Initial-condition, boundary and physical integral-mass penalties are held fixed in all four cells.
- Adding numerical mass labels to the PDE-enabled model reduces mean error against the upwind reference from 1.2892% to 1.2081%, but increases refined-reference error from 2.6277% to 2.9165%. All three paired seeds show this direction. This is a reference-dependent ranking, not a universal indictment of mass supervision.
- Frozen discharge sensitivities of final retained mass have 2.20–3.93% relative derivative error against converged finite differences at three plain-terrain settings. The app exposes this diagnostic within the audited discharge range. No optimization or field-control benefit is claimed.

## Relation to established work

Mamud et al. (2024), *Scientific Reports* 14, 15541, DOI 10.1038/s41598-024-65472-9, already demonstrate that PINN mass-balance accuracy needs independent evaluation. Vadyala, Betgeri and Betgeri (2022), *Array* 13, 100110, DOI 10.1016/j.array.2021.100110, already compare advection PINNs with numerical schemes and discuss numerical diffusion. The present work's contribution is the controlled combination of these checks with an explicit additional-supervision factor and a verified hydraulic derivative task, within a fully specified synthetic benchmark. A broad claim of algorithmic priority is not supported.

## Limits

The refined reference exploits separability, spatially constant coefficients and a fixed pulse; it is not a general hydrodynamic solver. The three seeds are replication units, not the correlated scenario rows. Additional hydraulic tests are interpolation along the prescribed Manning relationship. Mass budgets are global and approximate for the neural model. No field data, new initial-condition family, neural extrapolation beyond 300 s, guaranteed uncertainty coverage, or speed superiority is established. Numerical mass labels remain part of the frozen original full-hybrid model; the factorial alternatives are reported without retrospectively relabeling an evaluated checkpoint as an independently selected winner.

See `outputs/reviewer_strengthening/RESULTS.md`, the raw JSON/CSV files and the protocol written before the new experiments. The original schema-5 outputs remain available and are not overwritten.
