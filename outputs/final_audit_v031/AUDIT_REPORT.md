# Final audit, 24 November 2026

## Executed verification

- Re-evaluated all 72 supported formulation/seed/configuration rows. All declared numerical checks passed.
- Space-time error, initial error, mass-budget error and minimum density agree exactly with the archived metrics. Baseline neural and reference arrays also agree exactly. See reproduction_comparison.json.
- Twenty-five scientific unit tests pass; the pinned scientific environment passes pip check.
- Streamlit app load, verification action and discharge-sensitivity action pass without exceptions.
- Regenerated all 28 manuscript figures from released numerical records/checkpoints or explicitly identified mathematical schematics. No generated-image model supplies experimental plots.
- Recomputed GRWL width statistics from 42,077 raw records; the resulting four midpoints match config.py. Raw files were hashed and CHIRPS/HydroSHEDS metadata inspected.

## Manuscript and bibliography

The final manuscript has 29 pages, 28 figures, 15 tables and 22 numbered references. Citation numbers follow first appearance. The broad neural-network textbook was replaced with the primary GRWL reference, keeping the reference count at 22. DOI metadata and publisher-page access are recorded separately in reference_audit.json. Publisher access failures are not represented as successful page verification.

The final DOCX has a Word read-only restriction, with no password; its SHA-256 is recorded alongside the PDF and embedded figure hashes. This is an editorial version freeze, not a guarantee against editing or a claim of scientific infallibility. Later changes must receive a new version and audit.

## Findings that cannot be concealed by editing

- GRWL informs global width ranges; terrain assignment remains a modelling abstraction.
- CHIRPS rainfall factors are not the active discharge factors. HydroSHEDS files do not feed the active slope/discharge calculations. No ERA5 ingestion was found. Four-dataset coupling is not claimed.
- No measured tracer/plastic field validation exists. Neural mass conservation is approximate; the neural horizon remains 300 seconds. No rollout speed advantage is demonstrated.
- This paper is a controlled verification/sensitivity study. Architecture novelty and journal acceptance are not guaranteed.
- Zenodo DOI remains pending by instruction.
- Both authors are explicitly credited. The historical Git author email mismatch is documented and mapped without rewriting published commits; GitHub's contributor widget may not reflect this mapping.

## Timing

Fresh timings are retained as an audit, not substituted into the manuscript to improve a benchmark. Manuscript timing figures and tables consistently refer to the original ten-repeat experiment. No checkpoints were retrained, selected or replaced during this audit.
