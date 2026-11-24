# Dataset provenance and limits

The local raw-file audit is recorded in `outputs/data_provenance/audit.json`, including SHA-256 hashes, file sizes, dataset metadata and the executable width derivation. Large third-party datasets are not redistributed or covered by the software MIT licence.

| Resource | Verified local evidence | Role in reported experiment |
|---|---|---|
| GRWL summary statistics | 42,077 records; medians of minimum/median/mean/maximum width fields: 30/108/124/509 m | Informs prescribed width ranges and their midpoints; no geographically selected study reach |
| CHIRPS v2, 2022 monthly precipitation | NetCDF file and earlier stored precipitation factors 0.08/1/1.36 | Not the model's discharge factors, which are 0.7/1/1.3; no calibrated rainfall-runoff transformation |
| HydroSHEDS v1, 15 arc-second accumulation/direction | Two readable global EPSG:4326 rasters | Available locally but not connected to active slope/discharge values; no measured discharge or derived reach slope is claimed |
| ERA5 | No input or implemented ingestion located in the audited project | Not used in the reported experiment |

## Reproduced width derivation

For each GRWL summary field, the median is taken over the full supplied file without a geographic subset. Denote the resulting minimum, median and maximum summaries by 30, 108 and 509 m. The historical abstraction defines mountain [30, 0.7×108], plain [0.7×108, 1.5×108], urban [0.5×108, 108] and coastal [1.5×108, 509] m. Their midpoints are **52.8, 118.8, 81.0 and 335.5 m**, exactly the current configuration.

The terrain assignments and multipliers are modelling choices, not GRWL-provided classifications. This is dataset-informed parameterization of an idealized experiment, not observed terrain-specific geometry. The previous blanket description of all parameters as having no dataset lineage was too broad and is corrected here.

No available dataset supplies observed tracer density or validates the passive-tracer/plastic-process assumptions. Model labels come from the finite-volume solver. Rainfall, flow accumulation, direction and river width must not be described as measured tracer validation. HydroSHEDS accumulation is not discharge, and a direction raster alone is not a reach-slope measurement.

## Reproduce the audit

Use a separate Python 3.11 environment with `requirements-data-audit-lock.txt`. Place the original files at the paths recorded in audit.json, then run:

```powershell
python scripts/audit_data_provenance.py
```

The audit checks configured widths against recomputed raw GRWL statistics and fails if they differ. It reads CHIRPS and HydroSHEDS metadata and hashes raw inputs. It does not claim to reproduce the old global CHIRPS quantiles or create a new hydrological calibration. Scientific inference uses the pinned main environment and does not require these optional data libraries.

## Authoritative sources

- GRWL: Allen and Pavelsky (2018), *Global extent of rivers and streams*, Science 361, 585–588. https://doi.org/10.1126/science.aat0636. Dataset: https://doi.org/10.5281/zenodo.1297434; USGS catalogue: https://water.usgs.gov/catalog/datasets/120270a9-e0b6-42d8-9b1f-17db852fd2b4/
- HydroSHEDS core products and technical definitions: https://www.hydrosheds.org/products/hydrosheds and https://data.hydrosheds.org/file/technical-documentation/HydroSHEDS_TechDoc_v1_4.pdf
- CHIRPS v2: https://www.chc.ucsb.edu/data/chirps
- ERA5 description (not an input to this study): https://www.ecmwf.int/en/forecasts/datasets/complete-era5-global-atmospheric-reanalysis

Adding location-specific coupling would require a declared reach, dates, extraction/aggregation rules, a hydrological model or measured discharge, uncertainty assessment and new experiments. Existing results cannot be relabelled as if those steps had occurred.
