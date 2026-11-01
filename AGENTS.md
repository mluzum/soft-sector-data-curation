# soft-sector-data-curation: developer guide

A curated database of soft-sector heavy-ion measurements, and a pipeline that
computes the same observables from simulations. User documentation is in
README.md, docs/sim_interface.md (simulation input) and docs/validation.md.

## Design principle

**The curated record drives everything.** Comparison code reads the
measurement parameters from the record, never from hard-coded values:
- the η and pT acceptance;
- the subevent definitions and gaps;
- the centrality estimator and its acceptance;
- the harmonic, method and species.

- A new observable means a curated record plus a computation function that
  reads it.
- A schema in `observables/` defines validation **and** the metadata a
  computation needs. For example, `particles: AB` with
  `subevents.{A,B}.kinematic_range` triggers a two-subevent v_n{2}; without
  `particles` it is a single-acceptance v_n{2}.
- `value_type` links a record to its column in the raw HEPData table.

Simulations enter only through the two containers (below). No code here
reads a simulation's own file format, except the built-in JETSCAPE readers
(`--sim 2d|3d`). Other formats come through simulation packages and
external adapters.

## Layout

```
observables/   schemas: _common.yaml (collision_system, particle_id, kinematic_range,
               reference), centrality.yaml, one file per observable type
data/<type>/   curated records, one multi-document YAML file per HEPData table (all computable)
hepdata_raw/   raw HEPData YAML by INSPIRE id (records/ins<id>/), PHOBOS tables, record_index.yaml
curation/      curation decisions used by several records (e.g. flow_reference.yaml)
analysis/      library: dispatcher, containers, cores, weights, jackknife, toy MC, sim packages
scripts/       command-line tools (see below)
config/        sim_paths.example.yaml (copy to the untracked sim_paths.yaml)
tests/         run_all.sh; each test file runs standalone and prints "N/M passed"
docs/          sim_interface.md, validation.md, jetscape3d_pTeta_format.txt
data_validation/jetscape_bayes/   record <-> JETSCAPE model matching layer (tested)
```

## Scripts

- Database:
  - `download_hepdata.py`: HEPData search and download; resumable.
  - `classify_tables.py`: draft candidate records in `data_candidates/`, which is untracked.
  - `validate.py`, `validate_metadata.py`, `audit_metadata.py`, `check_cent_order.py`: schema and consistency checks.
  - `import_manual.py`, `import_phobos_data.py`: data without a HEPData record.
  - `search_records.py`: selection, and manifests with stable `obs_<hash>` ids.
  - `build_database_yaml.py`, `export_for_analysis.py`: catalog and export.
  - `curate_*.py`, `transcribe_*.py`: provenance of specific records.
- Simulations:
  - `run_pipeline.py`: records → simulation → HDF5.
  - `compute_flow_vn.py`: one record.
  - `check_sim_package.py`, `run_campaign.py`, `collect_campaign.py`.
  - `compute_ratios.py`: observable_ratios records from two simulation sets
    (linked numerator/denominator records, joint jackknife; `analysis/ratios.py`).
  - `plot_comparison.py`.

Run everything from the repository root: record paths are relative, as in
`data/<type>`.

## Records

Each YAML document has:
- `observable_type`, matching the schema file name;
- `inspire_id`, `version`, `hepdata_file`: the raw table;
- `value_type`: the table column; `dv_index` and `dv_average` select or
  average dependent variables;
- the schema's fields: `collision_system`, `centrality` (classes, estimator,
  η and pT), `kinematic_range`, `particle_id`, method fields.

Validate after every edit (`python scripts/validate.py data/`). Records are
the product. Never fix a value in code that belongs in a record.

**What goes in data/:** published records that compute and compare end to
end: resolve() dispatches them, their centrality classes are percentile
bins, and their experimental values load.  Ratio records qualify when both
linked sides do, and their link targets stay with them.
`tests/test_public_records.py` enforces this; it is the practical test of a
fully vetted record.  Everything else lives in a separate collection with
the same layout (data/<type>/, records carry `staging_reason`), used through
`--extra-records`, and moves here once it is vetted and computable:
- observable types without a computation;
- minimum-bias or p+p records (the pipeline needs centrality classes);
- PHOBOS ASCII tables until converted to a `value_table`;
- preliminary results (`status: preliminary`, with `source`).

Out-of-scope observables (jets, heavy flavor, electromagnetic probes) are
not curated here.

Fields that change dispatch:
- `efficiency_corrected: false`: a detector-level quantity (uncorrected
  track counts), refused.  A ratio of two such quantities is computed as the
  ratio of corrected ones only if it declares `efficiency_cancels: true`.
- `flow_reference.plane: spectator`: flow relative to the spectator plane
  (ZDC/SMD), refused, because models have no spectators.
- Ratio records link `numerator_record` / `denominator_record` (path, with
  `#<doc>` for multi-document files); `tests/test_ratio_links.py` checks every link.

## Pipeline

`resolve(record)` (in `analysis/dispatcher.py`) maps (method, cumulant
order, particles pattern, dimensionality) via `DISPATCH_TABLE`, or the
observable type via `DISPATCH_TABLE_BY_TYPE`, to a compute key.
`analysis/pipeline_dispatch.py:PIPELINE_DISPATCH[key]` provides:
- `prepare_data_2d` / `prepare_data_3d`, which cut the container with the
  record's acceptance;
- a `core_fn(data_dict, weights)` returning a flat array for one centrality
  class.

`run_pipeline.py` groups records by collision system, loads the container
once per group, builds the centrality weights, and runs one block
jackknife over all observables. It writes values, errors, the full
covariance, labels and the experimental points (also aligned with the
labels).

Containers (`analysis/qvector.py`):
- **QVectorData (2D)**: `Qn[event, species, harmonic, pT]`, slot 0 the counts.
  It is flat in rapidity over |y| < `rap_max`. An η window is applied through
  the exact dy/dη Jacobian (`cutQn`). Nesting the cut gives the
  multi-weighted sums needed for self-correlation terms; `dQ0 = Σw²`.
  Because of boost invariance, gapped and ungapped v_n{2} differ only by
  self-correlations.
- **QGridData (3D)**: `Qn[event, η, pT, harmonic]` on a grid. Particles are
  unit-weighted, so `dQ0 = Q0`. Disjoint subevents have no self-correlation,
  so longitudinal decorrelation is visible. `MultiSpeciesGrid.select()`
  sums per-species grids.
- **Simulation packages** (`analysis/sim_package.py`): `simulation.yaml` plus
  an external adapter returning the container arrays. `build_container`
  validates them. See docs/sim_interface.md §4.

Centrality (`analysis/weights.py`):
- simple percentiles, or Poisson-fluctuated weights, of the multiplicity in
  the record's estimator acceptance;
- anchored to the total cross section by `resolve_anchor` (fraction f,
  N_ref, {b_max, sigma_tot} in fm², or {sigma_eff, sigma_tot});
- bins beyond the anchor get zero weight, and bins below `--min-cent-weight`
  are dropped from the output.

Errors: a delete-block jackknife (20 blocks by default) with the
bias-corrected central value θ̂ = Nθ − (N−1)θ̄.

## Conventions and traps (each was a real bug once)

- **Output layout**:
  - cores return harmonic-major arrays; v2 is index `harmonic - 1` for
    multi-harmonic cores;
  - single-row cores index by bin;
  - differential cores are `(h-1)*n_bins + b`, except `vn2_single_diff_pT`
    (`h*n_bins + b`).

  `run_pipeline` asserts labels == values before writing; keep that check.
- **Charge factors**:
  - 2D JETSCAPE containers hold positive charges only, so charged
    quantities take ×2, and self-correlation terms take ×CF², consistently;
  - containers with `pdg_ids` resolve exact charges;
  - `QGridData.positives_only=False` for grids that already hold all
    charged hadrons.
- **Centrality order**: the sort is by descending multiplicity. The first
  bin starts at the record's lower edge, not 0.
- **Integrated records never get differential bins**:
  `extract_differential_bins` returns none for `dimensionality: integrated`.
- **Experimental scales**: percent or ×100 columns and MeV headers are
  normalized in `load_experimental`.
- **Experimental errors**: `load_experimental` returns the total error, all
  HEPData components in quadrature (asymmetric: the mean of the two sides;
  `x%`: relative to the value).  Until November 2026 a third component was
  ignored and an asymmetric second one dropped.
- **Reproducibility**: outputs agree run to run to about 1e-16, not bit for
  bit (numpy SIMD reductions depend on memory alignment). Compare outputs
  with a relative tolerance, and containers exactly.
- **References by document index go stale**: record files change (records
  added, files split, records moved out of data/).
  - `run_pipeline --manifest` checks each `fpath`/`doc_index` pin against
    the entry's id and `centrality_classes`, and skips a stale pin rather
    than guessing.
  - `compare_postproc.py` checks inventory rows the same way.
  - Ratio links are checked by `tests/test_ratio_links.py`.

## Tests

`bash tests/run_all.sh`. The toy-MC tests (`tests/test_toy_mc_*.py`,
`analysis/toy_mc.py`) generate particles with known flow and spectra, push
them through the production path, and compare with particle-level
estimators. They need no external data. A new observable or core needs a
toy test, and its output layout registered in `toy_mc._extract_with_ncent`.
