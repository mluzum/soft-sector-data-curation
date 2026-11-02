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
- Directed flow (`harmonic: 1`) needs `flow_reference.odd` (an eta-odd
  reference) and `sign_convention`: computed by `v1_odd_sp_diff_eta`
  (3D only).  Otherwise refused: the even-harmonic machinery would return
  ~0.  Rapidity-even v1 is not implemented.
- `kinematic_range.y.mass`: a rapidity window with one assumed mass (PHOBOS
  y_pi), spectra only.
- `method: event_plane` or `scalar_product` with `particles: AB` (STAR's
  eta-sub event plane: the particle in one TPC half, the plane from the
  other; ALICE's two-subevent scalar product): the
  two-subevent correlation below, the event-plane result at low
  resolution.
- Two-subevent v_n{2} (`particles: AB`) of an identified species: computed
  by `vn2_gap_poi`, the particle of interest (record species, top-level
  pT) in one subevent with the reference (`subevents.<X>.particle_id`,
  default charged) in the other, over the reference v_n{2}ab.  A pT bin
  without POI pairs gives NaN (until November 2026: 0).  Its 3D
  prepare receives the unselected container (`selects_species`,
  `pipeline_dispatch.container_for_prepare`).
- `observable_type: radial_flow` (`analysis/radial_flow.py`, key
  `radial_flow`): `quantity` v0 | v0_pT | normalized_covariance |
  v0_pT_over_v0, two subevents (`particles: AB`, `subevents.{A,B}`
  eta windows), `reference_pT` (the [pT] range), kinematic_range.pT (the
  range normalizing n(pT), plus the pT bins), optional
  `multiplicity_baseline` (deviations at fixed multiplicity, the
  counterpart of narrow N_ch bins: per-class linear regression).  Pairs of
  a particle with itself are removed exactly from per-cell sums of
  w_a w_b (1, pT, pT^2): they matter on 2D containers, where both windows
  hold every particle.  One value per pT bin (`SINGLE_ROW_KEYS`);
  tests/test_toy_mc_radial_flow.py.
- `value_scale`: a factor the tabulated values carry (1000 for "x 10^3");
  load_experimental divides by it.
- `particle_spectra` `value_type`: `invariant_yield` (1/(2 pi pT)
  d2N/(dpT dy)) or `dN_dpT_dy` (d2N/(dpT dy), ALICE 1759506); anything
  else is refused.  audit_metadata checks it against the header.
- `errors_excluded`: error labels contained in another component (ALICE
  'syst. uncorr.' inside 'syst.'), not added again by load_experimental.
- `table_rows`: the table rows a record holds (ascending).  For tables
  whose rows overlap as classes (0-1% and 0-5% in one ATLAS table), one
  record per non-overlapping set; the loader and the audits honour it.
- `centrality.points` + `point_window`: a table that gives one centrality
  value per point and no class edges; the classes are the windows around
  the points (validate.py checks this).
- A `particle_id.species` the simulations cannot provide (heavy flavour,
  quarkonia, leptons, photons, eta, resonances such as Sigma(1385), ...)
  is refused (`analysis.species.species_known`) instead of computing zero.
  pi0 and K0_S come from isospin, (pi+ + pi-)/2 and (K+ + K-)/2, unless
  the container has an explicit 111 / 310 axis (`species._ISOSPIN`).
  A species the container lacks (phi, Omega, Lambda on 2D JETSCAPE; any
  identified species on a 3D grid without species grids) is skipped by
  run_pipeline and the toy harness (`species.species_unavailable`):
  until November 2026 it computed zero, or (3D) charged hadrons, and
  'Lambda' read the proton axis x 2.
- `particle_id.charge_average`: the measurement is (h+ + h-)/2 (or
  (pi+ + pi-)/2, ...); absolute quantities are half of the sum.
- Ratio records link `numerator_record` / `denominator_record` (path, with
  `#<doc>` for multi-document files); `tests/test_ratio_links.py` checks every link.
- Symmetric cumulants (`correlation_type: symmetric_cumulant`, key
  `sc_mn`): two harmonics only, and `normalized` is required (true: NSC =
  SC / (<v_m^2><v_n^2>)); `normalization_eta_gap` takes the denominator
  from the two halves of the window separated by the gap (ALICE, STAR:
  |Delta eta| > 1).  SC of three harmonics is refused.
- Asymmetric systems (p+Pb, d+Au, 3He+Au, Cu+Au): `collision_system.frame`
  (lab | nn_cms), `rapidity_shift` (y_lab = y_cms + shift; ALICE p-Pb
  -0.465) and `forward_beam` (the beam toward +eta) state the frame of
  every eta/y window, the estimator's included.  2D containers are
  boost-invariant and frame-independent.  On 3D containers
  `analysis/frames.py` maps the windows into the container frame
  (nn_cms, `forward_beam` from `sim_frames` in config/sim_paths.yaml or
  the package's system entry): eta_c = +-(eta - shift), bins keep their
  row order; mirrored/segmented references are refused, and a container
  without a declared frame skips the record.  JETSCAPE 3D p+Pb and d+Au
  put the light projectile at +eta (tests/test_frames.py).
- `centrality.estimator.observable: spectator_neutrons` (ALICE ZNA): refused,
  models have no spectators.
- Centrality labels that are not percentile bins (a lone `MB` too: minimum
  bias is no class of the model's events) are refused
  (`centrality_classes_suspicious`); a record without
  `collision_system.energy_gev` matches no simulation group and is not
  public (`test_public_records`).
- `centrality.actual_classes` (per-system actual ranges of the nominal
  labels) assign the model classes; `centrality.normalization`
  {estimator_threshold, fraction_above, efficiency} replaces the group's
  sigma_inel anchor for that record: f = fraction_above / <P(Ntrk >=
  threshold)>, Ntrk = Poisson(efficiency x N) in the estimator acceptance
  (`weights.threshold_anchor`, run_pipeline `_record_anchor`).  f > 1 (a
  low-multiplicity model) is kept: its most peripheral events fall beyond
  100%; the record's own anchor is written as record_* attrs.  STAR
  isobars (1914564, 2822399): Table III ranges, F50 = 0.4712 (Ru), 0.4590
  (Zr), efficiency 0.87 (scripts/curate_star_isobar_centrality.py).
- `efficiency_corrected: false` with `detector_efficiency` (yields,
  spectra) is computed as efficiency x N; without it, refused.
- `centrality.combined_from: unit_multiplicity`: class values combined
  from bins of one unit of the estimator multiplicity (STAR's
  centrality-bin-width correction); refused, since the pipeline computes
  over the whole class.
- `centrality.combined_from: analysis_classes`: class values averaged from
  narrower percentile classes (`analysis_classes`, or
  `analysis_class_width`: ALICE 1%), weighted by `combination_weight`
  (yield of the record's species in its acceptance | charged_yield |
  events).  `analysis/cent_combine.py`: the observable is computed per
  sub-class and the values averaged, inside the jackknife; sub-classes
  must tile each class.  run_pipeline wraps the entry's core when tagging
  it (`_tag_entries`), the toy harness likewise;
  tests/test_toy_mc_cent_combine.py.
- A dependence on a lower pT cut (`p_T,min` axis, e.g. ALICE SC and v_n
  vs pT_min): one integrated record per point, pT [pT_min, max] and
  `table_rows: [j]`; audit_metadata checks the pairing.

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
once per group (the config key of a reaction string, 'AU AU', 'D AU',
'^3He Au', ...: `analysis.reactions.system_key`; a system without a
configured simulation is skipped, the Au+Au 200 GeV proxy only with
`--sim-proxy`; until November 2026 the proxy was the default and 'D AU'
records missed the configured d+Au simulation), builds the centrality weights, and runs one block
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
  are dropped from the output;
- Poisson weights are each event's probability to land in a class, with
  edges from the total-cross-section CDF (1-f) + f * F_sim (the integer
  count at an edge split in proportion, so classes cover exactly their
  percentile range); rows are NOT
  renormalized, so a class's summed weight is n_ref x width whatever other
  classes the record lists (until November 2026 single-class records gave
  every event weight 1).

Central values are the full-sample estimates; errors and the covariance
come from a delete-block jackknife (20 blocks by default).  The jackknife
bias estimate (N−1)(θ̄ − θ) is a diagnostic (`jk_bias`; the run reports
values where it exceeds the error): for a smooth estimator it is O(1/N),
far below the error, and for a non-linear one near a singularity (v_n{4}
near c_4 = 0, ratios) the correction is unstable (zr96c2 dp0001, v2{4}
0-5%: 0.012 uncorrected, 0.199 corrected, error 0.060).  Until November
2026 the bias-corrected value θ̂ = Nθ − (N−1)θ̄ was reported.
- Blocks (`--jk-blocks`): by event id when the container has
  `meta['event_ids']` (block b: id % n_blocks == b), else contiguous in
  container order.  With common random numbers (event id k has the same
  seed at every design point) the replicates of different points then
  delete the same seeds and give the cross-point noise covariance directly,
  whatever events are missing.  The output records `jk_block_rule`;
  compute_ratios warns when its two sides differ.

## Conventions and traps (each was a real bug once)

- **Output layout**:
  - cores return harmonic-major arrays; v2 is index `harmonic - 1` for
    multi-harmonic cores;
  - single-row cores index by bin (`pipeline_dispatch.SINGLE_ROW_KEYS`,
    shared by run_pipeline and the toy harness);
  - differential cores are `(h-1)*n_bins + b`, all of them (until November
    2026 the 2D `vn2_single_diff_pT` core emitted `h*n_bins + b`, so
    run_pipeline returned v_{n-1}(pT); the toy harness followed the core).

  `run_pipeline` asserts labels == values before writing; keep that check.
- **3D spectra and grid cells**: measurement bins need not follow the grid.
  Spectra count cut cells by overlap fraction (`QGridData.acceptance_fractional`:
  uniform in eta, uniform in pT^2, i.e. constant invariant yield).
  `acceptance()` snaps to whole cells; it is fine for ratios like v_n but
  not for yields, which use the fractional overlap too (dN/deta windows
  off the grid, e.g. shifted p+Pb windows).  A rapidity window (dN/dy,
  no eta) takes the species' mass per pT cell (`acceptance_rapidity`);
  a mixture (charged) gives NaN.  Until November 2026 it gave 0 on 3D.
- **Charge-scaled 3D grids**: normalized observables (flow, correlations,
  mean pT and its fluctuations) are computed on unit weights
  (`pipeline_dispatch._unit_weights`); only `ABSOLUTE_KEYS` keep the
  charge factor.  Their self-correlation terms assume unit weights
  (tests/test_charge_scaled_grids.py).
- **Cumulants of a particle of interest** (`analysis/correlators.py`):
  `vn2_single`, `vn4_single` and their pT/eta-differential keys correlate
  the particle of interest (record species and acceptance; each
  differential bin narrows it) with the charged reference (record
  acceptance; pT from `flow_reference.pT` when given).  Sums over distinct
  particles come from inclusion-exclusion over set partitions, valid for a
  POI inside, partly inside or outside the reference and for the 2D
  Jacobian weights.  v'{2} = d2/sqrt(c2), v'{4} = -d4/(-c4)^(3/4); a
  charged record gives v{2}, v{4}.  The 3D prepares get the unselected
  container (`selects_species`).  Until November 2026: 2D ignored the
  species, v_n{4}(pT) was (v' v^3)^(1/4) with an unbalanced four-particle
  term, v_n{4} ignored the record's pT window, and 3D v_n{2}(pT)
  correlated each bin with itself.
  v_n{4} needs Q_{2n} only in weight-squared self-correlation terms
  (relative order 1/M_eff, M_eff = (sum w)^2/sum w^2); a container that
  does not store it drops them with a printed note (v3{4} on 2D JETSCAPE
  containers, n <= 5, M_eff ~ 2e4: < 1% of c3{4}).  SC(m,n), a small
  difference, refuses a missing Q_{m+n} instead.
- **Symmetric cumulants** (`sc_mn`, `analysis.correlators.sc_arrays`):
  SC(m,n) = <<4>>_{m,n,-m,-n} - <<2>>_m <<2>>_n over distinct particles
  (2D Jacobian weights or 3D unit weights), event weights the numbers of
  4-tuples and pairs; needs Q_{m+n} (the prepare raises otherwise).
  tests/test_toy_mc_sc.py.  Until November 2026 it used |Q_n|^2/M^2 (no
  self-correlation removal), ignored the pT window and the 2D weights,
  never normalized, and read SC of three harmonics as SC of the first two.
- **Scalar product / event plane, differential** (`vn2_sp_diff_*`, which
  also serve event_plane records that declare a `flow_reference`): the
  POI is the record species per bin; the reference is charged hadrons in
  the `flow_reference` windows (mirrored or segmented per bin; default the
  record window), pT from the reference block.  |Q_ref|^2 loses Sum w^2
  (`dQ0_ref`), Re<q_poi Q_ref*> the particles in both samples (2D:
  Sum w_poi w_ref; 3D: the POI count inside the reference, zero for a POI
  outside it).  `selects_species`; tests/test_toy_mc_sp_poi.py.  Until
  November 2026 2D ignored the species and the reference and subtracted
  Sum w (charged v2(pT) ~1.6x too high, or ~1e30), and 3D used an
  identified-only reference (identified records gave 0) and kept the
  POI's own particles for a reference overlapping it (STAR TPC halves).
  The integrated `vn2_sp` uses the differential prepares with one bin
  (the record's pT) and keeps its normalization; until November 2026 its
  3D prepare took all pT and ignored the species and `flow_reference`,
  and its 2D reference took the record's pT.
  A forward-detector reference (FCal, HF, BBC, EPD, RXN, Octagon) states
  `pT: {min: 0.0, max: 100.0}`; without it the reference takes the
  record's pT (until November 2026 ATLAS v_n(eta) at 4 < pT < 8 GeV had a
  near-empty FCal reference).  A non-positive reference power, or a bin
  without POI particles, gives NaN.
- **Charge factors**:
  - 2D JETSCAPE containers hold positive charges only, so charged
    quantities take ×2, and self-correlation terms take ×CF², consistently;
  - containers with `pdg_ids`, and 3D species grids (`MultiSpeciesGrid`),
    resolve exact charges; a positives-only set gives pbar from p by
    charge symmetry (until November 2026 3D summed both charges and
    then applied x2);
  - `QGridData.positives_only=False` for grids that already hold all
    charged hadrons.
- **Centrality order**: the sort is by descending multiplicity. The first
  bin starts at the record's lower edge, not 0.
- **Integrated records never get differential bins**:
  `extract_differential_bins` returns none for `dimensionality: integrated`.
- **Experimental scales**: percent columns (units `%`, a `(%)` in the
  column name, or `*100`) and MeV headers are normalized in
  `load_experimental`.  A record's `value_scale` (required by
  validate_metadata for pre-scaled headers) replaces the percent guess,
  so a column is never scaled twice.
- **Rapidity with an assumed mass** (`kinematic_range.y.mass`, e.g. PHOBOS
  y_π): the window is converted per species and pT (2D:
  `rapidity_window_extent`) or per pT cell (3D: `acceptance_rapidity`).
  It is implemented for spectra only; the dispatcher refuses it elsewhere.
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
