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
  - `check_sim_package.py`, `run_campaign.py`, `collect_campaign.py` (runs x
    observables, plus a per-observable model-precision summary and a
    selection for emulator training, `analysis/observable_selection.py`:
    whole observables only, values unchanged).
  - `compute_ratios.py`: observable_ratios records from two simulation sets (or one, within a system)
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

Every paper in data/ was vetted against the paper (November 2026; checklist
in the private workbench).  `scripts/audit_metadata.py` compares each record
with its HEPData table (reaction, energy, centrality, windows, bins, header),
and `tests/test_audit_metadata.py` requires zero errors over data/: a
deliberate departure from HEPData's own labels (a mislabelled energy or
class, checked against the paper's figures) is stated in the record as
`hepdata_corrections` {energy | centrality | pT: reason}, which the audit
reports as a warning.  Warnings are informational.

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
- Directed flow (`harmonic: 1`): rapidity-odd v1 needs `flow_reference.odd`
  (an eta-odd reference) and `sign_convention`: computed by
  `v1_odd_sp_diff_eta` (3D only).  Rapidity-even v1 declares `v1_even`
  (`fit: factorization`): the experiments fit v11(pTa, pTb) = v1(pTa)
  v1(pTb) - K pTa pTb, K from global momentum conservation, which the
  models lack, so the model's pair matrix (distinct pairs; on 3D only pairs
  beyond the record's `eta_gap`, `pair_gap_fractions`) is factorized without K
  (`v1_even_factorization`, analysis/v1_even.py: pair-weighted rank-1 fit
  on `fit_pT_bins`, default the record's bins, other bins projected on
  it).  The sign is open: `sign_convention` {pT window, sign}, or
  `absolute: true` for a tabulated |v1|.  vs centrality or pT, 2D and 3D;
  tests/test_toy_mc_v1_even.py.  Otherwise refused: the even-harmonic
  machinery would return ~0.
- `kinematic_range.y.mass`: a rapidity window with one assumed mass (PHOBOS
  and ATLAS y_pi): spectra, yields (a window, or per y bin) and <pT>.
- Per-bin yields and <pT>: eta bins (`dimensionality: differential_eta`)
  or y bins (`differential_y`; per-bin yields need no dimensionality),
  all bins of a record in one pass (`pipeline_dispatch._linear_sums_2d` /
  `_linear_sums_3d`: the per-particle weights of every bin, one matrix
  product; a pT cell the window cuts: below).  Per-bin yields compute on 2D too (until November 2026: 3D
  only); <pT> per bin until then returned the integrated value under
  per-bin labels.  A bin without particles gives NaN;
  tests/test_toy_mc_bins_ypi.py.
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
- `errors_excluded`: error labels load_experimental does not add: contained
  in another component (ALICE 'syst. uncorr.' inside 'syst.'), or not
  carried by the record's quantity (ATLAS 2601282 'syst, TAA': spectra
  tabulated over <T_AA>, multiplied back by value_scale = 1/<T_AA> into
  per-event yields, which have no <T_AA> uncertainty); '<label>#<k>' names
  the k-th component of that label (ALICE 1244523: 'sys' twice, the total
  and its multiplicity-uncorrelated part, 'sys#2').
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
  `#<doc>` for multi-document files); `tests/test_ratio_links.py` checks every link
  against `numerator_system` / `denominator_system` (the nuclei, and the energy
  when the side names one: energy ratios, `ratio_type: energy_ratio`, 'Pb+Pb
  5.02 TeV' over 'Pb+Pb 2.76 TeV', the two sides on two packages).  Each side
  is computed at the ratio's centrality classes (`ratios.at_ratio_classes`:
  ALICE's ratio tables are binned otherwise than its v_n tables; table_rows
  dropped, a point-valued side refused, combined_from must tile) and, when
  differential, in the ratio's bins (`at_ratio_bins`: binning keys replaced,
  the side's min/max kept, since they may set the reference); compute_ratios
  aligns the ratio's points per class and bin (spectra and per-eta yields,
  which carry no `dimensionality`, count as differential: the dispatcher's
  `extract_differential_bins`).  Ratios within one system
  (`ratio_type: centrality_ratio`, each class over `denominator_class`;
  `rapidity_ratio`, two windows at one class; ATLAS 3170344) run on one set,
  once, every replicate paired with itself; `side_normalization` {pT: {min,
  max}} divides each spectrum side by its yield over that range in every
  replicate (`ratios.side_normalized`; weights (hi^2 - lo^2)/2, exact for
  the pipeline's invariant yield; a bin across a limit is refused).  ATLAS's
  normalization is the fiducial 0.27-5 GeV yield although its bins start at
  0.21 GeV (its ratio tables reproduced to 0.1%);
  tests/test_ratios.py.  Particle ratios within one system link two
  species' records the same way (ALICE 1244523 K/pi, p/pi vs pT and
  integrated); a ratio's `scale` multiplies it (Lambda/K0S =
  (Lambda + Lambdabar) / (2 K0S): 0.5).
- Flow-[pT] correlations (`observable_type: flow_pT_correlations`, key
  `rho_vn2_pT`, analysis/flow_pt.py): value_type `rho_n`, rho(v_n^2,
  [pT]) = cov_n / sqrt(var(v_n^2) c_k), or `cov_n` (GeV).  The record
  states each term's windows: `covariance_method` standard | two_subevent
  (flow pair from A and C, the pT particle from A or C) | three_subevent
  (pT particle from B) | combined_subevent (the mean of two and three);
  `variance_terms` {c4, c2}: standard (the full window) or two_subevent
  (A|C), var = c_n{4} + c_n{2}^2; `ck_window` full | A | B | C;
  subevents A, B, C (`particles: ABC`).  ALICE 1966120: three subevents,
  var and c_k from subevents; ATLAS 2075412: c4 standard, c2 two-subevent,
  c_k full, narrow Sigma E_T bins (`multiplicity_baseline`: [pT] in cov and
  c_k, and the v_n^2 terms of var, at fixed multiplicity).  Every term is a
  sum over distinct particles: set-partition inclusion-exclusion with one
  window, harmonic and pT power per member (`flow_pt.CellSums`, weights a
  window share times 1/nsamples per particle), so coincidences vanish for
  disjoint 3D windows and are removed exactly on 2D; the pT-weighted flow
  vector of a coincidence term is the cell's flow vector times its mean
  pT.  Signed root of var x c_k.  tests/test_toy_mc_flow_pt.py: three-
  subevent cov exact against explicit particle triplets (others within
  0.1%), the injected rho on 2D and 3D, zero without it, oversampled
  events, the baseline.  ATLAS's points carry no class edges: classes run
  midway between the tabulated mean centralities.  rho(v_m^2, v_n^2,
  [pT]) (ALICE Table 4) is not computed.
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
covariance (or, with `--covariance replicates`, the delete-one-block
replicates that give it exactly: `analysis.jackknife.covariance_from_replicates`;
6788 observables: 370 MB per run as a matrix, 1 MB as replicates), labels
and the experimental points (also aligned with the labels).

Records of one compute key that differ only in their centrality classes and
labels (one record per class of a table: 3414 of the 6020 dispatchable
public records reuse an earlier one's) share one prepared data dict
(`run_pipeline.prepare_key`: every measurement param but those in
`_PREPARE_KEY_SKIP`, which no prepare reads; a new param that a prepare
reads must stay out of that set), masked once per replicate
(`masked_entries`).  Cores must not modify their data dict.  Until November
2026 each record prepared and masked its own (ATLAS O+O, 8k synthetic
events: 49.8 s -> 19.1 s, 1 GB less held); tests/test_shared_prepares.py.  A
single run uses numpy's BLAS threads, which spin without gain on these sums
(the same O+O run: 2100% CPU, system time 4x user time; one thread,
`OPENBLAS_NUM_THREADS=1`, 16 s at 100%), and numpy madvises transparent huge
pages on large arrays: with THP defrag `madvise` and a full page cache every
page fault compacts memory directly (40k O+O: memory pressure 69.5% with 90
GB available, where systemd-oomd kills at 50%, taking tmux panes with it;
`NUMPY_MADVISE_HUGEPAGE=0`: 0.00, and 13.5 -> 6.5 min with the shared
prepares).  run_campaign sets both for its jobs; set them for single runs.

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
  sums per-species grids.  A species grid may be binned in rapidity
  (`y_edges`, `y_mass`; O+O JETSCAPE-framework QnVector output):
  `QGridData.window_weights` maps eta windows per pT cell with the mass
  (fractional for yields, spectra, <pT>, E_T and POI sums via
  `poi_acceptance`; whole cells by mapped centre in `acceptance()` for
  unit-weight sums), `shared()` gives the expected POI-reference
  self-pairs, grids of different masses are not summed, and a y grid is
  never a reference (tests/test_toy_mc_rapidity_grid.py).  Qn may be
  complex64; window sums accumulate in double precision, from a lazily
  built summed-area table (`summed_area_tables`; ~2x the complex64 Qn in
  memory, keyed to the Qn object; tests/test_qgrid_window_sums.py).
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
  every event weight 1);
- the class edges, and a record's `centrality.normalization`, are estimated
  from the simulated events, so each jackknife replicate rebuilds them from
  the events it keeps (`weights.CentralityWeights`, one per estimator
  multiplicity, shared by its records; the anchor's f is external and
  stays).  Until November 2026 the weights were fixed from the full
  sample: <N> per class came out with 2-4x too small an error (toy: 3-5x
  at 5-60%), flow ~1.0-1.1x (found by the isobar analysis, zr96c2).
  Errors and covariances change, values do not.  With sharp edges
  (`--cent-method simple`) a replicate also re-ranks a few kept events
  across each edge, so the error of a quantity whose per-event noise is
  uncorrelated with the estimator (v_n) is conservative by ~1.2 (toy, 5-10%
  classes of 1500 events); the Poisson weights (the default) are smooth
  and calibrated (tests/test_jackknife_centrality.py).  The Poisson edges
  are searched from the full sample's crossings, and each edge is computed
  once per sample (2.1 s -> 0.5 s for 40k events and 16 classes, 0.2 s per
  replicate).

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
  uniform in eta, uniform in pT^2, i.e. constant invariant yield), all pT
  bins of a window in one pass (`QGridData.spectrum_counts`, equal to the
  per-bin sums; one pass per bin took 6 min of a 40k-event O+O run).  A
  cell that a bin covers only in part is shared among its bins by a local
  model of the class-averaged spectrum, exp(s m_T) times the window's
  dy/deta Jacobian, slopes fitted to the neighbouring cells
  (`analysis/spectra_shape.py`, in the spectra core: per class, inside the
  jackknife; 2D and 3D).  Every cell keeps its count, and bins of whole
  cells are unchanged.  Until November 2026 the share was flat in pT^2 (3D)
  or pT (2D), and three ATLAS bins inside one 0.2 GeV cell gave sim/data
  0.82, 1.01, 1.39; tests/test_toy_mc_spectra_subcell.py.
  `acceptance()` snaps to whole cells; it is fine for ratios like v_n but
  not for yields, which use the fractional overlap too (dN/deta windows
  off the grid, e.g. shifted p+Pb windows).  A rapidity window (dN/dy,
  no eta) takes the species' mass per pT cell (`window_weights`);
  a mixture (charged) gives NaN.  Until November 2026 it gave 0 on 3D;
  <pT> in a rapidity window (identified species) gave 0/0 until then.
- **pT windows that cut a cell** (yields and <pT>, integrated or per bin,
  2D and 3D; ATLAS O+O pT > 0.27 GeV on 0.2 GeV 3D cells and 0.05 GeV 2D
  cells): `_linear_sums_2d` / `_linear_sums_3d` sum each window with the
  cut cell at its flat share (uniform in pT^2) and attach a `pt_subcell`
  block; the core (`_pt_subcell_corrected`) replaces the flat share by the
  share of the spectra model above (`spectra_shape.window_shares`, all
  windows at once; the pT-weighted share for sums of pT), per class and
  jackknife replicate.  The share of a density applies to the whole cell's
  count in the window: in a y window mapped onto an eta grid the count in
  the window mapped at the part's pT carried <dy/deta>_cell /
  <dy/deta>_part (-0.6% at y_pi pT > 0.27).  Toy, 0.2 GeV cells: 3D eta
  windows within 0.1% of the particle level, y windows within 0.5% (each
  cell's window mapped at its centre, as with whole cells); 2D within
  0.35% on 0.05 GeV cells.  Until November 2026 an integrated 3D <pT> and
  every 2D yield and <pT> snapped the window to whole cells (pT > 0.27 on
  0.2 GeV cells: <pT> from 0.4 GeV, +16%; 2D yields -24%), and 3D yields
  and per-bin <pT> kept the flat share (+-3%);
  tests/test_toy_mc_pt_cut_cells.py.  dE_T/deta takes the flat share on
  3D and whole cells on 2D (no record cuts a cell); pT fluctuations take
  whole cells on 3D (unit weights) and on 2D the flat share of a cut cell,
  as below.
- **pT fluctuations** (`mean_pT_fluctuations`, key pT_fluct; value_type
  `sqrt_<dpt1_dpt2>_over_<pT>`, sqrt(<Delta pT_i Delta pT_j>)/<<pT>>, or
  `<dpt1_dpt2>`, the covariance in GeV^2 (a (MeV/c)^2 header loads as
  GeV^2); ALICE 1307102, STAR 681688, 1216565, 1712047, 3141716;
  `fluctuations` is for conserved-charge cumulants and stays
  undispatched): each particle carries its probability to be in the
  record's window (2D: the eta-window Jacobian over the flat-y plateau
  times the pT-cell share; 3D: the overlap share of a cut cell), and the
  distinct pairs come from the second-weight sums (Σa², Σa² pT, Σa² pT²;
  `compute_pT_fluct`).  `averaging`: `pairs` (ALICE, the default: all
  pairs of the class pooled, deviations from the mean pT of all particles)
  or `events` (STAR: the mean of the events' pair averages, deviations from
  and normalization by the mean of the event means; they differ when the
  multiplicity varies across the class).  `multiplicity_baseline` {eta,
  pT}: deviations from the mean pT at the event's multiplicity in those
  windows (a linear regression within the class), STAR's <<pT>> fitted vs
  N_ch or computed per reference-multiplicity bin; the normalization stays
  the class mean.  A container event of `nsamples` particlizations
  (JETSCAPE 2D: median 82) gives each particle weight a/nsamples, so the
  self-pair sums take 1/nsamples^2.  Until November 2026 they took
  1/nsamples (JETSCAPE Au+Au 200: sqrt(C)/<<pT>> from -0.01 to -0.08, which
  the clipped root hid as 0; now 0.7-1.1 x STAR, and 0.84-1.04 x ALICE
  Pb-Pb 2.76 TeV), an eta window gave NaN on 2D, dy mode ignored the pT
  window, 3D took whole cells (and an upper limit on a cell edge the next
  cell too: pT 0.15-2.0 on 0.2 GeV cells summed 0.2-2.2), and STAR records
  were computed with ALICE's averaging and the class mean;
  tests/test_toy_mc_fluct_et.py (2D eta windows within 2-6% of the brute
  force, 3D exact on cell edges and within 3% across cut cells, oversampled
  events).
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
  difference, refuses a missing Q_{m+n} instead.  v_n{6} and v_n{8} are
  exact weighted distinct-particle sums on 2D and 3D
  (`correlators.reference_cumulant_arrays`; set partitions aggregated by
  block type), every order weighted by its tuple count; harmonics beyond
  the stored ones enter only coincidence terms and are dropped with a note.
  Until November 2026 the 2D cores required an integer multiplicity, which
  the weighted M never is, so <6> and <8> were 0 (v{6} = (3/4)^(1/6) v
  without fluctuations, either sign with them); the 3D core averaged <6>
  over events unweighted; and a 3D prepare without Q_3n returned None,
  crashing the group's jackknife (tests/test_toy_mc_vn68_2d.py,
  tests/test_missing_harmonics.py).
- **Combinations of cumulant orders** (`cumulant_combination`: ratio,
  difference with optional `scale`, gamma1 = -6 sqrt2 v{4}^2 (v{4} - v{6}) /
  (v{2}^2 - v{4}^2)^{3/2}; `two_particle` for a gapped v{2} term;
  `harmonics` for terms of different harmonics, v3{2}/v2{2} = orders [2, 2],
  harmonics [3, 2]): computed
  by `vn_cumulant_combination` from its terms, each the record as an
  ordinary v_n{2k} record (`dispatcher.combination_terms`), on the same
  events inside the jackknife; one value per class (`SINGLE_ROW_KEYS`); a
  zero denominator gives NaN; tests/test_toy_mc_cumulant_combination.py.
- **Pair gaps** (`eta_gap` without subevents, cumulant order 2: the
  two-particle correlation functions of ATLAS and CMS, 2 < |Delta eta| < 5;
  STAR's Delta eta > 0.05 and > 1, pair by pair): on 3D containers only
  pairs beyond the gap are correlated (`_pair_gap_arrays_3d`): the pairs of
  eta cells i, j weighted by the fraction of their particle pairs with
  |Delta eta| > gap, exact for particles uniform within cells
  (`QGridData.pair_gap_fractions`), self-pairs removed; per-eta-cell sums
  over each window's own cells (`_row_sums`), F applied once, and the
  per-event sums cached on the container by (species, windows, bins, gap):
  they do not depend on the class.  On 2D containers
  the gap only removes self-pairs, which the distinct sums do.  Until
  November 2026 the gap was not applied on 3D.  A gap where the paper
  describes subevents either side of mid-rapidity is curated as
  `particles: AB` instead.  No reference pairs gives NaN;
  tests/test_toy_mc_pair_gap.py.
- **Two-subevent v_n{2} on 3D grids** (`prepare_vn2_gap_3d`): a subevent
  window that cuts a cell takes it with its overlap fraction (each subevent
  sum enters linearly), and cells inside both windows give their expected
  self-pairs sum f_A f_B Q0; windows on cell edges are unchanged.  ALICE's
  |Delta eta| > 1.4 subevents (0.1 wide) on a 0.2-wide O+O grid gave 0
  until November 2026 (edge snapping: one window a whole cell, the other
  none).  No pairs (an empty subevent) gives NaN, not 0;
  tests/test_toy_mc_gap_subcell.py.
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
  near-empty FCal reference).  The reference amplitude is a signed root of
  its power (below); an empty reference or a bin without POI particles
  gives NaN.
- **Signed roots**: every v_n from a cumulant or a reference power keeps
  its sign: v{2} = sign(c2) sqrt|c2|, v{4} = sign(-c4) |c4|^(1/4), v{6} =
  sign(c6) |c6/4|^(1/6), v{8} = sign(-c8) |c8/33|^(1/8)
  (`correlators.vn_from_cumulant`); a POI flow divides by the signed
  reference (v'{2} = d2/v{2}, v'{4} = -d4/v{4}^3), a scalar-product or
  amplitude estimator by sign(P) sqrt|P|; likewise sqrt(<Delta pT_i Delta
  pT_j>)/<<pT>> and the radial-flow v0 (until November 2026 clipped to 0
  and NaN).  Invertible (c = sign(v)|v|^k x
  norm), no NaN, and a jackknife replicate whose cumulant crosses zero is
  kept (a NaN replicate is replaced by the full-sample value, shrinking the
  error).  Near c = 0 the root's slope diverges, so errors on v there are
  non-Gaussian.  Until November 2026 v'{4} dropped the sign of -c4 and a
  non-positive reference power or amplitude gave NaN;
  tests/test_signed_roots.py.  (The JETSCAPE replica vn2_single_mor keeps
  JETSCAPE's clipping.)
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
  `rapidity_window_extent`) or per pT cell (3D: `acceptance_rapidity`,
  `window_weights(var='y', mass=...)`).  Implemented for spectra, yields
  and <pT> (`dispatcher.MASS_ASSUMED_RAPIDITY_KEYS`); refused elsewhere.
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
