# Simulation input interface

How to plug **any** simulation code into the analysis pipeline.

The pipeline never reads simulation files directly.  Everything downstream —
dispatch, kinematic cuts from curated records, centrality weights, cumulant
cores, jackknife — operates on one of two in-memory containers.  Getting your
sim data into a container is the only integration step.

The way to do that for data you keep or distribute is a **simulation
package** (§4): a small directory shipped with the data, whose manifest
names an adapter module (maintained outside this repository) that returns
the container arrays.  `run_pipeline.py --sim-package <dir>` then needs no
configuration in this repository.

---

## 1. The two container contracts

### QVectorData — "2D" boost-invariant  (`analysis/qvector.py`)

| field | shape | meaning |
|-------|-------|---------|
| `Qn` | `(n_events, n_species, n_harm, n_pt)` complex | slot 0 = counts Q₀; slots 1..n = Σ e^{inφ}; accumulated per event/species/pT-bin |
| `pt_bin_edges` | `(n_pt+1,)` | pT grid (any edges; production JETSCAPE uses `PTCUTS`) |
| `masses`, `species_names` | `(n_species,)` | species table; row order defines species indices |
| `nsamples` | `(n_events,)` | oversample count per event (1 if none) |
| `rap_max` | float | half-width of the rapidity plateau |
| `sum_pt` *(opt.)* | `(n_events, n_species, n_pt)` | Σ pT per cell — enables mean-pT |

**Assumption:** particles are flat in *rapidity* over `|y| < rap_max`.
η acceptances from experimental records are applied analytically through the
Jacobian dy/dη.  Use this container when your sim is boost-invariant or you
only trust its rapidity distribution.

### QGridData — "3D" explicit pseudorapidity

| field | shape | meaning |
|-------|-------|---------|
| `Qn` | `(n_events, n_eta, n_pt, n_harm)` complex | same slot convention, binned on an η×pT grid |
| `eta_edges`, `pt_edges` | | rectangular grid edges |
| `cent_mult` | `(n_events,)` | per-event multiplicity for centrality sorting |
| `nsamples` | `(n_events,)` | oversampling (divides counts) |
| `sum_pt` *(opt.)* | `(n_events, n_eta, n_pt)` | Σ pT per cell |

**Assumptions:** unit-weighted particles (dQ0 = Q0); disjoint η subevents have
zero self-correlation (this is what makes longitudinal-decorrelation studies
possible).  No boost-invariance assumed.

---

## 2. Universal input: particle lists

Any sim that can dump per-event particle records integrates in ~10 lines:

```python
events = [dict(pt=..., phi=..., species_idx=..., eta=...),   # per event
          ...]        # 'y' also accepted; the other coordinate is derived

from analysis.converters import (
    particles_to_qvector_data, particles_to_qgrid_data)

qv = particles_to_qvector_data(
    events,
    pt_edges=np.arange(0, 10.01, 0.2),   # your grid
    rap_max=1.0,                          # your plateau half-width
    species_names=['pion', 'kaon', 'proton'],
    masses=[0.13957, 0.49368, 0.93827],
)

grid = particles_to_qgrid_data(
    events,
    eta_edges=np.linspace(-7, 7, 141),
    pt_edges=np.geomspace(0.05, 50, 61),
)
```

Notes
- `species_idx` indexes into `masses`/row order; keep it stable across runs.
- Provide either `y` or `eta` per particle; conversion is exact
  (`y_to_eta` / `eta_to_y` round-trip at machine precision).
- Positive-charges-only samples are fine: the flow prepares apply the
  charge factor CF²-consistently (self-correlation terms included).

## 3. Native-format shortcut

If your format already stores *accumulated* Q-vectors (like JETSCAPE's
`Qns_*.npy`), skip particle lists:

```python
qv = QVectorData.from_2d_npy(qn_path, nsamples_path, pt_bin_edges,
                             rap_max=1.0)      # layout: (δf, ev, sp, harm, pT)
```

The existing loaders (`analysis/simulation_data.load_3d_grid`,
`scripts/compute_flow_vn.cut_qvectors`) serve as reference implementations
for writing a native reader.

## 4. Simulation packages

A package is a directory, usually next to the data, holding
`simulation.yaml` and whatever the adapter needs (cross-section tables, the
design, a README for people).  Implementation: `analysis/sim_package.py`;
working example: [soft-sector-sim-adapters](https://github.com/mluzum/soft-sector-sim-adapters)
(`sim_adapters/iebe_music.py` reads merged iEBE-MUSIC campaigns;
`campaigns/zr96c2/` is the package of an A=96 isobar campaign).

### Manifest (`simulation.yaml`, format 1)

```yaml
format_version: 1
name: zr96c2                      # used in output provenance
title: "..."                      # optional, for people
container: 2d                     # 2d -> QVectorData, 3d -> QGridData
data_root: ../merged              # relative to this file; --data-root overrides
adapter:
  module: sim_adapters.iebe_music # importable module implementing the API below
  repo: soft-sector-sim-adapters  # checkout directory name, used to find it
  options: {xsec_table: cross_sections_a96.csv}   # keyword args of open_simulation
systems:                          # record groups the data may be compared to
  - {reaction: Ru Ru, energy_gev: 200}
  - {reaction: Zr Zr, energy_gev: 200}
variants: [Grad22, CE, PTB]       # or null; e.g. delta f models
design: {file: design.csv, id_column: label}   # optional; id's trailing integer = design point
selection:                        # optional default record selection (run_pipeline options)
  type: [flow_vn, particle_yields, particle_spectra]
  reaction: '^(Ru\+Ru|Zr\+Zr)$'
  energy: 200
software: {...}                   # optional pins written by the adapter repo's deploy tool
```

`systems` uses the normalized reaction names of `_normalize_reaction` in
`run_pipeline.py` ("Au Au", "Pb Pb", "Ru Ru", ...); record groups of other
systems are skipped.

### Adapter API (version 1)

```python
API_VERSION = 1
def open_simulation(data_root, package_dir, **options) -> sim
sim.design_points() -> list[int]
sim.load(design_point, variant) -> dict          # container arrays
sim.anchor(design_point) -> anchor spec or None  # optional
sim.source_files(design_point) -> list[path]     # optional
```

The adapter imports numpy and its own readers, never this repository.
`load` returns the container fields of §1 as plain arrays:

- 2D: `Qn`, `pt_bin_edges` (n_pt+1), `masses`, `species_names`,
  `nsamples`, `rap_max`; optional `pdg_ids`, `sum_pt`, `sum_pt2`, `sum_mT`,
  `cent_mult`, `event_weights`;
- 3D: `Qn`, `eta_edges`, `pt_edges`, `cent_mult`, `nsamples`,
  `positives_only` (required: it sets the charge factor); optional
  `event_weights`, `sum_pt`, `sum_pt2`, `sum_ET`.  Several species:
  `{"species": {pdg: <3D dict>}, "combined": <3D dict> or None}`.

Anything else goes under `meta`.  Unknown keys are refused, as are
non-finite values, negative or complex counts, wrong edge lengths and
non-positive `nsamples`.  `anchor` returns anything `resolve_anchor`
accepts (the coverage of the loaded events in the total cross section;
§"Centrality anchor" in AGENTS.md); `--cent-anchor` overrides it.

The adapter module is searched for in this order: `--adapter-path`,
`$SIM_ADAPTERS_PATH`, `<package>/software/<adapter.repo>` (where the
package's `setup.sh` clones it), `<adapter.repo>` next to this repository's
checkout, then installed modules.

### Running

```bash
python scripts/check_sim_package.py <package>          # contract check + summary
python scripts/run_pipeline.py --sim-package <package> --design-point 5 --variant CE \
    --type flow_vn particle_yields --reaction '^Ru\+Ru$' --energy 200 --output dp0005_CE.h5
```

The output's file attrs record `sim_package_*` (name, manifest, data root,
adapter module/file/version/commit, analysis commit), `variant` and
`pipeline_argv`.  Each system group holds `values`, `errors`, `covariance`
(block jackknife over events) and `labels`, the experimental points per
record (`exp_<k>`), and the same points aligned with the labels
(`exp_values`, `exp_errors`, `label_record`; NaN where a record's points
cannot be matched one-to-one).

A whole campaign, then one table for emulation:

```bash
python scripts/run_campaign.py --sim-package <package> --out-dir runs/ --jobs 8
python scripts/collect_campaign.py runs/ --output table.h5
```

`run_campaign.py` runs one pipeline process per (design point, variant) with
the manifest's `selection`, and skips runs whose arguments, source files,
selected records and pipeline code are unchanged (the last two from
`run_pipeline.py --print-fingerprint`, stored in each output as
`records_fingerprint` / `code_fingerprint`), so rerunning it after new data
or record corrections computes only what changed.  `--ignore-code-changes`
keeps outputs made with older code.  `collect_campaign.py` writes runs × observables (`values`, `errors`),
the aligned experimental points and the design parameters.

## 5. Without a package

```bash
python scripts/run_pipeline.py \
    --type flow_vn particle_yields particle_spectra mean_pT \
    --reaction "Au Au" --energy 200 --sim 2d \
    --output results.h5
```

The built-in `--sim 2d|3d` readers (JETSCAPE `Qns_*.npy`, JETSCAPE3D HDF5)
take their paths from `config/sim_paths.yaml`; every kinematic cut,
centrality definition and method comes from the curated record side.

## 6. Validating your converter

Run the toy-MC suite against *your* container: generate events with known
flow (`analysis/toy_mc.ToyGenConfig`), push them through **your** converter
instead of `make_qvector_data`, and call `evaluate()`.  If injected v₂ is
recovered within jackknife errors, your container is wired correctly.
`tests/test_toy_mc_*.py` double as executable documentation.
