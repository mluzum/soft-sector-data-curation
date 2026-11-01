# soft-sector-data-curation

A curated, machine-readable database of soft-sector measurements in
heavy-ion collisions, and a model-independent pipeline that computes the
same observables from any hydrodynamic or transport simulation.

Each curated record states how its measurement was made, not only its
values:
- the η and pT acceptance, and subevents and gaps;
- the centrality estimator and its acceptance;
- the method (cumulant order, scalar product, event plane) and the species.

The pipeline reads those fields and applies the record's exact cuts to the
simulation. A model is then compared with the data on the same footing,
record by record, for any selection of measurements.

- **3,676 records** from 202 publications (STAR, PHENIX, BRAHMS, PHOBOS,
  ALICE, ATLAS, CMS, LHCb), built from the HEPData tables in `hepdata_raw/`.
  **Every record can be computed from a simulation and compared with its
  data by the tools here**: it dispatches to a computation, its centrality
  classes are percentile bins, and its experimental values load
  (`tests/test_public_records.py` enforces this).  The point of the
  database is that automation; the bare numbers are already on HEPData.
  Measurements that cannot be computed yet (observable types without a
  computation, minimum-bias or p+p records, tables without a reader) are
  curated separately and join as they become computable.
- **Observables computed from simulations:**
  - flow: v_n{2,4,6,8}, with η-gap and subevent methods, and pT- and
    η-differential;
  - correlations: scalar-product and event-plane estimators, symmetric
    cumulants and normalized correlations;
  - identified and charged yields, spectra, mean pT and dE_T/dη;
  - ratios between collision systems (e.g. Ru+Ru / Zr+Zr) from two
    simulation sets.
  - Errors are a block jackknife with a full covariance matrix across
    observables.
- **Any simulation:** a simulation enters through one of two in-memory
  containers, boost-invariant (2D) or with explicit pseudorapidity (3D).
  A simulation package plus a small adapter supplies them
  (see [Simulations](#simulations)).

## Installation

```bash
git clone https://github.com/mluzum/soft-sector-data-curation
cd soft-sector-data-curation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, h5py, PyYAML (Python >= 3.10)
bash tests/run_all.sh                    # toy-MC validation suite, no external data needed
```

Run scripts from the repository root.

## The database

```
observables/        schema per observable type (+ _common.yaml, centrality.yaml)
data/<type>/        curated records, one YAML file per HEPData table (multi-document)
hepdata_raw/        raw HEPData downloads by INSPIRE id, plus PHOBOS tables
hepdata_database.yaml   catalog of the curated records
curation/           curation decisions shared by several records
```

```bash
python scripts/validate.py data/                                 # schema check of every record
python scripts/search_records.py --type flow_vn --reaction "Pb Pb" --count-only
python scripts/search_records.py --type flow_vn --harmonic 2 --output manifest.yaml
python scripts/export_for_analysis.py --type flow_vn             # records + raw values as JSON
```

Records kept outside this repository (preliminary results, records being
vetted) use the same layout and tools: pass their directory with
`--extra-records DIR` (or `$SOFT_SECTOR_EXTRA_RECORDS`).  A record without
a HEPData table carries its own HEPData-format `value_table`, and records
marked `status: preliminary` are only computed with `--include-preliminary`.

Each record gives its HEPData source (`inspire_id`, `version`,
`hepdata_file`, `value_type`). The schemas in `observables/` define both
validation and the fields a computation needs; for example, flow records
carry `method`, `cumulant_order`, `particles` and `subevents`. New
measurements are added with `scripts/download_hepdata.py` and
`scripts/classify_tables.py` (draft candidates), then curated and checked
with `scripts/validate.py` and `scripts/validate_metadata.py`. Data without
a HEPData record go through `scripts/import_manual.py`.

## Simulations

The pipeline never reads simulation files. It works on one of two containers
(`analysis/qvector.py`):

- **`QVectorData`** (2D, boost-invariant): Q-vectors per event, species,
  harmonic and pT bin, flat in rapidity. η windows are applied through the
  exact dy/dη Jacobian.
- **`QGridData`** (3D): Q-vectors on an η × pT grid. This supports η-gap
  methods with genuine longitudinal decorrelation.

Containers can be filled in three ways:
- from particle lists (`analysis/converters.py`);
- from the JETSCAPE formats, with the built-in `--sim 2d|3d` readers;
- through a **simulation package**: a small directory shipped with the data,
  whose `simulation.yaml` names an adapter module that reads the format.
  Adapters live in their own repositories, such as
  [soft-sector-sim-adapters](https://github.com/mluzum/soft-sector-sim-adapters).

```bash
python scripts/check_sim_package.py <package>                  # contract check + summary
python scripts/run_pipeline.py --sim-package <package> --design-point 5 --variant CE \
    --type flow_vn particle_yields --reaction '^Ru\+Ru$' --energy 200 --output dp0005.h5
python scripts/run_campaign.py --sim-package <package> --out-dir runs/ --jobs 8
python scripts/collect_campaign.py runs/ --output table.h5     # runs x observables for emulation
python scripts/compute_ratios.py --num-package <package> --num-design-point 12 \
    --den-package <package> --den-design-point 40 --output ratios.h5   # e.g. Ru+Ru / Zr+Zr
```

[`docs/sim_interface.md`](docs/sim_interface.md) documents the containers,
the package manifest and the adapter API.

The pipeline writes one HDF5 file per run. For each collision system it holds:
- the values, errors and jackknife covariance, with labels;
- the experimental points of every record;
- the same points aligned with the labels.

Centrality follows the record's estimator. It can be anchored to the total
cross section when the simulation covers only part of it (`--cent-anchor`,
or the adapter's `anchor()`).

## Validation

[`docs/validation.md`](docs/validation.md) covers the toy-MC tests against
particle-level truth and the cross-checks against independent JETSCAPE
analyses (2D and 3D), with their accuracies and known method systematics.

## License and data

The code is released under the MIT license (`LICENSE`). The curated
records (`data/`, `hepdata_database.yaml`, `curation/`) are released under
CC BY 4.0 (`LICENSE-DATA`): reuse them freely, with credit. The raw tables
keep their own terms:
- `hepdata_raw/records/` are downloads from [HEPData](https://www.hepdata.net);
- `hepdata_raw/phobos/` are the PHOBOS collaboration's public data tables;
- `data_validation/jetscape_bayes/exp_data/` comes from
  [hic-bayes](https://github.com/IFUSP-Heavy-Ions/hic-bayes).

When you use a measurement, cite the original publication: every record
gives its INSPIRE id. To cite this repository, see `CITATION.cff`.
