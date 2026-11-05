# Validation

What has been checked, and how far the results can be trusted.

## 0. The curated records

Every paper with records in `data/` (131 papers, 6,377 records) was checked
against the paper itself in November 2026: collision system and energy,
centrality classes and estimator, acceptance, species, method, and which
table column a record reads.  Records that could not be confirmed, or that
the pipeline cannot compute, were moved out of `data/`.  Two tests keep
the result:

- `tests/test_public_records.py`: every record dispatches, has percentile
  centrality classes, and loads its experimental values;
- `tests/test_audit_metadata.py`: `scripts/audit_metadata.py` finds no
  disagreement between a record and its HEPData table (qualifiers,
  independent variables, header).  Where HEPData's own labels are wrong
  (checked against the paper's figures), the record states the correction
  in `hepdata_corrections`.

## 1. Toy Monte Carlo: every observable against particle-level truth

`analysis/toy_mc.py` generates events with known flow, multiplicity and pT
spectra, fills the pipeline containers from the particle lists, and runs
the production path (record → dispatch → prepare → core → jackknife). The
results are compared with estimators computed directly on the particles.
The `tests/test_toy_mc_*.py` files cover:

- v_n{2} (n = 1–5), with and without η gaps and subevents (2D and 3D
  containers give the same result for single acceptances);
- v_n{4}: the fixed-v theorem v{4} = v_inj and the analytic relations for
  event-by-event fluctuating v_n;
- v_n{6} and v_n{8}: exact direct-Q correlators, checked term by term
  against brute-force enumeration;
- differential v_n(pT) and v_n(η), against pair estimators bin by bin;
- scalar-product and event-plane estimators, symmetric cumulants SC(m,n),
  and normalized correlations NAC;
- yields dN/dη and dN/dy, invariant spectra, mean pT, pT fluctuations and
  dE_T/dη, for charged hadrons and per species;
- centrality anchors (sub-sampled cross sections) and per-event weights;
- jackknife error calibration: the truth lies within 2σ in 12 of 12 seeds.
- jackknife errors with the centrality edges rebuilt per replicate: the spread
  of dN/dη and v2{2} per class over independent samples equals the mean error
  (Poisson weights; sharp edges: v2{2} conservative by ~1.2).

Known method systematics, which are kept as test tolerances:

| case | size |
|---|---|
| spectra on a binned 2D grid (1/pT at the bin centre on a falling spectrum) | ±7% per bin |
| coarse 3D grids where windows straddle bins (use `snap_window()`) | up to +20% in counts |
| Poisson centrality weights on very wide classes | biases cumulants; use `simple` for wide classes |
| dE_T/dη from 2D containers (m_T against 1/cosh η correlation inside the window) | ~4% |

## 2. JETSCAPE 2D, against an independent implementation

On the same JETSCAPE 2D simulation campaigns (design points 0 and 1),
compared with the model-side observables of D. Everett *et al.*,
PRC 103, 054904 (2021), arXiv:2011.01430:

- **Au+Au 200 GeV:**
  - v₂{2} agrees to 2–3% in 4 of 6 centrality bins (best ≈0.2%);
  - identified dN/dy to 1–6% in central collisions;
  - mean pT to 0.0–0.5% (≈2% peripheral);
  - v₃{2} to 10–15% (statistically noisier).

  The growth of the yield differences toward peripheral collisions (up to
  ~30%) comes from different class-boundary conventions, not from the
  observable computation.
- **Pb+Pb 2.76 TeV:** the most central bin agrees for all observables. The
  outer bins of the reference were assigned with a TRENTo-geometry proxy
  that cannot be reproduced from multiplicity percentiles.

Conventions fixed by this comparison:

- identified yields are single charge sign, per unit rapidity;
- v_n{2} has no η gap, |η| < 1, 0.2 < pT < 5 GeV;
- centrality is the hard percentile of the charged multiplicity in |η| < 1.

## 3. JETSCAPE 3D Bayesian calibration (Au+Au 200 GeV)

The pipeline was run through the 3D path on the JETSCAPE 3D campaign of
arXiv:2601.17234 (Mankolli *et al.*). The comparison is with that analysis's
stored model observables (model output and converted experimental tables
from [IFUSP-Heavy-Ions/hic-bayes](https://github.com/IFUSP-Heavy-Ions/hic-bayes)),
matched record by record, centrality class by class and bin by bin. The
matching layer is `data_validation/jetscape_bayes/compare_postproc.py`,
tested by `tests/test_postproc_matching.py`. Many of the matched records
(the PHOBOS tables among them) are curated outside this repository until
they compute end to end; `--extra-records` points the matching layer at
them, and it checks every inventory row against the document it names. At
design point 0, over 3.7 million comparisons:

- PHOBOS dN/dη (3.4 million comparisons): median difference from the model
  −0.15%.
- v₂: the remaining differences trace to definitions, not computation:
  - the model's centrality classes differ from the STAR reference
    multiplicity at 0–5%;
  - at |η| > 3 there is a known tension in STAR v₂(η), reproduced as in
    the original analysis.

This comparison found two production bugs, both fixed and covered by tests:

- label and value misalignment for single-row cores;
- a doubled charge factor on JETSCAPE3D grids that already hold all
  charged hadrons.

## 4. Simulation packages

An adapter returns plain arrays, and `analysis.sim_package.build_container`
validates them. It rejects:

- unknown fields;
- non-finite values;
- negative or complex counts;
- wrong bin-edge lengths;
- non-positive oversampling.

`tests/test_sim_package.py` runs the pipeline through a toy package and
through the built-in `--sim 2d` reader on the same Q-vectors, and requires
identical labels and values. It also tests the campaign driver's
run, skip and rerun cycle and the collected table.

## 5. Reproducibility

Pipeline outputs reproduce run to run to about 1e-16 relative, not bit for
bit. numpy's SIMD reductions sum in an order that depends on the memory
alignment of each array. Compare outputs with a relative tolerance (1e-12
is ample), and compare containers exactly.
