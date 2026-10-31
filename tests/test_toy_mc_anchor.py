"""Anchor-aware toy-MC test: sim sets covering only part of the cross section.

Mirrors the production path (run_pipeline / compute_flow_vn) at the toy
level: a sim set that physically covers only a fraction f of the total
cross section should (a) recover the correct observable in covered
centrality bins and (b) exclude (NaN) peripheral bins beyond the anchor.

The anchor machinery clips each bin's upper percentile edge to anchor_cent
(= 100*f).  Bins whose upper edge lies beyond the anchor become zero-width
(no events selected) and must be excluded.  With f=0.5 (anchor_cent=50),
any centrality class with hi > 50% is excluded.
"""


def _run():
    import numpy as np
    import sys
    sys.path.insert(0, '.')
    from analysis.toy_mc import (ToyGenConfig, generate_events,
                                 make_qvector_data, record_flow_vn2, evaluate)

    results = []
    def check(name, cond):
        results.append((name, bool(cond)))
        print(("  PASS " if cond else "  FAIL ") + name)

    def build_qv(n_events, seed=7):
        cfg = ToyGenConfig(n_events=n_events,
                           flow={2: dict(kind='const', value=0.05)},
                           mult_mean=300, mult_sigma=40, seed=seed)
        events, truth = generate_events(cfg)
        return make_qvector_data(events, truth), len(events)

    # Bins spanning the anchor: 0-25, 25-50 covered; 50-75, 75-100 excluded.
    rec = record_flow_vn2(cent_classes=('0-25%', '25-50%', '50-75%', '75-100%'))

    # 1) Default anchor (full set) — regression: all bins valid, v2 = 0.05.
    qv, n = build_qv(800)
    vals, errs = evaluate(rec, qv=qv, cent_method='simple')
    check('default all finite', np.all(np.isfinite(vals)))
    check('default v2~0.05', np.allclose(vals, 0.05, atol=0.01))

    # 2) Anchor f=0.5 (anchor_cent=50): top two bins zero-width -> NaN.
    #    Raw output so the NaN-exclusion is visible (run_pipeline drops these
    #    bins; the JK guard would otherwise salvage them).
    vals_t, _ = evaluate(rec, qv=qv, cent_method='simple', cent_anchor=0.5,
                         with_jackknife=False)
    check('covered bin0 finite', np.isfinite(vals_t[0]))
    check('covered bin0 ~0.05', abs(vals_t[0] - 0.05) < 0.01)
    check('covered bin1 finite', np.isfinite(vals_t[1]))
    check('covered bin1 ~0.05', abs(vals_t[1] - 0.05) < 0.01)
    check('beyond-anchor bin2 NaN', not np.isfinite(vals_t[2]))
    check('beyond-anchor bin3 NaN', not np.isfinite(vals_t[3]))

    # 3) Poisson anchor likewise excludes the beyond-anchor bins (raw).
    vals_p, _ = evaluate(rec, qv=qv, cent_method='poisson', cent_anchor=0.5,
                         with_jackknife=False)
    check('poisson covered finite', np.isfinite(vals_p[0]))
    check('poisson beyond-anchor NaN', not np.isfinite(vals_p[2]))

    # 3b) Jackknife path still runs without crashing on the anchored set.
    vals_j, _ = evaluate(rec, qv=qv, cent_method='simple', cent_anchor=0.5)
    check('jk covered bin0 finite', np.isfinite(vals_j[0]))

    # 4) Equivalent anchors give identical results (raw).
    vals_a, _ = evaluate(rec, qv=qv, cent_method='simple',
                         cent_anchor={'b_max': 10.0, 'sigma_tot': 2 * np.pi * 100.0},
                         with_jackknife=False)
    check('spec equivalence', np.allclose(vals_a, vals_t, equal_nan=True, atol=1e-9))

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    _run()
