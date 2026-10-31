"""Toy-MC validation of the flow pipeline (production path).

Generates particle events with known flow, fills QVectorData/QGridData,
and runs records through extract → resolve → prepare → core → jackknife.
Truth sources:
  - fixed v_n: v{2} = v{4} = v_inj exactly (no-ebE-fluctuation theorem)
  - fluctuating v_n: analytic ⟨v⟩ relations from the generated v_e sequence
  - brute-force pair estimator ref_vn2 (independent code)
Run: python3 tests/test_toy_mc_flow.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import (
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data,
    evaluate, record_flow_vn2, record_flow_vn4, record_flow_gap,
    record_flow_diff_pt, record_flow_diff_eta, ref_vn2,
    analytic_vn2_from_ebe, analytic_vn4_from_ebe,
)

# ── shared fixtures (generated once) ────────────────────────────────────────
_CACHE = {}

NE = 1500      # events per config


def _cfg_fixed():
    if 'fixed' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50,
                           seed=101,
                           flow={2: dict(kind='const', value=0.05),
                                 3: dict(kind='const', value=0.03),
                                 4: dict(kind='const', value=0.02)})
        events, truth = generate_events(cfg)
        _CACHE['fixed'] = dict(
            cfg=cfg, events=events, truth=truth,
            qv=make_qvector_data(events, {'cfg': cfg}, n_pt=80),
            grid=make_qgrid_data(events, {'cfg': cfg}, n_pt=80))
    return _CACHE['fixed']


def _cfg_fluct():
    if 'fluct' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50,
                           seed=202,
                           flow={2: dict(kind='const', value=0.06)},
                           fluctuate={2: 0.03})
        events, truth = generate_events(cfg)
        _CACHE['fluct'] = dict(cfg=cfg, events=events, truth=truth,
                               qv=make_qvector_data(events, {'cfg': cfg},
                                                    n_pt=80))
    return _CACHE['fluct']


def _cfg_noflow():
    if 'noflow' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=200, mult_sigma=40,
                           seed=303)
        events, truth = generate_events(cfg)
        _CACHE['noflow'] = dict(cfg=cfg, events=events, truth=truth,
                                qv=make_qvector_data(events, {'cfg': cfg},
                                                     n_pt=80))
    return _CACHE['noflow']


CENT3 = ('0-10%', '10-40%', '40-80%')


# ── helpers ─────────────────────────────────────────────────────────────────

def assert_close(name, got, want, tol):
    ok = abs(got - want) <= tol
    msg = (f"{'PASS' if ok else 'FAIL'} {name}: got={got:.5f} "
           f"want={want:.5f} tol={tol:.5f}")
    print('  ' + msg)
    if not ok:
        raise AssertionError(msg)


# ── tests ───────────────────────────────────────────────────────────────────

def test_noflow_v2_consistent_with_zero():
    fx = _cfg_noflow()
    for harm in (2, 3):
        rec = record_flow_vn2(harmonic=harm, cent_classes=('0-100%',))
        vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
        assert_close(f"no-flow v{harm}{{2}}", abs(vals[0]),
                     0.0, max(4 * errs[0], 0.01))


def test_integrated_vn2_all_harmonics_2d():
    """Fixed-v injection: v_n{2} must recover v_inj in every cent class."""
    fx = _cfg_fixed()
    vinj = {2: 0.05, 3: 0.03, 4: 0.02}
    rec = record_flow_vn2(cent_classes=CENT3)
    for harm in (2, 3, 4):
        r = dict(rec)
        r['harmonic'] = harm
        r['value_type'] = f'v_{harm}{{2}}'
        vals, errs = evaluate(r, qv=fx['qv'], cent_method='simple')
        for ic, v in enumerate(vals):
            assert_close(f"v{harm}{{2}} {CENT3[ic]}", v, vinj[harm],
                         max(4 * errs[ic], 0.006))


def test_integrated_vn2_3d_matches_2d():
    """Single-acceptance v₂{2} from the 3D grid ≈ 2D result."""
    fx = _cfg_fixed()
    rec = record_flow_vn2(cent_classes=('0-100%',))
    v2d, e2 = evaluate(rec, qv=fx['qv'], cent_method='simple')
    v3d, e3 = evaluate(rec, grid=fx['grid'], cent_method='simple')
    tol = max(3 * max(e2[0], e3[0]), 0.006)
    assert_close("3D vs 2D v₂{2}", v3d[0], v2d[0], tol)


def test_integrated_vn4_fixed_flow_equals_vn2():
    """No ebE fluctuations ⇒ v{4} = v{2} = v_inj (analytic theorem)."""
    fx = _cfg_fixed()
    for harm in (2, 3):
        rec = record_flow_vn4(harmonic=harm, cent_classes=('0-100%',))
        vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
        assert_close(f"fixed-flow v{harm}{{4}}", vals[0],
                     {2: 0.05, 3: 0.03}[harm], max(5 * errs[0], 0.008))


def test_fluctuating_split_vn2_vn4():
    """With ebE fluctuations: pipeline matches analytic ⟨v⟩ relations and
    v{4} < v{2}."""
    fx = _cfg_fluct()
    ve = fx['truth']['vn_ebe'][2]
    a2 = analytic_vn2_from_ebe(ve)
    a4 = analytic_vn4_from_ebe(ve)
    assert a4 < a2, "analytic precondition v{4}<v{2}"

    r2 = record_flow_vn2(harmonic=2, cent_classes=('0-100%',))
    vals, errs = evaluate(r2, qv=fx['qv'], cent_method='simple')
    assert_close("fluct v₂{2}", vals[0], a2, max(4 * errs[0], 0.006))

    r4 = record_flow_vn4(harmonic=2, cent_classes=('0-100%',))
    vals4, errs4 = evaluate(r4, qv=fx['qv'], cent_method='simple')
    assert_close("fluct v₂{4}", vals4[0], a4, max(6 * errs4[0], 0.010))


def test_bruteforce_reference_agrees_pipeline():
    """Independent pair-sum estimator vs pipeline v₂{2} (same acceptance)."""
    fx = _cfg_fixed()
    rec = record_flow_vn2(cent_classes=('0-100%',), pt=(0.3, 1.5))
    vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
    ref = ref_vn2(fx['events'], 2, exact_jac_eta=(-1, 1),
                  rap_max=fx['cfg'].rap_max, pt_range=(0.3, 1.5))
    assert_close("brute-force vs pipeline v₂{2}", vals[0], ref,
                 max(4 * errs[0], 0.006))


def test_differential_pT_linear_profile():
    """Injected rising v₂(pT): per-bin recovery vs brute force."""
    fx = _cfg_fixed()
    bins = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.2)]
    rec = record_flow_diff_pt(bins, harmonic=2, cent_classes=('0-100%',))
    vals, errs = evaluate(rec, qv=fx['qv'], n_blocks=12)
    n_b = len(bins)
    for b, (lo, hi) in enumerate(bins):
        ref = ref_vn2(fx['events'], 2, exact_jac_eta=(-1, 1),
                      rap_max=fx['cfg'].rap_max, pt_range=(lo, hi))
        assert_close(f"diff-pT bin {lo}-{hi}", vals[b], ref,
                     max(5 * errs[min(b, len(errs)-1)], 0.015))


def test_differential_eta_flat():
    """Flat-in-y generation with const v: every η bin recovers v_inj."""
    fx = _cfg_fixed()
    bins = [(-0.9, -0.4), (-0.4, 0.4), (0.4, 0.9)]
    rec = record_flow_diff_eta(bins, harmonic=2, cent_classes=('0-100%',))
    vals, errs = evaluate(rec, qv=fx['qv'], n_blocks=12)
    for b, (lo, hi) in enumerate(bins):
        assert_close(f"diff-η bin {lo}-{hi}", vals[b], 0.05,
                     max(5 * errs[min(b, len(errs)-1)], 0.012))


def test_gap_ab_recovers_reference():
    """AB subevent with common Ψ: gap method ≈ single-acceptance value."""
    fx = _cfg_fixed()
    rec_gap = record_flow_gap(sub_a=(-4, -1), sub_b=(1, 4),
                              cent_classes=('0-100%',))
    vg, eg = evaluate(rec_gap, qv=fx['qv'], cent_method='simple')
    rec_ref = record_flow_vn2(pt=(0.2, 2.0), cent_classes=('0-100%',))
    vr, er = evaluate(rec_ref, qv=fx['qv'], cent_method='simple')
    assert_close("gap AB vs single", vg[0], vr[0],
                 max(4 * max(eg[0], er[0]), 0.008))


def test_gap_3d_matches_2d():
    """No longitudinal decorrelation injected ⇒ 3D gap ≈ 2D gap."""
    fx = _cfg_fixed()
    rec = record_flow_gap(sub_a=(-4, -1), sub_b=(1, 4),
                          cent_classes=('0-100%',))
    v3d, e3 = evaluate(rec, grid=fx['grid'], cent_method='simple')
    v2d, e2 = evaluate(rec, qv=fx['qv'])
    assert_close("3D vs 2D gap", v3d[0], v2d[0],
                 max(4 * max(e2[0], e3[0]), 0.008))


def test_jackknife_error_coverage():
    """Error-bar calibration: |rec−truth| < 2σ_JK in ≥ 9/12 seeds."""
    n_seed, hits = 12, 0
    for s in range(n_seed):
        cfg = ToyGenConfig(n_events=600, mult_mean=220, mult_sigma=45,
                           seed=1000 + s,
                           flow={2: dict(kind='const', value=0.05)})
        events, _ = generate_events(cfg)
        qv = make_qvector_data(events, {'cfg': cfg}, n_pt=60)
        rec = record_flow_vn2(cent_classes=('0-100%',))
        vals, errs = evaluate(rec, qv=qv, n_blocks=10,
                              cent_method='simple')
        if abs(vals[0] - 0.05) < 2 * max(errs[0], 1e-6):
            hits += 1
    frac = hits / n_seed
    print(f"  coverage ±2σ: {hits}/{n_seed}")
    assert frac >= 0.7, f"±2σ coverage too low: {hits}/{n_seed}"


# ── run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = 0
    n_fail = 0
    for name in tests:
        try:
            globals()[name]()
            print(f"  PASS  {name}\n")
            n_pass += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}\n")
            n_fail += 1
    print(f"{n_pass}/{n_pass + n_fail} passed")
    sys.exit(1 if n_fail else 0)
