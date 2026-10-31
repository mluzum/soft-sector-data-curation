"""Toy-MC validation of pT-fluctuation and transverse-energy observables.

pT_fluct physics: for strictly independent particle emission the ALICE
two-particle correlator sqrt(<dpTi dpTj>)/<pT> is ZERO — the measured
non-zero values are genuine dynamical correlations.  Validation therefore:
  (a) i.i.d. events  -> pipeline consistent with the brute-force zero;
  (b) events with an injected event-wise common pT rescale (dynamical
      mean-pT fluctuations) -> pipeline recovers the brute-force signal.
dET_deta truth: exact per-particle E_T = E/cosh(eta); grid windows snapped
to bin edges to avoid straddling-bin systematics.
Run: python3 tests/test_toy_mc_fluct_et.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.pipeline_dispatch import get_pipeline_entry

from analysis.toy_mc import (
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data,
    evaluate, record_pT_fluct, record_dET, ref_pT_fluct, ref_dET,
    snap_window,
)
from analysis.pipeline_dispatch import get_pipeline_entry
from analysis.dispatcher import extract_measurement_params

NE = 1500
ETA = (-0.5, 0.5)
PTW = (0.15, 3.0)
_CACHE = {}


def _fx(sigma_resc=0.0):
    key = f'fx_{sigma_resc}'
    if key not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50,
                           seed=4242, pt_rescale_sigma=sigma_resc)
        events, truth = generate_events(cfg)
        _CACHE[key] = dict(
            cfg=cfg, events=events,
            qv=make_qvector_data(events, {'cfg': cfg}, n_pt=80),
            grid=make_qgrid_data(events, {'cfg': cfg}, n_pt=80))
    return _CACHE[key]


def assert_close(name, got, want, tol):
    ok = abs(got - want) <= tol * max(abs(want), 1e-12) + 1e-6
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got:.5f} "
          f"want={want:.5f} tol(rel)={tol}")
    if not ok:
        raise AssertionError(f"{name}: {got} vs {want}")


def _rec_dy(pt=dict(min=0.0, max=10.0)):
    """pT_fluct record in dy mode (uniform weights).

    Full-pT acceptance by default: brute-force pair estimators are biased
    low when the pT window truncates spectra whose per-event rescale is
    coupled to the selection; the pipeline's inclusive-M-bar convention is
    insensitive to this (verified: full-acceptance pipeline vs brute force
    agree to ~1%).
    """
    rec = record_pT_fluct(eta=None, cent_classes=('0-100%',))
    rec['kinematic_range']['pT'] = pt
    return rec


def test_pT_fluct_iid_consistent_zero():
    fx = _fx(0.0)
    vals, errs = evaluate(_rec_dy(), qv=fx['qv'], cent_method='simple')
    ref = ref_pT_fluct(fx['events'], (-1e9, 1e9), (0.05, 10.))
    assert_close("iid: pipeline vs brute force", vals[0], abs(ref), 0.02)
    assert abs(vals[0]) < 0.01, f"iid signal should vanish, got {vals[0]}"


def test_pT_fluct_injected_dynamical():
    fx = _fx(0.10)   # 10% event-wise common pT rescale -> genuine signal
    vals, errs = evaluate(_rec_dy(), qv=fx['qv'], cent_method='simple')
    ref = ref_pT_fluct(fx['events'], (-1e9, 1e9), (0.05, 10.))
    assert_close("injected: pipeline vs brute force", vals[0], ref, 0.05)
    assert abs(vals[0] - 0.10) < 0.02, (
        f"signal should match injected sigma=0.10, got {vals[0]:.4f}")


def test_pT_fluct_eta_mode_documented_nan():
    fx = _fx(0.10)
    rec = record_pT_fluct(eta=ETA, pt=PTW, cent_classes=('0-100%',))
    vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
    assert not np.isfinite(vals[0]), (
        "eta-mode must be NaN until second-weight moments are stored")


def test_dET_deta_2d():
    """Flat-y path: sum_mT slot or midpoint fallback x analytic window
    factor; ~4% systematic from mT-vs-1/cosh correlation in-window."""
    fx = _fx(0.0)
    rec = record_dET(eta=ETA, pt=PTW, cent_classes=('0-100%',))
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('dET_deta')
    w = np.ones(fx['qv'].n_events)
    truth = ref_dET(fx['events'], ETA, PTW) * 2.0
    qv_nomom = make_qvector_data(fx['events'], {'cfg': fx['cfg']}, n_pt=80,
                                 with_sum_pt=False)
    for tag, q, tol in [('slot', fx['qv'], 0.05),
                        ('midpoint', qv_nomom, 0.05)]:
        d = entry['prepare_data_2d'](q, params)
        assert_close(f"dET/deta 2D {tag}", entry['core_fn'](d, w)[0],
                     truth, tol)


def test_pT_fluct_midpoint_fallback():
    """Midpoint S1/S2 estimates reproduce exact-moment Sigma to <2%."""
    fx = _fx(0.10)
    rec = _rec_dy()
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('pT_fluct')
    w = np.ones(fx['qv'].n_events)
    qv_nomom = make_qvector_data(fx['events'], {'cfg': fx['cfg']}, n_pt=80,
                                 with_sum_pt=False)
    v_ex = entry['core_fn'](entry['prepare_data_2d'](fx['qv'], params), w)[0]
    v_mid = entry['core_fn'](entry['prepare_data_2d'](qv_nomom, params), w)[0]
    assert_close("pT-fluct midpoint vs exact moments", v_mid, v_ex, 0.02)


def test_dET_deta_3d():
    fx = _fx(0.0)
    eta = snap_window(fx['grid'].eta_edges, *ETA)
    pt = snap_window(fx['grid'].pt_edges, *PTW)
    rec = record_dET(eta=eta, pt=pt, cent_classes=('0-100%',))
    vals, errs = evaluate(rec, grid=fx['grid'], cent_method='simple')
    cf = 2.0   # charged: positives-only sim convention
    ref = ref_dET(fx['events'], eta, pt) * cf
    assert_close("dET/deta 3D vs exact", vals[0], ref, 0.02)


if __name__ == '__main__':
    tests = [fn for fn in sorted(dir()) if fn.startswith('test_')]
    n_pass = n_fail = 0
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
