"""Toy-MC validation of yields / spectra / mean-pT (production path).

Truth: plain geometric particle counts in the measurement window (the
selection itself applies the Jacobian; re-weighting would double-count).
Charge factors follow the pipeline convention (positives-only sim data).
Run: python3 tests/test_toy_mc_yields_spectra.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.pipeline_dispatch import get_pipeline_entry

from analysis.toy_mc import (
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data,
    evaluate, record_yield, record_spectra, record_mean_pt,
    ref_yield, ref_spectra, ref_mean_pt, species_charge_factor, snap_window,
)

NE = 1200
ETA = (-0.5, 0.5)
PTW = (0.15, 3.0)
BINS = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.2)]
CENT3 = ('0-10%', '10-40%', '40-80%')

_CACHE = {}


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50,
                           seed=777, pt_max=3.0)
        events, truth = generate_events(cfg)
        _CACHE['fx'] = dict(
            cfg=cfg, events=events,
            qv=make_qvector_data(events, {'cfg': cfg}, n_pt=80),
            grid=make_qgrid_data(events, {'cfg': cfg}, n_pt=80))
    return _CACHE['fx']


def assert_close(name, got, want, tol):
    ok = abs(got - want) <= tol * max(abs(want), 1e-12) + 4e-4
    msg = (f"{'PASS' if ok else 'FAIL'} {name}: got={got:.4f} "
           f"want={want:.4f} tol(rel)={tol:.3f}")
    print('  ' + msg)
    if not ok:
        raise AssertionError(msg)


# ── yields ──────────────────────────────────────────────────────────────────

def test_yield_charged_2d():
    fx = _fx()
    rec = record_yield(eta=ETA, pt=PTW, cent_classes=('0-100%',))
    cf = species_charge_factor('charged')
    vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
    ref = ref_yield(fx['events'], ETA, PTW) * cf
    assert_close("yield charged 2D", vals[0], ref,
                 max(6 * errs[0] / max(ref, 1e-9), 0.02))


def test_yield_species_2d():
    fx = _fx()
    for sp, sel in [('K+ + K-', [1]), ('p + pbar', [2])]:
        rec = record_yield(species=sp, eta=ETA, pt=PTW,
                           cent_classes=('0-100%',))
        cf = species_charge_factor(sp)
        vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
        ref = ref_yield(fx['events'], ETA, PTW, species_sel=sel) * cf
        assert_close(f"yield {sp} 2D", vals[0], ref,
                     max(6 * errs[0] / max(ref, 1e-9), 0.02))


def test_yield_charged_3d():
    fx = _fx()
    eta = snap_window(fx['grid'].eta_edges, *ETA)
    pt = snap_window(fx['grid'].pt_edges, *PTW)
    rec = record_yield(eta=eta, pt=pt, cent_classes=('0-100%',))
    cf = species_charge_factor('charged')
    vals, errs = evaluate(rec, grid=fx['grid'], cent_method='simple')
    ref = ref_yield(fx['events'], eta, pt) * cf
    assert_close("yield charged 3D", vals[0], ref,
                 max(6 * errs[0] / max(ref, 1e-9), 0.03))


def test_centrality_simple_vs_poisson():
    """Documented systematic: two weight schemes must agree within ~5-7%.

    With anchor=100 (full coverage) the simple (rank) and Poisson (CDF
    inversion) methods agree in central bins, but Poisson spreads more
    weight to peripheral bins because the CDF assigns non-zero probability
    to events with multiplicity below the bin's lower edge.  This produces
    a documented 5-7% systematic at the 40-80% cent class (peripheral,
    wide bin).
    """
    fx = _fx()
    rec = record_yield(eta=ETA, pt=PTW, cent_classes=CENT3)
    v_s, e_s = evaluate(rec, qv=fx['qv'], cent_method='simple')
    v_p, e_p = evaluate(rec, qv=fx['qv'], cent_method='poisson')
    TOLS = (0.05, 0.05, 0.08)  # central, mid, peripheral
    for ic in range(len(CENT3)):
        assert_close(f"cent-{CENT3[ic]} simple-vs-poisson",
                     v_p[ic], v_s[ic], TOLS[ic])


# ── spectra ─────────────────────────────────────────────────────────────────

# Inherent 2D-method systematic: binned spectra normalize by 1/(pT_center),
# while the falling spectrum weights each bin toward its lower edge.  The
# sign of the resulting bias tracks the local spectral slope (+ where the
# Jacobian rises steeply, − on the falling side); magnitude ~7% at these
# bin widths.  Tolerances below QUANTIFY this documented systematic.
SPECTRA_TOL = [0.09, 0.09, 0.05]


def test_spectra_2d():
    fx = _fx()
    rec = record_spectra(eta=ETA, bins=BINS, cent_classes=('0-100%',))
    cf = species_charge_factor('charged')
    vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
    for b, (lo, hi) in enumerate(BINS):
        ref = ref_spectra(fx['events'], lo, hi, ETA) * cf
        assert_close(f"spectra {lo}-{hi} 2D", vals[b], ref,
                     SPECTRA_TOL[b])


def test_spectra_3d():
    fx = _fx()
    eta = snap_window(fx['grid'].eta_edges, *ETA)
    sb = [snap_window(fx['grid'].pt_edges, lo, hi) for lo, hi in BINS]
    rec = record_spectra(eta=eta, bins=sb, cent_classes=('0-100%',))
    cf = species_charge_factor('charged')
    vals, errs = evaluate(rec, grid=fx['grid'], cent_method='simple')
    for b, (lo, hi) in enumerate(sb):
        ref = ref_spectra(fx['events'], lo, hi, eta) * cf
        assert_close(f"spectra {lo:.3f}-{hi:.3f} 3D", vals[b], ref,
                     SPECTRA_TOL[b])


def test_spectra_3d_bins_off_the_grid():
    """Measurement bins that cut grid cells count the cut cells by their
    overlap fraction.  Whole-cell snapping replaced a 0.3-0.5 bin on
    ~0.2-wide cells by other cells, and a bin inside one cell got nothing."""
    fx = _fx()
    coarse = make_qgrid_data(fx['events'], {'cfg': fx['cfg']}, n_pt=15)
    bins = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.2)]
    rec = record_spectra(eta=(-0.5, 0.5), bins=bins, cent_classes=('0-100%',))
    cf = species_charge_factor('charged')
    vals, _ = evaluate(rec, grid=coarse, cent_method='simple')
    for b, (lo, hi) in enumerate(bins):
        ref = ref_spectra(fx['events'], lo, hi, (-0.5, 0.5)) * cf
        assert_close(f"spectra {lo}-{hi} 3D, coarse grid", vals[b], ref, 0.06)


def test_yield_rapidity_window_3d():
    """dN/dy of an identified species in a rapidity window on a per-species
    3D grid: the eta window of the species' mass per pT cell.  A mixture
    (charged) has no single mass: NaN.  (Until Nov 2026 every rapidity-window
    yield was 0 on 3D containers.)"""
    from analysis.qvector import MultiSpeciesGrid
    from analysis.toy_mc import SPECIES_MASSES
    fx = _fx()
    grids = {}
    for idx, pdg in ((0, 211), (2, 2212)):
        evs = []
        for ev in fx['events']:
            m = ev['species_idx'] == idx
            evs.append({k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v)
                        for k, v in ev.items()})
        grids[pdg] = make_qgrid_data(evs, {'cfg': fx['cfg']}, n_eta=120, n_pt=60)
    msg = MultiSpeciesGrid(grids)
    for sp, idx, pdg in (('pi+ + pi-', 0, 211), ('p + pbar', 2, 2212)):
        rec = record_yield(species=sp, cent_classes=('0-100%',))
        rec['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
        vals, _ = evaluate(rec, grid=msg, cent_method='simple')
        m = SPECIES_MASSES[idx]
        n = 0
        for ev in fx['events']:
            s = ev['species_idx'] == idx
            mt = np.sqrt(ev['pt'][s] ** 2 + m ** 2)
            pz = ev['pt'][s] * np.sinh(ev['eta'][s])
            y = np.arcsinh(pz / mt)
            n += np.count_nonzero(np.abs(y) < 0.5)
        ref = 2.0 * n / len(fx['events']) / 1.0          # x2: charge symmetry
        assert_close(f"dN/dy |y| < 0.5 {sp} 3D", vals[0], ref, 0.03)
    rec = record_yield(species='charged', cent_classes=('0-100%',))
    rec['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
    vals, _ = evaluate(rec, grid=msg, cent_method='simple')
    assert np.isnan(vals[0]), vals


def test_spectra_dN_dpT_form():
    """value_type dN_dpT_dy: d2N/(dpT dy) of the bin, the counts over the
    bin width and the window (2 pi pT_c x the invariant yield)."""
    fx = _fx()
    cf = species_charge_factor('charged')
    for name, kw in (('2D', dict(qv=fx['qv'])), ('3D', dict(grid=fx['grid']))):
        eta = ETA if name == '2D' else snap_window(fx['grid'].eta_edges, *ETA)
        bins = BINS if name == '2D' else [snap_window(fx['grid'].pt_edges, lo, hi) for lo, hi in BINS]
        rec = record_spectra(eta=eta, bins=bins, cent_classes=('0-100%',))
        inv, _ = evaluate(rec, cent_method='simple', **kw)
        rec['value_type'] = 'dN_dpT_dy'
        vals, _ = evaluate(rec, cent_method='simple', **kw)
        for b, (lo, hi) in enumerate(bins):
            ref = ref_spectra(fx['events'], lo, hi, eta) * cf * 2 * np.pi * (lo + hi) / 2
            assert_close(f"dN/dpT {lo:.3f}-{hi:.3f} {name}", vals[b], ref, SPECTRA_TOL[b])
            assert abs(vals[b] - inv[b] * 2 * np.pi * (lo + hi) / 2) <= 1e-12 * abs(vals[b])
    from analysis.dispatcher import resolve
    rec['value_type'] = 'E d3N/dp3'
    try:
        resolve(rec)
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown spectrum form was not refused')


# ── mean pT ─────────────────────────────────────────────────────────────────

def test_mean_pt_2d():
    fx = _fx()
    for sp, sel in [('charged', None), ('K+ + K-', [1]), ('p + pbar', [2])]:
        rec = record_mean_pt(species=sp, eta=ETA, pt=PTW,
                             cent_classes=('0-100%',))
        vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
        ref = ref_mean_pt(fx['events'], ETA, PTW, species_sel=sel)
        assert_close(f"mean-pt {sp} 2D", vals[0], ref, 0.02)


def test_mean_pt_3d_charged():
    fx = _fx()
    rec = record_mean_pt(eta=ETA, pt=PTW, cent_classes=('0-100%',))
    vals, errs = evaluate(rec, grid=fx['grid'], cent_method='simple')
    ref = ref_mean_pt(fx['events'], ETA, PTW)
    assert_close("mean-pt charged 3D", vals[0], ref, 0.02)


def test_mean_pt_cent_classes_2d():
    """⟨pT⟩ roughly constant across centrality in this toy model."""
    fx = _fx()
    rec = record_mean_pt(eta=ETA, pt=PTW, cent_classes=CENT3)
    vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple',
                          n_blocks=10)
    spread = (vals.max() - vals.min()) / vals.mean()
    print(f"  centrality spread of <pT>: {spread:.3f}")
    assert spread < 0.10, f"<pT> spread across centrality too large {spread}"



def test_mean_pt_midpoint_fallback():
    """No sum_pt stored -> bin-midpoint estimate; must agree with the exact
    moment within ~1% on a fine grid and truth within 2%."""
    fx = _fx()
    # rebuild containers WITHOUT sum_pt
    qv_nomom = make_qvector_data(fx['events'], {'cfg': fx['cfg']}, n_pt=80,
                                 with_sum_pt=False)
    grid_nomom = make_qgrid_data(fx['events'], {'cfg': fx['cfg']}, n_pt=80,
                                 with_sum_pt=False)
    ref = ref_mean_pt(fx['events'], ETA, PTW)
    rec = record_mean_pt(eta=ETA, pt=PTW, cent_classes=('0-100%',))
    e = get_pipeline_entry('mean_pt')
    w = np.ones(qv_nomom.n_events)
    d_exact = e['prepare_data_2d'](fx['qv'], _params(rec))
    v_exact = e['core_fn'](d_exact, w)[0]
    d_mid = e['prepare_data_2d'](qv_nomom, _params(rec))
    v_mid = e['core_fn'](d_mid, w)[0]
    assert_close("midpoint vs exact-sum_pt", v_mid, v_exact, 0.01)
    assert_close("midpoint vs particle truth", v_mid, ref, 0.02)


def _params(rec):
    from analysis.dispatcher import extract_measurement_params as emp
    return emp(rec)


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
