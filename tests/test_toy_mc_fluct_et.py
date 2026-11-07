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
    # signed root: a negative covariance (here a statistical one) keeps its sign
    assert_close("iid: pipeline vs brute force", vals[0], ref, 0.02)
    assert abs(vals[0]) < 0.02, f"iid signal should vanish, got {vals[0]}"


def test_pT_fluct_injected_dynamical():
    fx = _fx(0.10)   # 10% event-wise common pT rescale -> genuine signal
    vals, errs = evaluate(_rec_dy(), qv=fx['qv'], cent_method='simple')
    ref = ref_pT_fluct(fx['events'], (-1e9, 1e9), (0.05, 10.))
    assert_close("injected: pipeline vs brute force", vals[0], ref, 0.05)
    assert abs(vals[0] - 0.10) < 0.02, (
        f"signal should match injected sigma=0.10, got {vals[0]:.4f}")


def test_pT_fluct_eta_window_2d():
    """An eta window on a 2D (flat-y) container: each particle weighted by
    its probability to be in the window, distinct pairs from the
    second-weight sums (until November 2026 NaN).  Against the brute force on
    the particles in the window (the injected signal is reduced by the pT
    window, in both)."""
    fx = _fx(0.10)
    for eta, pt, tol in ((ETA, PTW, 0.05), ((0.2, 0.9), (0.4, 2.0), 0.10)):
        rec = record_pT_fluct(eta=eta, pt=pt, cent_classes=('0-100%',))
        vals, errs = evaluate(rec, qv=fx['qv'], cent_method='simple')
        ref = ref_pT_fluct(fx['events'], eta, pt)
        assert_close(f"eta {eta}, pT {pt}: pipeline vs brute force", vals[0], ref, tol)


def test_pT_fluct_dy_pt_window():
    """dy mode honours the record's pT window (until November 2026 every pT
    of the container entered)."""
    fx = _fx(0.10)
    vals, errs = evaluate(_rec_dy(pt=dict(min=0.4, max=2.0)), qv=fx['qv'], cent_method='simple')
    ref = ref_pT_fluct(fx['events'], (-1e9, 1e9), (0.4, 2.0))
    assert_close("dy, pT 0.4-2: pipeline vs brute force", vals[0], ref, 0.05)


def _with_trend(events, trend):
    """A <pT> trend with the event multiplicity M: pT x (1 + trend (M -
    <M>)/<M>), eta re-derived from y (the converters derive eta from y);
    returns M."""
    from analysis.converters import y_to_eta
    mult = np.array([len(e['pt']) for e in events], float)
    for e, m in zip(events, mult):
        e['pt'] = e['pt'] * (1 + trend * (m - mult.mean()) / mult.mean())
        e['eta'] = y_to_eta(e['y'], e['pt'], e['mass'])
    return mult


def _baseline_ref(events, eta, pt, mult, average='pairs', baseline=True):
    """Brute force at fixed multiplicity: each event's deviations from the
    mean pT at its multiplicity, the event means regressed linearly on mult
    (weights: the events' particle counts, or 1 for the event average);
    normalized by the class mean.  average 'pairs' (ALICE: pooled pairs,
    mean of all particles) or 'events' (STAR: mean of the events' pair
    averages, mean of the event means)."""
    ps, ms = [], []
    for e, m in zip(events, mult):
        sel = (e['eta'] >= eta[0]) & (e['eta'] <= eta[1]) & (e['pt'] >= pt[0]) & (e['pt'] <= pt[1])
        if sel.sum() >= 2:
            ps.append(e['pt'][sel])
            ms.append(m)
    n = np.array([len(p) for p in ps], float)
    y = np.array([p.mean() for p in ps])
    ms = np.array(ms, float)
    u = np.ones_like(n) if average == 'events' else n
    m_bar = np.average(y, weights=u)
    dm = ms - np.average(ms, weights=u)
    M = m_bar + (np.average(dm * (y - m_bar), weights=u) / np.average(dm ** 2, weights=u) * dm if baseline else 0.0)
    M = np.broadcast_to(M, n.shape)
    P = np.array([(p - Mk).sum() ** 2 - ((p - Mk) ** 2).sum() for p, Mk in zip(ps, M)])
    C = np.mean(P / (n * (n - 1))) if average == 'events' else P.sum() / float((n * (n - 1)).sum())
    return float(np.sign(C) * np.sqrt(abs(C)) / m_bar)


def test_pT_fluct_multiplicity_baseline():
    """`multiplicity_baseline` (STAR: <<pT>> fitted vs the centrality
    multiplicity): a <pT> trend with the event multiplicity across a wide
    class adds its spread to the class-mean correlator; deviations from the
    mean at the event's multiplicity remove it.  2D (dy mode) and 3D (eta
    window) against the brute force at fixed multiplicity, and close to the
    events without the trend."""
    cfg = ToyGenConfig(n_events=NE, mult_mean=250, mult_sigma=50, seed=4243, pt_rescale_sigma=0.05)
    events, _ = generate_events(cfg)
    flat = [dict(e) for e in events]
    mult = _with_trend(events, 0.3)
    # pT_max above the rescaled spectra (the converters put larger pT in the last cell)
    qv = make_qvector_data(events, {'cfg': cfg}, n_pt=100, pt_max=5.0)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_pt=100, pt_max=5.0)
    base = {'observable': 'N', 'eta': [{'min': -6.0, 'max': 6.0}], 'pT': {'min': 0.0, 'max': 100.0}}
    eta3 = snap_window(grid.eta_edges, *ETA)
    pt3 = snap_window(grid.pt_edges, *PTW)
    cases = (('2D dy', dict(qv=qv), _rec_dy(), (-1e9, 1e9), (0.05, 10.)),
             ('3D eta', dict(grid=grid), record_pT_fluct(eta=eta3, pt=pt3, cent_classes=('0-100%',)), eta3, pt3))
    for tag, cont, rec, eta, pt in cases:
        plain = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
        rec = dict(rec, multiplicity_baseline=base)
        fixed = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
        ref = _baseline_ref(events, eta, pt, mult)
        no_trend = ref_pT_fluct(flat, eta, pt)
        assert_close(f"{tag}: fixed multiplicity vs brute force", fixed, ref, 0.03)
        assert_close(f"{tag}: fixed multiplicity vs no trend", fixed, no_trend, 0.10)
        assert plain > 1.3 * fixed, f"{tag}: the trend should inflate the class-mean value ({plain:.4f} vs {fixed:.4f})"
        print(f"  class mean {plain:.4f}, fixed multiplicity {fixed:.4f}, no trend {no_trend:.4f}")


def test_pT_fluct_event_average():
    """`averaging: events` (STAR Eq. 1-3): the mean of the events' pair
    averages over the mean of the event means, with and without the
    multiplicity baseline, 2D (dy mode) and 3D, against the brute force.
    A wide multiplicity spread (sigma/mean 0.4) separates it from the pooled
    pairs."""
    cfg = ToyGenConfig(n_events=NE, mult_mean=150, mult_sigma=60, seed=4244, pt_rescale_sigma=0.05)
    events, _ = generate_events(cfg)
    events = [e for e in events if len(e['pt']) >= 20]
    mult = _with_trend(events, 0.2)
    qv = make_qvector_data(events, {'cfg': cfg}, n_pt=100, pt_max=5.0)
    grid = make_qgrid_data(events, {'cfg': cfg}, n_pt=100, pt_max=5.0)
    base = {'observable': 'N', 'eta': [{'min': -6.0, 'max': 6.0}], 'pT': {'min': 0.0, 'max': 100.0}}
    eta3, pt3 = snap_window(grid.eta_edges, *ETA), snap_window(grid.pt_edges, *PTW)
    cases = (('2D dy', dict(qv=qv), _rec_dy(), (-1e9, 1e9), (0.05, 10.)),
             ('3D eta', dict(grid=grid), record_pT_fluct(eta=eta3, pt=pt3, cent_classes=('0-100%',)), eta3, pt3))
    for tag, cont, rec, eta, pt in cases:
        got = {}
        for av in ('pairs', 'events'):
            for bl in (False, True):
                r = dict(rec, averaging=av, **({'multiplicity_baseline': base} if bl else {}))
                got[av, bl] = evaluate(r, cent_method='simple', with_jackknife=False, **cont)[0][0]
                ref = _baseline_ref(events, eta, pt, mult, average=av, baseline=bl)
                assert_close(f"{tag}, {av}, baseline {bl}: vs brute force", got[av, bl], ref,
                             0.01 if tag.startswith('3D') else 0.05)
        assert abs(got['events', False] / got['pairs', False] - 1) > 0.03, got


def test_pT_fluct_oversampled():
    """Containers whose events sum nsamples particlizations (JETSCAPE 2D:
    median 82): a particle counts 1/nsamples, so the self-pairs take
    1/nsamples^2 (until November 2026 1/nsamples: JETSCAPE Au+Au gave
    sqrt(C)/<pT> ~ -0.01 to -0.08 where STAR measures +0.01 to +0.05).  The
    super-events (5 samples sharing one pT rescale) against the brute force
    over all their particles, 2D and 3D, both averagings."""
    from analysis.converters import particles_to_qvector_data, particles_to_qgrid_data
    from analysis.toy_mc import SPECIES_NAMES, SPECIES_MASSES
    ns, n_super = 5, 400
    cfg = ToyGenConfig(n_events=ns * n_super, mult_mean=120, mult_sigma=20, seed=4245)
    base, _ = generate_events(cfg)
    rng = np.random.default_rng(7)
    sup = []
    for i in range(n_super):
        parts = base[i * ns:(i + 1) * ns]
        k = rng.normal(1.0, 0.08)
        ev = {key: np.concatenate([p[key] for p in parts]) for key in ('y', 'pt', 'phi', 'species_idx', 'mass')}
        ev['pt'] = ev['pt'] * k
        ev['eta'] = None
        sup.append(ev)
    from analysis.converters import y_to_eta
    for ev in sup:
        ev['eta'] = y_to_eta(ev['y'], ev['pt'], ev['mass'])
    nsam = np.full(n_super, float(ns))
    pt_edges = np.linspace(cfg.pt_min, 5.0, 101)
    qv = particles_to_qvector_data(sup, pt_edges, rap_max=cfg.rap_max, species_names=list(SPECIES_NAMES),
                                   masses=SPECIES_MASSES, n_harm=3, nsamples=nsam, with_sum_pt=True,
                                   with_sum_pt2=True)
    grid = particles_to_qgrid_data(sup, np.linspace(-6, 6, 51), pt_edges, n_harm=3, nsamples=nsam,
                                   cent_mult=np.array([len(e['pt']) for e in sup], float), with_sum_pt=True,
                                   with_sum_pt2=True, masses=SPECIES_MASSES)
    eta3, pt3 = snap_window(grid.eta_edges, *ETA), snap_window(grid.pt_edges, *PTW)
    mult = np.zeros(n_super)
    for tag, cont, rec, eta, pt in (('2D dy', dict(qv=qv), _rec_dy(), (-1e9, 1e9), (0.05, 10.)),
                                    ('3D eta', dict(grid=grid),
                                     record_pT_fluct(eta=eta3, pt=pt3, cent_classes=('0-100%',)), eta3, pt3)):
        for av in ('pairs', 'events'):
            got = evaluate(dict(rec, averaging=av), cent_method='simple', with_jackknife=False, **cont)[0][0]
            ref = _baseline_ref(sup, eta, pt, mult, average=av, baseline=False)
            assert_close(f"{tag}, {av}, nsamples {ns}: vs brute force", got, ref, 0.03)
            assert got > 0.05, got


def test_pT_fluct_raw_covariance():
    """value_type '<dpt1_dpt2>' (STAR's (MeV/c)^2 tables): the covariance
    itself, GeV^2, against the brute force (2D eta window, 3D)."""
    fx = _fx(0.10)
    eta3 = snap_window(fx['grid'].eta_edges, *ETA)
    pt3 = snap_window(fx['grid'].pt_edges, 0.15, 2.0)   # below the grid's last cell (it holds larger pT too)
    for tag, cont, eta, pt, tol in (('2D', dict(qv=fx['qv']), ETA, PTW, 0.10),
                                    ('3D', dict(grid=fx['grid']), eta3, pt3, 0.02)):
        rec = record_pT_fluct(eta=eta, pt=pt, cent_classes=('0-100%',))
        rel = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
        rec['value_type'] = '<dpt1_dpt2>'
        cov = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
        ps = [e['pt'][(e['eta'] >= eta[0]) & (e['eta'] <= eta[1]) & (e['pt'] >= pt[0]) & (e['pt'] <= pt[1])]
              for e in fx['events']]
        M = np.concatenate(ps).mean()
        ref = sum((p - M).sum() ** 2 - ((p - M) ** 2).sum() for p in ps) / sum(len(p) * (len(p) - 1) for p in ps)
        assert_close(f"{tag}: <dpT dpT> vs brute force (GeV^2)", cov, ref, tol)
        assert_close(f"{tag}: sqrt(<dpT dpT>)/<pT> consistent", np.sqrt(cov) / M, rel, 0.02)


def test_pT_fluct_squared_normalized():
    """value_type '<dpt1_dpt2>/<pT>^2' (STAR's preliminary isobar <delta pT^2>/<pT>^2):
    the covariance over the squared mean, no root -- against the brute force,
    and exactly the square of the signed root form (both averagings, 2D, 3D)."""
    fx = _fx(0.10)
    eta3 = snap_window(fx['grid'].eta_edges, *ETA)
    pt3 = snap_window(fx['grid'].pt_edges, 0.15, 2.0)
    for tag, cont, eta, pt, tol in (('2D', dict(qv=fx['qv']), ETA, PTW, 0.10),
                                    ('3D', dict(grid=fx['grid']), eta3, pt3, 0.02)):
        for averaging in ('pairs', 'events'):
            rec = record_pT_fluct(eta=eta, pt=pt, cent_classes=('0-100%',))
            rec['averaging'] = averaging
            root = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
            rec['value_type'] = '<dpt1_dpt2>/<pT>^2'
            sq = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
            ok = abs(sq - np.sign(root) * root ** 2) <= 1e-12 * abs(sq)
            print(f"  {'PASS' if ok else 'FAIL'} {tag} {averaging}: squared form = (signed root)^2: "
                  f"{sq:.6g} vs {np.sign(root) * root ** 2:.6g}")
            if not ok:
                raise AssertionError(f'{tag} {averaging}: {sq} vs {root}')
        ps = [e['pt'][(e['eta'] >= eta[0]) & (e['eta'] <= eta[1]) & (e['pt'] >= pt[0]) & (e['pt'] <= pt[1])]
              for e in fx['events']]
        M = np.concatenate(ps).mean()
        ref = sum((p - M).sum() ** 2 - ((p - M) ** 2).sum() for p in ps) / sum(len(p) * (len(p) - 1)
                                                                               for p in ps) / M ** 2
        rec['averaging'] = 'pairs'
        sq = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
        assert_close(f"{tag}: <dpT dpT>/<pT>^2 vs brute force", sq, ref, tol)


def _two_sub_ref(events, full, a, b, pt, average):
    """<dpT_i dpT_j> over pairs i in a, j in b (disjoint windows), deviations
    from the full window's mean pT: pooled (pairs) or the mean of the event
    averages (events, STAR 2937844 Eq. 7)."""
    def sel(e, w):
        return (e['eta'] >= w[0]) & (e['eta'] < w[1]) & (e['pt'] >= pt[0]) & (e['pt'] < pt[1])
    rows = []
    for e in events:
        pf, pa, pb = e['pt'][sel(e, full)], e['pt'][sel(e, a)], e['pt'][sel(e, b)]
        if len(pa) and len(pb) and len(pf):
            rows.append((pf, pa, pb))
    if average == 'events':
        M = np.mean([pf.mean() for pf, _, _ in rows])
        return float(np.mean([(pa - M).sum() * (pb - M).sum() / (len(pa) * len(pb)) for _, pa, pb in rows]))
    M = np.concatenate([pf for pf, _, _ in rows]).mean()
    return float(sum((pa - M).sum() * (pb - M).sum() for _, pa, pb in rows)
                 / sum(len(pa) * len(pb) for _, pa, pb in rows))


def test_pT_fluct_two_subevent():
    """Two subevents (particles AB; STAR 2937844: i in -1 < eta < -0.1, j in
    0.1 < eta < 1, deviations from the full window's mean pT): the raw
    covariance against the brute force over the pairs across the
    subevents, both averagings -- exact on 3D cells (windows on edges),
    within 10% on 2D -- and equal to the one-window value for the toy's
    eta-independent rescale."""
    fx = _fx(0.10)
    g = fx['grid']
    full = snap_window(g.eta_edges, -1.0, 1.0)
    a, b = snap_window(g.eta_edges, -1.0, -0.1), snap_window(g.eta_edges, 0.1, 1.0)
    pt = snap_window(g.pt_edges, 0.2, 2.0)
    subs = {lab: {'kinematic_range': {'eta': {'min': w[0], 'max': w[1]}, 'pT': {'min': pt[0], 'max': pt[1]}}}
            for lab, w in (('A', a), ('B', b))}
    for tag, cont, tol in (('3D', dict(grid=g), 1e-9), ('2D', dict(qv=fx['qv']), 0.10)):
        for average in ('pairs', 'events'):
            rec = record_pT_fluct(eta=full, pt=pt, cent_classes=('0-100%',))
            rec.update(value_type='<dpt1_dpt2>', averaging=average)
            one = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
            rec.update(particles='AB', subevents=subs)
            two = evaluate(rec, cent_method='simple', with_jackknife=False, **cont)[0][0]
            ref = _two_sub_ref(fx['events'], full, a, b, pt, average)
            assert_close(f"{tag}, {average}: two-subevent <dpT dpT> vs brute force", two, ref, tol)
            assert_close(f"{tag}, {average}: two-subevent = one window (no eta structure)", two, one, 0.12)


def test_pT_fluct_3d_cut_cells():
    """3D windows that cut cells (eta |eta| < 0.5 on 0.24-wide cells, pT
    0.15-2.0 on 0.037 GeV cells): a cut cell's particles carry its overlap
    share, as on 2D (until November 2026 whole cells, and an upper limit on
    a cell edge took the next cell too)."""
    fx = _fx(0.10)
    for eta, pt in ((ETA, (0.15, 2.0)), ((-0.3, 0.9), (0.4, 1.5))):
        rec = record_pT_fluct(eta=eta, pt=pt, cent_classes=('0-100%',))
        got = evaluate(rec, grid=fx['grid'], cent_method='simple', with_jackknife=False)[0][0]
        assert_close(f"3D eta {eta}, pT {pt}: pipeline vs brute force", got, ref_pT_fluct(fx['events'], eta, pt), 0.03)


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
