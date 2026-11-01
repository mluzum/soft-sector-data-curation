"""Toy-MC validation of spectra in a window of rapidity computed with an
assumed mass (kinematic_range.y.mass, e.g. PHOBOS 0.2 < y_pi < 1.4).

Truth: particles counted with y_pi = asinh(pT sinh(eta) / mT_pi) inside the
window, per unit y_pi.  The toy is flat in true rapidity over |y| < 3 with
pions, kaons and protons; for protons at low pT the window covers less than
half the true-rapidity range of a true-rapidity window, so treating y_pi as
y would fail these checks by a factor ~2.
Run: python3 tests/test_toy_mc_ypi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import (ToyGenConfig, generate_events, make_qvector_data,  # noqa: E402
                             make_qgrid_data, evaluate, species_charge_factor, snap_window)

M_PI = 0.13957
YWIN = (0.2, 1.4)
BINS = [(0.2, 0.3), (0.5, 0.6), (1.0, 1.2)]
_CACHE = {}
results = []


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=1500, mult_mean=400, mult_sigma=50, rap_max=3.0, seed=4242,
                           pt_min=0.05, pt_max=3.0)
        events, _ = generate_events(cfg)
        # The toy 3D grid is not species-resolved; protons get a grid of their own.
        protons = [{k: v[ev['species_idx'] == 2] for k, v in ev.items()} for ev in events]
        _CACHE['fx'] = dict(events=events, qv=make_qvector_data(events, {'cfg': cfg}, n_pt=80),
                            grid=make_qgrid_data(events, {'cfg': cfg}, n_eta=120, n_pt=30),
                            proton_grid=make_qgrid_data(protons, {'cfg': cfg}, n_eta=120, n_pt=30))
    return _CACHE['fx']


def record(species, bins, mass=M_PI):
    return dict(observable_type='particle_spectra', value_type='invariant_yield', hepdata_file='toy.yaml',
                inspire_id='0', version=1, particle_id=dict(species=species),
                kinematic_range=dict(y=dict(min=YWIN[0], max=YWIN[1], mass=mass),
                                     pT=dict(min=bins[0][0], max=bins[-1][1],
                                             bin_ranges=[list(b) for b in bins])),
                centrality=dict(classes=['0-100%'], eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]))


def ref(events, lo, hi, species_sel=None, cf=1.0):
    n = np.zeros(len(events))
    for e, ev in enumerate(events):
        mt = np.sqrt(ev['pt'] ** 2 + M_PI ** 2)
        ypi = np.arcsinh(ev['pt'] * np.sinh(ev['eta']) / mt)
        sel = (ypi >= YWIN[0]) & (ypi < YWIN[1]) & (ev['pt'] >= lo) & (ev['pt'] < hi)
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        n[e] = sel.sum()
    return n.mean() * cf / (2 * np.pi * 0.5 * (lo + hi) * (hi - lo) * (YWIN[1] - YWIN[0]))


def check(name, got, want, tol):
    ok = abs(got - want) <= tol * abs(want)
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got:.4f} want={want:.4f} tol={tol:.2f}")


def run(species, sel, grid):
    fx = _fx()
    cf = species_charge_factor(species)
    if grid:
        bins = [snap_window(fx['grid'].pt_edges, lo, hi) for lo, hi in BINS]
        g = fx['proton_grid'] if species == 'p + pbar' else fx['grid']
        # the proton-only grid holds positives of one species: read it as 'charged'
        vals, _ = evaluate(record('charged' if grid and species == 'p + pbar' else species, bins), grid=g,
                           cent_method='simple', with_jackknife=False)
    else:
        bins = BINS
        vals, _ = evaluate(record(species, bins), qv=fx['qv'], cent_method='simple', with_jackknife=False)
    for b, (lo, hi) in enumerate(bins):
        check(f"{species} {lo:.2f}-{hi:.2f} {'3D' if grid else '2D'}", vals[b], ref(fx['events'], lo, hi, sel, cf), 0.06)


def test_window_is_not_true_rapidity():
    """Sanity of the toy: for protons at pT 0.25 a y_pi window covers less
    than half the true rapidity of the same window in y."""
    from analysis.observables_2d import rapidity_window_extent
    ext = rapidity_window_extent(0.25, 0.93827, *YWIN, M_PI)
    check('proton true-rapidity extent of the y_pi window (< 0.5 of 1.2)', ext / 1.2, 0.4, 0.25)
    assert abs(rapidity_window_extent(0.25, M_PI, *YWIN, M_PI) - 1.2) < 1e-12


def test_charge_average():
    """particle_id.charge_average ((h+ + h-)/2): half of the charged sum."""
    fx = _fx()
    for grid in (False, True):
        bins = [snap_window(fx['grid'].pt_edges, lo, hi) for lo, hi in BINS] if grid else BINS
        kw = dict(grid=fx['grid']) if grid else dict(qv=fx['qv'])
        rec = record('charged', bins)
        avg = dict(rec, particle_id=dict(species='charged', charge_average=True))
        a, _ = evaluate(rec, cent_method='simple', with_jackknife=False, **kw)
        b, _ = evaluate(avg, cent_method='simple', with_jackknife=False, **kw)
        ok = np.allclose(b, 0.5 * a)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'} charge average = half the charged sum ({'3D' if grid else '2D'})")


def test_refused_elsewhere():
    from analysis.dispatcher import resolve
    rec = record('charged', BINS)
    rec.update(observable_type='particle_yields', value_type='dN_dy', observable='dN_dy')
    try:
        resolve(rec)
        ok = False
    except ValueError as e:
        ok = 'assumed mass' in str(e)
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} a yield with a mass-assumed rapidity window is refused")


if __name__ == '__main__':
    test_window_is_not_true_rapidity()
    run('charged', None, grid=False)
    run('p + pbar', [2], grid=False)
    run('charged', None, grid=True)
    run('p + pbar', [2], grid=True)
    test_charge_average()
    test_refused_elsewhere()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
