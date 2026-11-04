"""Yields and <pT> in a pT window that cuts a container cell.

ATLAS O+O and Ne+Ne measure dN/deta, dN/dy_pi and <pT> above pT = 0.27
GeV; the 3D grids have 0.2 GeV cells, the 2D containers 0.05 GeV cells.
The prepares sum each window with the cut cell at its flat share (uniform in
pT^2) and attach a pt_subcell block; the core replaces the flat share by the
share of the local model of the class-averaged spectrum
(analysis.spectra_shape.window_shares, as for spectra; the pT-weighted share
for sums of pT), applied to the cell's count in the window.  Checks:
  - window_shares: one window = shares(); several windows = one call per
    window; an exactly exponential invariant yield (m = 0, no Jacobian):
    count and pT shares exact;
  - toy events with thermal spectra, pT > 0.27 and > 0.33 GeV, against the
    particle level:
      3D eta windows (0.2 GeV cells): <pT> and dN/deta within 0.3% (until
      November 2026 the integrated <pT> started at 0.4 GeV, +16%; the flat
      share gave yields +2.7%, per-bin <pT> -2.7%), charged and pi+;
      3D y_pi and true-rapidity windows: within 0.8% (the window mapped at
      each cell's centre: 0.1-0.4% already with whole cells);
      2D (0.05 GeV cells): within 0.6% (until then the window snapped to
      0.3 GeV);
    and the flat share alone misses by more;
  - an integrated window = the same window as one bin, to rounding (3D:
    separable sums vs window matrices; 2D);
  - windows on cell edges carry no block, unchanged (3D eta: the particle
    count exactly);
  - the jackknife masks the block (errors finite);
  - 3D dE_T/deta at pT > 0.27 takes the cut cell's flat share (until
    November 2026 it snapped to whole cells).
Run: python3 tests/test_toy_mc_pt_cut_cells.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis import pipeline_dispatch as pd  # noqa: E402
from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.spectra_shape import shares, window_shares  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, evaluate, SPECIES_MASSES,
)

MPI = 0.13957
WIN = (-0.8, 0.8)
results = []
_CACHE = {}


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def fx():
    if not _CACHE:
        cfg = ToyGenConfig(n_events=300, mult_mean=3000, mult_sigma=200, seed=5, rap_max=3.0, pt_min=0.0,
                           pt_max=3.0)
        ev, tr = generate_events(cfg)
        ee, pe = np.linspace(-3, 3, 31), np.linspace(0.0, 3.0, 16)        # 0.2 eta and 0.2 GeV cells
        mult = np.array([len(e['pt']) for e in ev], float)

        def grid(slots):
            evs = [{k: v[np.isin(e['species_idx'], slots)] for k, v in e.items()} for e in ev]
            return particles_to_qgrid_data(evs, ee, pe, n_harm=3, masses=SPECIES_MASSES, cent_mult=mult,
                                           with_sum_pt=True, with_sum_ET=True)
        _CACHE.update(events=ev,
                      grid=MultiSpeciesGrid({211: grid([0]), 321: grid([1]), 2212: grid([2])},
                                            combined=grid([0, 1, 2])),
                      qv=make_qvector_data(ev, tr, n_pt=60, n_harm=3))   # 0.05 GeV cells
    return _CACHE


def rec(t, lo, var='eta', species='charged', mass=None, bins=None):
    r = {'observable_type': t, 'hepdata_file': 't.yaml', 'inspire_id': '0', 'version': 1, 'value_type': 'x',
         'collision_system': {'reaction': 'AU AU', 'energy_gev': 200.0}, 'particle_id': {'species': species},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -3.0, 'max': 3.0}]}}
    w = {'min': WIN[0], 'max': WIN[1]}
    if mass:
        w['mass'] = mass
    if bins:
        w['bin_ranges'] = bins
        r['dimensionality'] = 'differential_' + var
    elif t == 'mean_pT':
        r['dimensionality'] = 'integrated'
    r['kinematic_range'] = {var: w, 'pT': {'min': lo, 'max': 3.0}}
    return r


def truth(lo, var, species, mass):
    """(<pT>, dN/dx) of the particles: charged hadrons x 2 (positives only in the toy)."""
    pts = []
    for e in fx()['events']:
        sel = (e['pt'] > lo) & (e['pt'] < 3.0)
        if species == 'pi+':
            sel &= e['species_idx'] == 0
        if var == 'eta':
            x = e['eta']
        elif mass:
            x = np.arcsinh(e['pt'] * np.sinh(e['eta']) / np.sqrt(e['pt'] ** 2 + mass ** 2))
        else:
            x = e['y']
        pts.append(e['pt'][sel & (x > WIN[0]) & (x < WIN[1])])
    pt = np.concatenate(pts)
    return pt.mean(), (1.0 if species == 'pi+' else 2.0) * len(pt) / len(pts) / (WIN[1] - WIN[0])


def run(r, kind, jk=False):
    c = fx()
    kw = {'qv': c['qv']} if kind == '2d' else {'grid': c['grid']}
    v, e = evaluate(r, cent_method='simple', with_jackknife=jk, **kw)
    return np.atleast_1d(v), (np.atleast_1d(e) if jk else None)


def flat(r, kind):
    """The value with the flat share (the core without _pt_subcell_corrected)."""
    c = fx()
    params = extract_measurement_params(r)
    key = 'mean_pt' if r['observable_type'] == 'mean_pT' else 'yield_dNch_deta'
    if kind == '2d':
        d = pd.PIPELINE_DISPATCH[key]['prepare_data_2d'](c['qv'], params)
    else:
        d = pd.PIPELINE_DISPATCH[key]['prepare_data_3d'](c['grid'].select(r['particle_id']['species']), params)
    d = {k: v for k, v in d.items() if k != 'pt_subcell'}
    return np.atleast_1d(pd.PIPELINE_DISPATCH[key]['core_fn'](d, np.ones(len(c['events']))))[0]


def test_window_shares():
    rng = np.random.default_rng(2)
    e = np.linspace(0.0, 2.0, 11)
    y = np.exp(-np.sqrt(np.linspace(0.1, 1.9, 10) ** 2 + MPI ** 2) / 0.25) * np.diff(e ** 2)
    parts = ((1, 0.27, 0.4), (7, 1.4, 1.53))
    wins = [(-0.8, 0.8), (1.0, 1.5), (-2.5, -2.0)]
    Y = y[None, :] * rng.uniform(0.9, 1.1, (3, 10))
    got = window_shares(e, Y, parts, MPI, wins)['count']
    one = np.array([shares(e, Y[k], parts, MPI, wins[k]) for k in range(3)])
    check('window_shares: several windows = one call per window = shares()', np.allclose(got, one, rtol=0, atol=1e-14))
    # exponential invariant yield exp(-p/T), m = 0, no Jacobian: count ~ int e p dp, pT sum ~ int e p^2 dp
    T = 0.3

    def F(a, b, k):
        from scipy.integrate import quad
        return quad(lambda p: np.exp(-p / T) * p ** (1 + k), a, b)[0]
    counts = np.array([F(a, b, 0) for a, b in zip(e[:-1], e[1:])])
    f = window_shares(e, counts[None, :], parts, 0.0, None, ('count', 'pT'))
    want_c = np.array([F(lo, hi, 0) / F(e[c], e[c + 1], 0) for c, lo, hi in parts])
    want_p = np.array([F(lo, hi, 1) / F(e[c], e[c + 1], 1) for c, lo, hi in parts])
    check('exponential invariant yield: count and pT shares exact',
          np.allclose(f['count'][0], want_c, rtol=1e-12) and np.allclose(f['pT'][0], want_p, rtol=1e-12),
          f"{f['count'][0]} {f['pT'][0]}")


def test_accuracy():
    cases = (('3d', 'eta', 'charged', None, 0.003), ('3d', 'eta', 'pi+', None, 0.003),
             ('3d', 'y', 'charged', MPI, 0.008), ('3d', 'y', 'pi+', None, 0.008),
             ('2d', 'eta', 'charged', None, 0.006), ('2d', 'y', 'charged', MPI, 0.006),
             ('2d', 'eta', 'pi+', None, 0.006))
    for lo in (0.27, 0.33):
        for kind, var, sp, mass, tol in cases:
            tm, ty = truth(lo, var, sp, mass)
            vm = run(rec('mean_pT', lo, var, sp, mass), kind)[0][0]
            vy = run(rec('particle_yields', lo, var, sp, mass), kind)[0][0]
            fm, fy = flat(rec('mean_pT', lo, var, sp, mass), kind), flat(rec('particle_yields', lo, var, sp, mass), kind)
            dm, dy, gm, gy = vm / tm - 1, vy / ty - 1, fm / tm - 1, fy / ty - 1
            check(f'{kind} {var}{"_pi" if mass else ""} {sp} pT > {lo}: <pT> {dm:+.4f}, yield {dy:+.4f} '
                  f'(flat share {gm:+.4f}, {gy:+.4f})',
                  abs(dm) < tol and abs(dy) < tol and max(abs(gm), abs(gy)) > max(abs(dm), abs(dy)))


def test_integrated_equals_one_bin():
    for kind in ('3d', '2d'):
        for t in ('mean_pT', 'particle_yields'):
            for var, mass in (('eta', None), ('y', MPI)):
                a = run(rec(t, 0.27, var, mass=mass), kind)[0]
                b = run(rec(t, 0.27, var, mass=mass, bins=[list(WIN)]), kind)[0]
                check(f'{kind} {t} {var}: integrated = one bin', np.allclose(a, b, rtol=1e-12, atol=0), f'{a} {b}')


def test_cell_edges():
    c = fx()
    g = c['grid'].select('charged')
    for lo in (0.0, 0.4):
        p = extract_measurement_params(rec('particle_yields', lo))
        d3 = pd.prepare_yield_3d(g, p)
        d2 = pd.prepare_yield_2d(c['qv'], p)
        _, ty = truth(lo, 'eta', 'charged', None)
        v = run(rec('particle_yields', lo), '3d')[0][0]
        check(f'pT > {lo} (cell edges): no block; 3D eta yield = particle count', 'pt_subcell' not in d3
              and 'pt_subcell' not in d2 and abs(v / ty - 1) < 1e-12, f'{v} vs {ty}')


def test_jackknife():
    for kind in ('3d', '2d'):
        v, e = run(rec('mean_pT', 0.27, 'eta', bins=[[-0.8, 0.0], [0.0, 0.8]]), kind, jk=True)
        check(f'{kind} jackknife with the block: finite errors', np.all(np.isfinite(e)) and np.all(e > 0), f'{v} +- {e}')


def test_et_flat_share():
    g = fx()['grid'].select('charged')
    p = extract_measurement_params({'observable_type': 'transverse_energy', 'value_type': 'dET/deta',
                                    'hepdata_file': 't.yaml', 'inspire_id': '0', 'version': 1,
                                    'collision_system': {'reaction': 'AU AU', 'energy_gev': 200.0},
                                    'particle_id': {'species': 'charged'},
                                    'centrality': {'classes': ['0-100%'], 'eta': [{'min': -3.0, 'max': 3.0}]},
                                    'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8},
                                                        'pT': {'min': 0.27, 'max': 3.0}}})
    got = pd.prepare_dET_3d(g, p)['sum_et']
    fe = np.clip((np.minimum(g.eta_edges[1:], 0.8) - np.maximum(g.eta_edges[:-1], -0.8)) / 0.2, 0, 1)
    fp = np.clip((np.minimum(g.pt_edges[1:], 3.0) ** 2 - np.maximum(g.pt_edges[:-1], 0.27) ** 2)
                 / np.diff(g.pt_edges ** 2), 0, 1)
    want = np.einsum('ijk,j,k->i', g.sum_ET, fe, fp) / g.nsamples
    check('3D dE_T/deta at pT > 0.27: the cut cell at its flat share', np.allclose(got, want, rtol=1e-12))


if __name__ == '__main__':
    for t in (test_window_shares, test_accuracy, test_integrated_equals_one_bin, test_cell_edges, test_jackknife,
              test_et_flat_share):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
