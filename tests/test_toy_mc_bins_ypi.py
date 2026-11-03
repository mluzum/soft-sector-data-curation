"""Yields and <pT> per eta or y_pi bin, and in windows of rapidity with an assumed mass.

ATLAS O+O (3170344) tabulates dn/deta(eta), dn/dy_pi(y_pi) and <pT>(eta),
<pT>(y_pi) per class, y_pi the rapidity of every particle with the pion
mass.  Toy events (pions, kaons, protons, flat in true rapidity over
|y| < 3) through the production path, against particle-level sums:
  - dn/deta per eta bin and dn/dy_pi per y_pi bin, 2D and 3D;
  - dn/dy_pi and <pT> in |y_pi| < 0.5, and <pT> per eta bin;
  - the bins of a per-bin yield add up to the yield of their union;
  - integrated <pT> and yields in eta windows unchanged by the per-bin path.
Run: python3 tests/test_toy_mc_bins_ypi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, make_qvector_data, evaluate  # noqa: E402

M_PI = 0.13957
PT = (0.3, 3.0)
ETA_BINS = [[-2.0, -1.5], [-1.5, -1.0], [-0.5, 0.0], [0.0, 0.5], [1.0, 1.5]]
Y_BINS = [[-2.0, -1.5], [-0.5, 0.0], [0.0, 0.5], [1.5, 2.0]]
results = []
_CACHE = {}


def check(name, got, want, tol):
    got, want = np.atleast_1d(got), np.atleast_1d(want)
    ok = np.all(np.isfinite(got)) and np.all(np.abs(got - want) <= tol * np.abs(want))
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}: worst {np.max(np.abs(got / want - 1)):.3f} (tol {tol})")


def fx():
    if not _CACHE:
        cfg = ToyGenConfig(n_events=600, mult_mean=3000, mult_sigma=200, rap_max=3.0, seed=77, pt_min=0.3, pt_max=3.0)
        ev, tr = generate_events(cfg)
        _CACHE.update(ev=ev, d3=make_qgrid_data(ev, tr, n_eta=80, n_pt=27, n_harm=3, eta_half=4.0),
                      d2=make_qvector_data(ev, tr, n_pt=54, n_harm=3))
    return _CACHE


def rec(otype, kr, dim=None):
    r = {'observable_type': otype, 'inspire_id': '0', 'version': 1, 'hepdata_file': 'toy.yaml',
         'value_type': {'particle_yields': 'dNch_deta', 'mean_pT': 'mean_pT'}[otype],
         'particle_id': {'species': 'charged'},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -4.0, 'max': -3.2}, {'min': 3.2, 'max': 4.0}]},
         'kinematic_range': dict(kr, pT={'min': PT[0], 'max': PT[1]})}
    if dim:
        r['dimensionality'] = dim
    return r


def run(r, kind):
    c = fx()['d3' if kind == '3d' else 'd2']
    kw = dict(grid=c) if kind == '3d' else dict(qv=c)
    return np.asarray(evaluate(r, cent_method='simple', with_jackknife=False, **kw)[0])


def truth(var, windows, what):
    """Per window: dN/dvar per event (charged: positives x 2), or <pT>."""
    n, s, out = [], [], []
    for lo, hi in windows:
        cnt = spt = 0.0
        for e in fx()['ev']:
            x = e['eta'] if var == 'eta' else np.arcsinh(e['pt'] * np.sinh(e['eta']) / np.sqrt(e['pt'] ** 2 + M_PI ** 2))
            sel = (x >= lo) & (x < hi) & (e['pt'] >= PT[0]) & (e['pt'] < PT[1])
            cnt += sel.sum()
            spt += e['pt'][sel].sum()
        out.append(2 * cnt / len(fx()['ev']) / (hi - lo) if what == 'yield' else spt / cnt)
    return np.array(out)


def test_per_bin_yields():
    for kind, tol in (('3d', 0.02), ('2d', 0.03)):
        v = run(rec('particle_yields', {'eta': {'min': -2, 'max': 1.5, 'bin_ranges': ETA_BINS}}, 'differential_eta'), kind)
        check(f'{kind} dn/deta per eta bin', v, truth('eta', ETA_BINS, 'yield'), tol)
        v = run(rec('particle_yields', {'y': {'min': -2, 'max': 2, 'mass': M_PI, 'bin_ranges': Y_BINS}}), kind)
        check(f'{kind} dn/dy_pi per y_pi bin', v, truth('y', Y_BINS, 'yield'), tol)


def test_ypi_windows():
    w = [[-0.5, 0.5]]
    for kind, tol in (('3d', 0.02), ('2d', 0.03)):
        v = run(rec('particle_yields', {'y': {'min': -0.5, 'max': 0.5, 'mass': M_PI}}), kind)
        check(f'{kind} dn/dy_pi in |y_pi| < 0.5', v, truth('y', w, 'yield'), tol)
        v = run(rec('mean_pT', {'y': {'min': -0.5, 'max': 0.5, 'mass': M_PI}}), kind)
        check(f'{kind} <pT> in |y_pi| < 0.5', v, truth('y', w, 'mean'), 0.01)


def test_mean_pt_per_bin():
    for kind in ('3d', '2d'):
        v = run(rec('mean_pT', {'eta': {'min': -2, 'max': 1.5, 'bin_ranges': ETA_BINS}}, 'differential_eta'), kind)
        check(f'{kind} <pT> per eta bin', v, truth('eta', ETA_BINS, 'mean'), 0.01)
        v = run(rec('mean_pT', {'y': {'min': -2, 'max': 2, 'mass': M_PI, 'bin_ranges': Y_BINS}}, 'differential_y'), kind)
        check(f'{kind} <pT> per y_pi bin', v, truth('y', Y_BINS, 'mean'), 0.01)


def test_consistency():
    for kind in ('3d', '2d'):
        bins = [[-0.5, 0.0], [0.0, 0.5]]
        per = run(rec('particle_yields', {'eta': {'min': -0.5, 'max': 0.5, 'bin_ranges': bins}}, 'differential_eta'), kind)
        whole = run(rec('particle_yields', {'eta': {'min': -0.5, 'max': 0.5}}), kind)
        ok = np.isclose(0.5 * per.sum(), whole[0], rtol=1e-10)
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'} {kind} per-bin yields add up to the window's yield "
              f"({0.5 * per.sum():.4f} vs {whole[0]:.4f})")
        pb = run(rec('mean_pT', {'eta': {'min': -0.5, 'max': 0.5, 'bin_ranges': [[-0.5, 0.5]]}}, 'differential_eta'), kind)
        pi = run(rec('mean_pT', {'eta': {'min': -0.5, 'max': 0.5}}), kind)
        ok = np.isclose(pb[0], pi[0], rtol=1e-10)
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'} {kind} <pT>: one bin = the integrated window ({pb[0]:.5f} vs {pi[0]:.5f})")


if __name__ == '__main__':
    for t in (test_per_bin_yields, test_ypi_windows, test_mean_pt_per_bin, test_consistency):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
