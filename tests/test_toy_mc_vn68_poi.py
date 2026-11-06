"""Differential six- and eight-particle cumulants of a particle of interest (v_n{6}(pT), v_n{8}(pT)).

  - <<6'>> and <<8'>> (one POI, the others reference particles, all
    distinct; the POI a sub-window of the reference, so a POI particle is
    also a reference particle) equal explicit sums over ordered tuples on a
    3D grid;
  - with the POI equal to the reference (one bin over the reference pT
    range), v'{6} and v'{8} equal the integrated reference v{6}, v{8} (2D
    and 3D);
  - v2(pT) generated with a linear profile and no fluctuations: v2{6}(pT)
    and v2{8}(pT) follow it, 2D and 3D, as v2{4}(pT) does.
Run: python3 tests/test_toy_mc_vn68_poi.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.correlators import Container3D, correlator_arrays  # noqa: E402
from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data, evaluate, vn_of_pt,
)

BINS = [[0.3, 0.6], [0.6, 1.0], [1.0, 1.5], [1.5, 2.2], [2.2, 3.0]]
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def rec(order, bins=BINS, kind='differential_pT'):
    r = {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': order, 'harmonic': 2,
         'value_type': f'v2{{{order}}}', 'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1,
         'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                             'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
         'flow_reference': {'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.3, 'max': 3.0}},
         'dimensionality': kind,
         'centrality': {'classes': ['0-50%', '50-100%'], 'eta': [{'min': -1.2, 'max': 1.2}]}}
    if kind == 'integrated':
        r['kinematic_range'] = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.3, 'max': 3.0}}
        r.pop('flow_reference')
    return r


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=1500, mult_mean=1500, mult_sigma=150, seed=686, rap_max=1.2,
                           flow={2: dict(kind='linear', value=0.06, pt0=1.0), 3: dict(kind='const', value=0.03)})
        events, truth = generate_events(cfg)
        _CACHE['fx'] = {'2d': make_qvector_data(events, truth, n_pt=30, n_harm=9, pt_max=3.0),
                        '3d': make_qgrid_data(events, truth, n_eta=24, n_pt=30, n_harm=9, eta_half=1.2,
                                              pt_max=3.0),
                        'events': events}
    return _CACHE['fx']


def run(r, kind, jk=True):
    c = _fx()[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def test_dispatch():
    for k in (6, 8):
        check(f'v2{{{k}}}(pT) dispatches', resolve(rec(k))['compute'] == f'vn{k}_single_diff_pT')
        check(f'v2{{{k}}}(eta) dispatches', resolve(rec(k, kind='differential_eta') | {
            'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0, 'bin_ranges': [[-1, 0], [0, 1]]},
                                'pT': {'min': 0.3, 'max': 3.0}}})['compute'] == f'vn{k}_single_diff_eta')


def _brute(parts, poi, ref, h, m):
    """Ordered tuples of distinct particles: the first a POI, the other 2m - 1
    reference particles; e^{i h (phi_1 + .. + phi_m - phi_{m+1} - .. - phi_2m)}."""
    num = den = 0.0
    signs = np.array([1] * m + [-1] * m)
    for i in poi:
        others = [j for j in ref if j != i]
        for tup in itertools.permutations(others, 2 * m - 1):
            phis = parts[[i, *tup]]
            num += np.exp(1j * h * np.dot(signs, phis))
            den += 1
    return num, den


def test_exact_tuples():
    # events of eight particles in the reference window (cell edges), the POI
    # those in the lowest six pT cells: the generating-function sums against
    # explicit ordered tuples
    cfg = ToyGenConfig(n_events=4, mult_mean=60, mult_sigma=1, seed=5, rap_max=1.2,
                       flow={2: dict(kind='const', value=0.2)})
    events, truth = generate_events(cfg)
    eta_e, pt_e = np.linspace(-1.2, 1.2, 25), np.linspace(cfg.pt_min, 3.0, 31)   # the grid's cells
    rw = (eta_e[0], eta_e[24], pt_e[0], pt_e[30])
    pw = (eta_e[0], eta_e[24], pt_e[0], pt_e[6])
    small = []
    for ev in events:
        n = len(ev['phi'])
        sel = np.nonzero((np.abs(ev['eta']) < 1.2) & (ev['pt'] >= cfg.pt_min) & (ev['pt'] < 3.0))[0][:8]
        small.append({k: (v[sel] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in ev.items()})
    grid = make_qgrid_data(small, truth, n_eta=24, n_pt=30, n_harm=9, eta_half=1.2, pt_max=3.0)
    arr = correlator_arrays(Container3D(grid, pw, grid, rw, True), 9, 8, which=('poi',))
    for e, ev in enumerate(small):
        poi = [i for i in range(len(ev['phi'])) if ev['pt'][i] < pw[3]]
        for m in (3, 4):
            num, den = _brute(ev['phi'], poi, list(range(len(ev['phi']))), 2, m)
            ok = np.isclose(arr[f'N{2 * m}p'][e, 1], num, rtol=1e-9, atol=1e-9) and \
                np.isclose(arr[f'D{2 * m}p'][e], den, rtol=1e-12)
            check(f"event {e} ({len(poi)} POI of 8): <{2 * m}'> = explicit sum over {den:.0f} tuples", ok,
                  f"{arr[f'N{2 * m}p'][e, 1]:.6f} vs {num:.6f}")


def test_poi_equals_reference():
    for kind in ('2d', '3d'):
        for k in (6, 8):
            vd, _ = run(rec(k, bins=[[0.3, 3.0]]), kind, jk=False)
            vi, _ = run(rec(k, kind='integrated'), kind, jk=False)
            check(f"{kind}: POI = reference, v'2{{{k}}} = v2{{{k}}}", np.allclose(vd, vi, rtol=1e-8, atol=1e-12),
                  f'{vd} vs {vi}')


def test_profile():
    centers = np.array([0.5 * (a + b) for a, b in BINS])
    want = vn_of_pt(centers, 'linear', 0.06, 1.0)
    for kind in ('3d', '2d'):
        for k in (4, 6, 8):
            v, e = run(rec(k), kind)
            v, e = v.reshape(2, -1), e.reshape(2, -1)
            ok = np.all(np.abs(v - want) < 3 * e + 0.06 * want)
            check(f'{kind}: v2{{{k}}}(pT) follows the generated profile', ok, f'{v.round(4)} +- {e.round(4)} '
                  f'vs {want.round(4)}')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    n_ok = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
            n_ok += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
    print(f'\n{n_ok}/{len(tests)} passed')
    sys.exit(0 if n_ok == len(tests) else 1)
