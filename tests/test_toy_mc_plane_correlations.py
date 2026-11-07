"""Symmetry-plane correlations, SP method (correlation_type plane_correlation) through the production path.

  - the distinct-particle powers of flow vectors (flow_pt.distinct_gf,
    a generating function) equal explicit sums over ordered tuples of
    distinct particles, up to six members, with overlapping (2D-like)
    window weights, and the set-partition sums (flow_pt.distinct) up to four;
  - toy events with V_4 = V_4^L + chi V_2^2 and V_5 = V_5^L + chi V_2 V_3
    (ToyGenConfig.nonlinear): <cos 4(Phi_2 - Phi_4)> (two subevents) and
    <cos(2 Phi_2 + 3 Phi_3 - 5 Phi_5)> (three subevents, the three types)
    recover <v_2^2 v_4 cos 4(Psi_2 - Psi_4)> / sqrt(<v_2^4><v_4^2>) and
    <v_2 v_3 v_5 cos(...)> / sqrt(<v_2^2><v_3^2><v_5^2>) of the generated
    events, 2D and 3D;
  - without coupling <cos 4(Phi_2 - Phi_4)> is consistent with zero;
  - the Gaussian Estimator (estimator GE, one window: ALICE), sqrt(pi/4)
    N / sqrt(D): equal to explicit sums over distinct tuples on a small 3D
    sample (<cos 4(Psi_4 - Psi_2)>, <cos(2 Psi_2 + 3 Psi_3 - 5 Psi_5)>), and
    recovering sqrt(pi/4) <v_4 v_2^2 cos 4(Psi_4 - Psi_2)> / sqrt(<v_4^2
    v_2^4>), the three-plane and the twelve-particle <cos 8(Psi_4 - Psi_2)>
    of the generated events, 2D and 3D;
  - the EP method, planes with sum c n != 0, missing subevents and harmonics
    the container does not store are refused.
Run: python3 tests/test_toy_mc_plane_correlations.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.flow_pt import CellSums, distinct, distinct_gf  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data, evaluate,
)

TWO = {'P': [{'min': 0.5, 'max': 4.8}], 'N': [{'min': -4.8, 'max': -0.5}]}
THREE = {'A': [{'min': 0.5, 'max': 2.7}], 'B': [{'min': -4.8, 'max': -3.3}, {'min': 3.3, 'max': 4.8}],
         'C': [{'min': -2.7, 'max': -0.5}]}
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx(coupled=True):
    key = ('fx', coupled)
    if key not in _CACHE:
        flow = {2: dict(kind='const', value=0.09), 3: dict(kind='const', value=0.04),
                4: dict(kind='const', value=0.015), 5: dict(kind='const', value=0.008)}
        cfg = ToyGenConfig(n_events=5000, mult_mean=2500, mult_sigma=300, seed=78, rap_max=5.0, flow=flow,
                           fluctuate={2: 0.04, 3: 0.02, 4: 0.008, 5: 0.004},
                           nonlinear={4: (1.2, 2, 2), 5: (1.6, 2, 3)} if coupled else None)
        ev, tr = generate_events(cfg)
        v, ps = tr['vn_ebe'], tr['psi']
        ge = np.sqrt(np.pi / 4.0)
        truth = {'ge_42': ge * np.mean(v[4] * v[2] ** 2 * np.cos(4 * (ps[4] - ps[2])))
                 / np.sqrt(np.mean(v[4] ** 2 * v[2] ** 4)),
                 'ge_235': ge * np.mean(v[2] * v[3] * v[5] * np.cos(2 * ps[2] + 3 * ps[3] - 5 * ps[5]))
                 / np.sqrt(np.mean(v[2] ** 2 * v[3] ** 2 * v[5] ** 2)),
                 'ge_8_42': ge * np.mean(v[4] ** 2 * v[2] ** 4 * np.cos(8 * (ps[4] - ps[2])))
                 / np.sqrt(np.mean(v[4] ** 4 * v[2] ** 8)),
                 'two': np.mean(v[2] ** 2 * v[4] * np.cos(4 * (ps[2] - ps[4])))
                 / np.sqrt(np.mean(v[2] ** 4) * np.mean(v[4] ** 2)),
                 'three': np.mean(v[2] * v[3] * v[5] * np.cos(2 * ps[2] + 3 * ps[3] - 5 * ps[5]))
                 / np.sqrt(np.mean(v[2] ** 2) * np.mean(v[3] ** 2) * np.mean(v[5] ** 2))}
        _CACHE[key] = {'2d': make_qvector_data(ev, tr, n_pt=20, n_harm=9),
                       '3d': make_qgrid_data(ev, tr, n_eta=50, n_pt=20, n_harm=9, eta_half=5.0), 'truth': truth}
    return _CACHE[key]


def rec(planes, subs, **extra):
    r = {'observable_type': 'flow_harmonic_correlations', 'correlation_type': 'plane_correlation',
         'harmonics': [n for n, _ in planes], 'planes': [{'harmonic': n, 'coefficient': c} for n, c in planes],
         'estimator': 'SP', 'plane_subevents': subs, 'value_type': 'toy', 'hepdata_file': 'toy.yaml',
         'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -4.8, 'max': 4.8}, 'pT': {'min': 0.0, 'max': 100.0}},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}}
    r.update(extra)
    return r


def run(r, kind, coupled=True, jk=True):
    c = _fx(coupled)[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def test_distinct_gf_exact():
    rng = np.random.default_rng(1)
    n_ev, n_cell, H = 3, 4, 13
    parts = [[(int(rng.integers(n_cell)), rng.uniform(0, 2 * np.pi)) for _ in range(rng.integers(6, 9))]
             for _ in range(n_ev)]
    Qn = np.zeros((n_ev, n_cell, H), complex)
    for e, ps in enumerate(parts):
        for c, phi in ps:
            Qn[e, c, :] += np.exp(1j * np.arange(H) * phi)
    c = Qn[..., 0].real
    wins = {'P': rng.uniform(0, 1, n_cell), 'N': rng.uniform(0, 1, n_cell)}   # overlapping shares (2D-like)
    S = CellSums(Qn, c, c, c, wins, np.ones(n_ev))

    def brute(members):
        out = np.zeros(n_ev, complex)
        for e, ps in enumerate(parts):
            for tup in itertools.permutations(range(len(ps)), len(members)):
                val = 1.0
                for (w, h), i in zip(members, tup):
                    cc, phi = ps[i]
                    val = val * wins[w][cc] * np.exp(1j * h * phi)
                out[e] += val
        return out
    for types in ([('P', 2, 2), ('N', -4, 1)], [('P', 2, 1), ('P', 3, 1), ('N', -5, 1)],
                  [('P', 2, 2), ('N', -2, 2)], [('P', 2, 3), ('N', -6, 1)],
                  [('P', 2, 2), ('P', 3, 1), ('N', -2, 1), ('N', -5, 1)], [('P', 2, 4), ('N', -4, 2)]):
        members = [(w, h) for w, h, cnt in types for _ in range(cnt)]
        gf, bf = distinct_gf(S, types), brute(members)
        ok = np.allclose(gf, bf, rtol=1e-10, atol=1e-10)
        if len(members) <= 4:
            ok = ok and np.allclose(distinct(S, [(w, h, 0) for w, h in members]), bf, rtol=1e-10, atol=1e-10)
        check(f'{len(members)} members {types}: generating function = explicit distinct tuples', ok)


def test_truth():
    for kind in ('3d', '2d'):
        for planes, subs, lay in (([(2, 2), (4, -1)], TWO, 'two'), ([(2, 1), (3, 1), (5, -1)], THREE, 'three')):
            v, e = run(rec(planes, subs), kind)
            t = _fx()['truth'][lay]
            check(f'{kind} {planes}: {v[0]:.4f} +- {e[0]:.4f} vs generated {t:.4f}', abs(v[0] - t) < 3 * e[0] + 0.02)


def test_no_coupling():
    for kind in ('3d', '2d'):
        v, e = run(rec([(2, 2), (4, -1)], TWO), kind, coupled=False)
        check(f'{kind}: no coupling, <cos 4(Phi2 - Phi4)> consistent with 0', abs(v[0]) < 4 * e[0] + 0.03,
              f'{v} +- {e}')


def test_refused():
    check('dispatches to plane_corr_sp', resolve(rec([(2, 2), (4, -1)], TWO))['compute'] == 'plane_corr_sp')
    for bad in (dict(estimator='EP'), dict(planes=[{'harmonic': 2, 'coefficient': 2}, {'harmonic': 4,
                                                                                            'coefficient': -2}]),
                dict(plane_subevents={'P': TWO['P']})):
        r = rec([(2, 2), (4, -1)], TWO)
        r.update(bad)
        try:
            evaluate(r, grid=_fx()['3d'], cent_method='simple', with_jackknife=False)
            ok = False
        except ValueError:
            ok = True
        check(f'refused {list(bad)}', ok)
    try:
        evaluate(rec([(3, 4), (12, -1)], TWO), grid=_fx()['3d'], cent_method='simple', with_jackknife=False)
        ok = False
    except ValueError:
        ok = True
    check('refused a plane harmonic the container does not store (Q_12)', ok)


def rec_ge(planes, eta=(-2.0, 2.0), **extra):
    r = rec(planes, None, estimator='GE',
            kinematic_range={'eta': {'min': eta[0], 'max': eta[1]}, 'pT': {'min': 0.0, 'max': 100.0}})
    del r['plane_subevents']
    r.update(extra)
    return r


def _brute_ge(events, planes, eta, max_ev=4):
    """sqrt(pi/4) N / sqrt(D) from explicit ordered tuples of distinct
    particles in the window, events weighted by their tuple counts."""
    num = [n if c > 0 else -n for n, c in planes for _ in range(abs(c))]
    den = [n for n, c in planes for _ in range(abs(c))] + [-n for n, c in planes for _ in range(abs(c))]
    acc = {}
    for ev in events[:max_ev]:
        phi = ev['phi'][(ev['eta'] >= eta[0]) & (ev['eta'] < eta[1])]
        for key, hs in (('N', num), ('D', den)):
            e = np.exp(1j * np.outer(hs, phi))                  # (members, particles)
            s, n = 0j, 0
            for t in itertools.permutations(range(len(phi)), len(hs)):
                s += np.prod(e[np.arange(len(hs)), t])
                n += 1
            acc[key] = acc.get(key, 0j) + s
            acc['W' + key] = acc.get('W' + key, 0) + n
    d = acc['D'].real / acc['WD']
    return np.sqrt(np.pi / 4.0) * (acc['N'].real / acc['WN']) / (np.sign(d) * np.sqrt(abs(d)))


def test_ge_exact():
    cfg = ToyGenConfig(n_events=4, mult_mean=40, mult_sigma=3, seed=31, rap_max=1.6,
                       flow={2: dict(kind='const', value=0.15), 3: dict(kind='const', value=0.08),
                             4: dict(kind='const', value=0.05), 5: dict(kind='const', value=0.04)},
                       nonlinear={4: (1.0, 2, 2), 5: (1.0, 2, 3)})
    events, truth = generate_events(cfg)
    grid = make_qgrid_data(events, truth, n_eta=32, n_pt=30, n_harm=13, eta_half=1.6)
    edges = np.linspace(-1.6, 1.6, 33)
    eta = (edges[12], edges[20])
    for planes in ([(4, 1), (2, -2)], [(2, 1), (3, 1), (5, -1)]):
        r = rec_ge(planes, eta=eta, centrality={'classes': ['0-100%'], 'eta': [{'min': -3.9, 'max': -3.0}]},
                   kinematic_range={'eta': {'min': eta[0], 'max': eta[1]}, 'pT': {'min': 0.0, 'max': 3.0}})
        v, _ = evaluate(r, grid=grid, cent_method='simple', with_jackknife=False)
        want = _brute_ge([dict(ev, phi=ev['phi'][ev['pt'] < 3.0], eta=ev['eta'][ev['pt'] < 3.0]) for ev in events],
                         planes, eta)
        check(f'3D GE {planes} = explicit tuple sums', np.isclose(v[0], want, rtol=1e-8), f'{v[0]:.8f} vs {want:.8f}')


def test_ge_truth():
    check('dispatches GE to plane_corr_ge', resolve(rec_ge([(4, 1), (2, -2)]))['compute'] == 'plane_corr_ge')
    for kind in ('3d', '2d'):
        for planes, key in (([(4, 1), (2, -2)], 'ge_42'), ([(2, 1), (3, 1), (5, -1)], 'ge_235'),
                            ([(4, 2), (2, -4)], 'ge_8_42')):
            v, e = run(rec_ge(planes), kind)
            t = _fx()['truth'][key]
            check(f'{kind} GE {planes}: {v[0]:.4f} +- {e[0]:.4f} vs generated {t:.4f}',
                  abs(v[0] - t) < 3 * e[0] + 0.02)
    v, e = run(rec_ge([(4, 1), (2, -2)]), '3d', coupled=False)
    check('3D GE, no coupling: consistent with 0', abs(v[0]) < 4 * e[0] + 0.03, f'{v} +- {e}')
    for bad in (dict(plane_subevents=TWO), dict(planes=[{'harmonic': 4, 'coefficient': 1},
                                                        {'harmonic': 2, 'coefficient': -1}])):
        r = rec_ge([(4, 1), (2, -2)])
        r.update(bad)
        try:
            evaluate(r, grid=_fx()['3d'], cent_method='simple', with_jackknife=False)
            ok = False
        except ValueError:
            ok = True
        check(f'GE refused {list(bad)}', ok)


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
