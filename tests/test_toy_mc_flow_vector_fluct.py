"""pT-dependent flow-vector fluctuations (correlation_type pT_flow_vector_fluctuations) through the production path.

Toy events whose flow angle turns with pT, Psi_2(pT) = Psi_2 + delta_e (pT
- pt0) (ToyGenConfig.plane_twist), and whose v_2 magnitude fluctuates:
  - on a 3D grid (windows on cell edges), v2{2}/v2[2], r_2, A_2^f and M_2^f
    agree with their expectation given the generated events: the same
    correlators with each e^{i n phi} replaced by the particle's flow vector
    v_n(pT) e^{i n Psi_n(pT)} (distinct particles, the same windows and tuple
    weights), and they depart from 1;
  - without the twist all four are 1 within errors, 2D and 3D;
  - angle_lower_limit = sqrt((A_f + 1)/2), magnitude_upper_limit = v_ratio /
    angle_lower_limit;
  - records without the subevents, trigger or bins they need are refused.
Run: python3 tests/test_toy_mc_flow_vector_fluct.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data, evaluate, vn_of_pt, _vn_profile_scale,
)

BINS = [[0.2, 0.6], [0.6, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 3.0]]
TRIG = (0.2, 0.6)
SUB2 = {'A': (-0.8, -0.4), 'B': (0.4, 0.8)}
SUB4 = {'A': (-0.8, 0.0), 'B': (0.0, 0.8)}
REF_PT = (0.2, 5.0)
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def rec(quantity, **extra):
    r = {'observable_type': 'flow_harmonic_correlations', 'correlation_type': 'pT_flow_vector_fluctuations',
         'harmonics': [2], 'quantity': quantity, 'value_type': quantity, 'hepdata_file': 'toy.yaml',
         'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'charged'}, 'dimensionality': 'differential_pT',
         'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 3.0, 'bin_ranges': BINS}},
         'flow_reference': {'eta': [{'min': -0.8, 'max': 0.8}], 'pT': {'min': REF_PT[0], 'max': REF_PT[1]}},
         'particles': 'AB',
         'subevents': {k: {'kinematic_range': {'eta': {'min': lo, 'max': hi}}} for k, (lo, hi) in SUB2.items()},
         'four_particle_subevents': {k: {'eta': {'min': lo, 'max': hi}} for k, (lo, hi) in SUB4.items()},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -1.2, 'max': 1.2}]}}
    if quantity == 'r':
        r['trigger_pT'] = {'min': TRIG[0], 'max': TRIG[1]}
    r.update(extra)
    return r


def _fx(twist=True):
    key = ('fx', twist)
    if key not in _CACHE:
        cfg = ToyGenConfig(n_events=1200, mult_mean=4000, mult_sigma=200, seed=2403, rap_max=1.2, pt_min=0.2,
                           pt_max=5.0, flow={2: dict(kind='linear', value=0.05, pt0=1.0)},
                           fluctuate={2: 0.02}, plane_twist={2: (0.25, 0.8)} if twist else None)
        events, truth = generate_events(cfg)
        _CACHE[key] = {'events': events, 'truth': truth,
                       '3d': make_qgrid_data(events, truth, n_eta=24, n_pt=48, n_harm=5, eta_half=1.2, pt_max=5.0),
                       '2d': make_qvector_data(events, truth, n_pt=48, n_harm=5, pt_max=5.0)}
    return _CACHE[key]


def run(r, kind, twist=True, jk=True):
    c = _fx(twist)[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def _expected(fx, quantity, n=2):
    """The correlators with e^{i n phi} -> the particle's flow vector u, over
    distinct particles of the same windows, events weighted by tuple counts."""
    cfg, tr = fx['truth']['cfg'], fx['truth']
    fl = cfg.flow[n]
    sig_pt0 = cfg.plane_twist.get(n)
    acc = {}

    def add(key, num, cnt):
        acc[key] = acc.get(key, 0.0) + num
        acc['W' + key] = acc.get('W' + key, 0.0) + cnt

    for e, ev in enumerate(fx['events']):
        eta, pt = ev['eta'], ev['pt']
        v = vn_of_pt(pt, fl['kind'], fl['value'], fl.get('pt0')) * tr['vn_ebe'][n][e] / _vn_profile_scale(fl)
        psi = tr['psi'][n][e] + (tr['plane_twist'][n][e] * (pt - sig_pt0[1]) if sig_pt0 else 0.0)
        u = v * np.exp(1j * n * psi)

        def sel(eta_w, pt_w):
            m = (eta >= eta_w[0]) & (eta < eta_w[1]) & (pt >= pt_w[0]) & (pt < pt_w[1])
            return u[m], m

        def two(x, y):            # disjoint windows
            return np.real(np.sum(x[0]) * np.conj(np.sum(y[0]))), x[1].sum() * y[1].sum()

        def pair_same(x):         # distinct ordered pairs in one window, both +n
            s1, s2 = np.sum(x[0]), np.sum(x[0] ** 2)
            return s1 ** 2 - s2, x[1].sum() * (x[1].sum() - 1)

        def pair_sub(xb, x):      # i in xb (a subset of x), j in x, i != j
            return np.sum(xb[0]) * np.sum(x[0]) - np.sum(xb[0] ** 2), xb[1].sum() * x[1].sum() - xb[1].sum()

        for X, Y in (('A', 'B'), ('B', 'A')):
            rx, ry = sel(SUB2[X], REF_PT), sel(SUB2[Y], REF_PT)
            add('ref2', *two(rx, ry))
            tx, ty = sel(SUB2[X], TRIG), sel(SUB2[Y], TRIG)
            add('tt2', *two(tx, ty))
            for b, pb in enumerate(BINS):
                bx, by = sel(SUB2[X], pb), sel(SUB2[Y], pb)
                add(f'pp2_{b}', *two(bx, by))
                add(f'pr2_{b}', *two(bx, ry))
                add(f'pt2_{b}', *two(bx, ty))
        for U, V in (('A', 'B'), ('B', 'A')):
            ru, rv = sel(SUB4[U], REF_PT), sel(SUB4[V], REF_PT)
            add('ref2cd', *two(ru, rv))
            (pu, cu), (pv, cv) = pair_same(ru), pair_same(rv)
            add('ref4', np.real(pu * np.conj(pv)), cu * cv)
            for b, pb in enumerate(BINS):
                bu, bv = sel(SUB4[U], pb), sel(SUB4[V], pb)
                (nu, ncu) = pair_same(bu)
                add(f'num4_{b}', np.real(nu * np.conj(pv)), ncu * cv)
                (du, dcu), (dv, dcv) = pair_sub(bu, ru), pair_sub(bv, rv)
                add(f'den4_{b}', np.real(du * np.conj(dv)), dcu * dcv)
                add(f'pp2cd_{b}', *two(bu, bv))

    def a(k):
        return acc[k] / acc['W' + k]
    out = []
    for b in range(len(BINS)):
        vr = a(f'pr2_{b}') / np.sqrt(a('ref2') * a(f'pp2_{b}'))
        af = a(f'num4_{b}') / a(f'den4_{b}')
        out.append({'v_ratio': vr, 'r': a(f'pt2_{b}') / np.sqrt(a(f'pp2_{b}') * a('tt2')), 'A_f': af,
                    'M_f': (a(f'den4_{b}') / (a(f'pp2cd_{b}') * a('ref2cd'))) / (a('ref4') / a('ref2cd') ** 2)}
                   [quantity])
    return np.array(out)


def test_dispatch_and_refusals():
    check('dispatches to flow_vector_fluct', resolve(rec('A_f'))['compute'] == 'flow_vector_fluct')
    for q, drop in (('r', 'trigger_pT'), ('A_f', 'four_particle_subevents'), ('v_ratio', 'particles')):
        r = rec(q)
        r.pop(drop)
        try:
            resolve(r)
            ok = False
        except ValueError:
            ok = True
        check(f'{q} without {drop} refused', ok)
    try:
        resolve(rec('A_f', dimensionality='integrated'))
        ok = False
    except ValueError:
        ok = True
    check('integrated refused (POI pT bins needed)', ok)


def test_expectation_3d():
    fx = _fx(True)
    for q in ('v_ratio', 'r', 'A_f', 'M_f'):
        v, e = run(rec(q), '3d')
        want = _expected(fx, q)
        ok = np.all(np.abs(v - want) < 3 * e + 0.005)
        check(f'3D {q}: measured = expectation from the generated flow vectors', ok,
              f'{v.round(4)} +- {e.round(4)} vs {want.round(4)}')
        if q in ('v_ratio', 'r', 'A_f'):
            b = int(np.argmax(np.where(e > 0, (1 - want) / np.where(e > 0, e, 1.0), -np.inf)))
            check(f'3D {q}: below 1 where the angle turns (bin {b})', want[b] < 0.97 and v[b] < 1 - 3 * e[b],
                  f'{want[b]:.4f}, {v[b]:.4f} +- {e[b]:.4f}')


def test_unity_without_twist():
    for kind in ('3d', '2d'):
        for q in ('v_ratio', 'r', 'A_f', 'M_f'):
            v, e = run(rec(q), kind, twist=False)
            check(f'{kind} {q} = 1 without pT-dependent fluctuations', np.all(np.abs(v - 1) < 3 * e + 0.01),
                  f'{v.round(4)} +- {e.round(4)}')


def test_limits():
    for kind in ('3d', '2d'):
        af = run(rec('A_f'), kind, jk=False)[0]
        vr = run(rec('v_ratio'), kind, jk=False)[0]
        lo = run(rec('angle_lower_limit'), kind, jk=False)[0]
        up = run(rec('magnitude_upper_limit'), kind, jk=False)[0]
        ok = np.allclose(lo, np.sqrt((af + 1) / 2), rtol=1e-12) and np.allclose(up, vr / lo, rtol=1e-12)
        check(f'{kind}: angle_lower_limit = sqrt((A_f + 1)/2), magnitude_upper_limit = v_ratio / it', ok)


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
