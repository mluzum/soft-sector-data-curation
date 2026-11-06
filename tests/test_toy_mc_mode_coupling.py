"""Linear and mode-coupled flow harmonics (correlation_type mode_coupling) through the production path.

Toy events with V_4 = V_4^L + chi_{4,22} V_2^2 and V_5 = V_5^L + chi_{5,23}
V_2 V_3 (ToyGenConfig.nonlinear; the linear parts with random angles),
fluctuating v_2 and v_3:
  - the two-subevent C_{4,22} and C_{5,23} equal explicit sums over
    particle triplets (1 in A, 2 != 3 in B, both orientations) on a few
    small events, 3D and 2D;
  - chi_{4,22} and chi_{5,23} recover the injected coefficients (2D, 3D,
    two subevents and one window);
  - v_inclusive, v_mode_coupled, v_linear and rho obey their definitions
    against the separately computed C, chi and v_k{2};
  - pT-differential: chi(pT) flat at the injected value (flat v_n(pT)),
    and the differential C summed over the bins' tuples gives the
    integrated C;
  - without coupling, chi is consistent with zero;
  - invalid harmonics and quantities are refused.
Run: python3 tests/test_toy_mc_mode_coupling.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data, evaluate,
)

CLASSES = ('0-30%', '30-60%')
A = {'eta': {'min': 0.4, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0}}
B = {'eta': {'min': -1.0, 'max': -0.4}, 'pT': {'min': 0.2, 'max': 3.0}}
CHI = {4: 1.2, 5: 1.6}
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
        cfg = ToyGenConfig(n_events=2500, mult_mean=1200, mult_sigma=300, seed=4242, rap_max=1.6,
                           flow=flow, fluctuate={2: 0.04, 3: 0.02, 4: 0.008, 5: 0.004},
                           nonlinear={4: (CHI[4], 2, 2), 5: (CHI[5], 2, 3)} if coupled else None)
        events, truth = generate_events(cfg)
        _CACHE[key] = {'2d': make_qvector_data(events, truth, n_pt=30, n_harm=9),
                       '3d': make_qgrid_data(events, truth, n_eta=32, n_pt=30, n_harm=9, eta_half=1.6),
                       'events': events, 'truth': truth}
    return _CACHE[key]


def rec(quantity, harmonics=(4, 2, 2), sub=True, **extra):
    r = {'observable_type': 'flow_harmonic_correlations', 'correlation_type': 'mode_coupling',
         'harmonics': list(harmonics), 'quantity': quantity, 'value_type': quantity,
         'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0}},
         'centrality': {'classes': list(CLASSES), 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}}
    if sub:
        r.update(particles='AB', subevents={'A': {'kinematic_range': A}, 'B': {'kinematic_range': B}})
    r.update(extra)
    return r


def run(r, kind, coupled=True, jk=False):
    c = _fx(coupled)[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def test_dispatch():
    check('dispatches to mode_coupling', resolve(rec('chi'))['compute'] == 'mode_coupling')
    for bad in (dict(harmonics=[4, 2, 3]), dict(quantity='kappa'), dict(harmonics=[4, 2])):
        r = rec('chi')
        r.update(bad)
        try:
            evaluate(r, grid=_fx()['3d'], cent_method='simple', with_jackknife=False)
            ok = False
        except ValueError:
            ok = True
        check(f'refused {bad}', ok)


def _brute_C(events, k, n, m, eta_a, eta_b, pt, max_ev=12, max_m=60):
    """<<cos(k phi1 - n phi2 - m phi3)>>, 1 in A and 2 != 3 in B, both
    orientations pooled, events weighted by their triplet counts (eta windows
    on particle eta; the first max_m particles of each event)."""
    num = den = 0.0
    for ev in events[:max_ev]:
        sel = (ev['pt'] >= pt[0]) & (ev['pt'] < pt[1])
        phi, eta = ev['phi'][sel][:max_m], ev['eta'][sel][:max_m]
        for (xa, xb) in ((eta_a, eta_b), (eta_b, eta_a)):
            a = phi[(eta >= xa[0]) & (eta < xa[1])]
            b = phi[(eta >= xb[0]) & (eta < xb[1])]
            for p1 in a:
                d = np.cos(k * p1 - n * b[:, None] - m * b[None, :])
                np.fill_diagonal(d, 0.0)
                num += d.sum()
                den += len(b) * (len(b) - 1)
    return num / den


def test_exact_triplets():
    # a small sample whose particles fill a 3D grid with windows on cell edges
    cfg = ToyGenConfig(n_events=12, mult_mean=60, mult_sigma=5, seed=9, rap_max=1.6,
                       flow={2: dict(kind='const', value=0.1), 3: dict(kind='const', value=0.05),
                             4: dict(kind='const', value=0.02)}, nonlinear={4: (1.0, 2, 2)})
    events, truth = generate_events(cfg)
    grid = make_qgrid_data(events, truth, n_eta=32, n_pt=30, n_harm=9, eta_half=1.6)
    edges = np.linspace(-1.6, 1.6, 33)
    a, b = (edges[20], edges[26]), (edges[6], edges[12])          # cell edges
    r = rec('C', sub=True, subevents={'A': {'kinematic_range': {'eta': {'min': a[0], 'max': a[1]},
                                                                'pT': {'min': 0.0, 'max': 3.0}}},
                                      'B': {'kinematic_range': {'eta': {'min': b[0], 'max': b[1]},
                                                                'pT': {'min': 0.0, 'max': 3.0}}}},
            kinematic_range={'eta': {'min': -1.6, 'max': 1.6}, 'pT': {'min': 0.0, 'max': 3.0}},
            centrality={'classes': ['0-100%'], 'eta': [{'min': -3.9, 'max': -3.0}]})
    v, _ = evaluate(r, grid=grid, cent_method='simple', with_jackknife=False)
    want = _brute_C(events, 4, 2, 2, a, b, (0.0, 3.0), max_m=10 ** 6)
    check('3D two-subevent C_{4,22} = explicit triplet sum', np.isclose(v[0], want, rtol=1e-9, atol=1e-12),
          f'{v[0]:.6e} vs {want:.6e}')


def test_chi_recovered():
    for kind in ('3d', '2d'):
        for k, h in ((4, (4, 2, 2)), (5, (5, 2, 3))):
            for sub in (True, False):
                v, e = run(rec('chi', h, sub=sub), kind, jk=True)
                ok = np.all(np.abs(v - CHI[k]) < 0.15 * CHI[k] + 3 * e)
                check(f"{kind} {'two-subevent' if sub else 'one-window'} chi_{k},{h[1]}{h[2]} = {CHI[k]}",
                      ok, f'{v} +- {e}')


def test_definitions():
    for kind in ('3d', '2d'):
        for h in ((4, 2, 2), (5, 2, 3)):
            vals = {q: run(rec(q, h), kind)[0] for q in ('C', 'v_inclusive', 'v_mode_coupled', 'v_linear', 'chi',
                                                         'rho')}
            p4 = vals['C'] / vals['chi']
            ok = (np.allclose(vals['v_mode_coupled'], vals['C'] / np.sqrt(p4), rtol=1e-10)
                  and np.allclose(vals['rho'], vals['v_mode_coupled'] / vals['v_inclusive'], rtol=1e-10)
                  and np.allclose(vals['v_linear'] ** 2, vals['v_inclusive'] ** 2 - vals['v_mode_coupled'] ** 2,
                                  rtol=1e-10))
            check(f'{kind} {h}: v_mc = C/sqrt(P4), rho = v_mc/v_inc, v_L^2 = v_inc^2 - v_mc^2', ok, str(vals))
            check(f'{kind} {h}: 0 < rho < 1 and v_linear < v_inclusive',
                  np.all((vals['rho'] > 0) & (vals['rho'] < 1)) and np.all(vals['v_linear'] < vals['v_inclusive']),
                  f"rho {vals['rho']}")


def test_differential():
    bins = [[0.2, 0.6], [0.6, 1.0], [1.0, 1.6], [1.6, 3.0]]
    kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0, 'bin_ranges': bins}}
    for kind in ('3d', '2d'):
        v, e = run(rec('chi', kinematic_range=kr, dimensionality='differential_pT'), kind, jk=True)
        check(f'{kind}: chi_4,22(pT) flat at {CHI[4]}', np.all(np.abs(v - CHI[4]) < 0.2 * CHI[4] + 3 * e),
              f'{v} +- {e}')
        vi, ei = run(rec('v_inclusive', kinematic_range=kr, dimensionality='differential_pT'), kind, jk=True)
        v2, _ = run(rec('v_inclusive'), kind)
        dev = np.abs(vi.reshape(2, -1) - v2[:, None])
        check(f'{kind}: v4(pT) of a flat v4 consistent with the integrated v4{{2}}',
              np.all(dev < 0.25 * v2[:, None] + 3 * ei.reshape(2, -1)), f'{vi} +- {ei} vs {v2}')


def test_no_coupling():
    for kind in ('3d', '2d'):
        v, e = run(rec('chi'), kind, coupled=False, jk=True)
        check(f'{kind}: no coupling, chi_4,22 consistent with 0', np.all(np.abs(v) < 4 * e + 0.05), f'{v} +- {e}')


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
