"""Combinations of cumulant orders (cumulant_combination) through the production path.

A record declaring cumulant_combination (ratio v{a}/v{b}, difference,
scaled difference, the skewness estimate gamma1) is computed from its terms,
each an ordinary v_n{2k} record, on the same events inside the jackknife.
Checks, on 2D and 3D containers with fluctuating v2:
  - every combination equals the formula applied to the separately
    computed v2{2}, v2{4}, v2{6}, v2{8} (machine precision);
  - gamma1 with a two-subevent v2{2} term (ALICE's v2{2, |Delta eta| > 1});
  - mean flow <v2> = sqrt((v{2}^2 + v{4}^2)/2) and fluctuation sigma =
    sqrt((v{2}^2 - v{4}^2)/2) with the gapped v{2} (ALICE 2825785), and the
    square of one order (STAR's <v2^2>);
  - a ratio of harmonics, v3{2}/v2{2} with the gap (harmonics [3, 2]);
  - a v{2} term with its own gap and reference (CMS v2{4}/v2{2, 2PC});
  - the jackknife error of v2{6}/v2{4} is far below the uncorrelated
    propagation of the two errors (the orders share the events);
  - invalid combinations are refused.
Run: python3 tests/test_toy_mc_cumulant_combination.py
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
SUB = {'particles': 'AB', 'eta_gap': 1.0,
       'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}, 'pT': {'min': 0.2, 'max': 3.0}}},
                     'B': {'kinematic_range': {'eta': {'min': 0.5, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 3.0}}}}}
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=1500, mult_mean=900, mult_sigma=250, seed=777,
                           flow={2: dict(kind='const', value=0.08)}, fluctuate={2: 0.03})
        events, truth = generate_events(cfg)
        _CACHE['fx'] = {'2d': make_qvector_data(events, truth, n_pt=30, n_harm=9),
                        '3d': make_qgrid_data(events, truth, n_eta=40, n_pt=30, n_harm=9, eta_half=4.0)}
    return _CACHE['fx']


def rec(order=None, comb=None, **extra):
    r = {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': order or 8, 'harmonic': 2,
         'dimensionality': 'integrated', 'value_type': 'v2', 'hepdata_file': 'toy.yaml', 'inspire_id': '0',
         'version': 1, 'particle_id': {'species': 'charged'},
         'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 3.0}},
         'centrality': {'classes': list(CLASSES), 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}}
    if comb:
        r['cumulant_combination'] = comb
    r.update(extra)
    return r


def run(r, kind, jk=False):
    c = _fx()[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def test_dispatch():
    check('dispatches to vn_cumulant_combination',
          resolve(rec(comb={'kind': 'ratio', 'orders': [6, 4]}))['compute'] == 'vn_cumulant_combination')
    for bad in ({'kind': 'ratio', 'orders': [6]}, {'kind': 'gamma1', 'orders': [2, 4, 8]},
                {'kind': 'product', 'orders': [4, 2]}, {'kind': 'ratio', 'orders': [5, 4]},
                {'kind': 'mean_flow', 'orders': [2, 6]}, {'kind': 'flow_fluctuation', 'orders': [4, 2]},
                {'kind': 'square', 'orders': [2, 4]}):
        try:
            resolve(rec(comb=bad))
            ok = False
        except ValueError:
            ok = True
        check(f'refused {bad}', ok)


def test_combinations_equal_formula():
    for kind in ('2d', '3d'):
        v = {k: run(rec(k), kind)[0] for k in (2, 4, 6, 8)}
        g = SUB.copy()
        v2gap = run(rec(2, **g), kind)[0]
        expect = {
            'v6/v4': ({'kind': 'ratio', 'orders': [6, 4]}, v[6] / v[4]),
            'v8/v6': ({'kind': 'ratio', 'orders': [8, 6]}, v[8] / v[6]),
            'v6-v8': ({'kind': 'difference', 'orders': [6, 8]}, v[6] - v[8]),
            '(v4-v6)/11': ({'kind': 'difference', 'orders': [4, 6], 'scale': 1 / 11}, (v[4] - v[6]) / 11),
            'gamma1': ({'kind': 'gamma1'},
                       -6 * np.sqrt(2) * v[4] ** 2 * (v[4] - v[6]) / (v[2] ** 2 - v[4] ** 2) ** 1.5),
            'gamma1 gap': ({'kind': 'gamma1', 'two_particle': SUB},
                           -6 * np.sqrt(2) * v[4] ** 2 * (v[4] - v[6]) / (v2gap ** 2 - v[4] ** 2) ** 1.5),
            '<v2>': ({'kind': 'mean_flow', 'orders': [2, 4], 'two_particle': SUB},
                     np.sqrt((v2gap ** 2 + v[4] ** 2) / 2)),
            'sigma_v2': ({'kind': 'flow_fluctuation', 'orders': [2, 4], 'two_particle': SUB},
                         np.sqrt((v2gap ** 2 - v[4] ** 2) / 2)),
            '<v2^2>': ({'kind': 'square', 'orders': [2], 'two_particle': SUB}, v2gap ** 2),
            'v2{4}^2': ({'kind': 'square', 'orders': [4]}, v[4] ** 2),
        }
        for name, (comb, want) in expect.items():
            got = run(rec(comb=comb), kind)[0]
            ok = np.all(np.isfinite(got)) and np.allclose(got, want, rtol=1e-10, atol=0)
            check(f'{kind} {name}', ok, f'{np.round(got, 5)} vs {np.round(want, 5)}')
        check(f'{kind} gamma1 uses the gap term', not np.allclose(v2gap, v[2]), f'{v2gap} vs {v[2]}')


def _fx_v3():
    if 'v3' not in _CACHE:
        cfg = ToyGenConfig(n_events=1200, mult_mean=900, mult_sigma=250, seed=778,
                           flow={2: dict(kind='const', value=0.08), 3: dict(kind='const', value=0.04)},
                           fluctuate={2: 0.03, 3: 0.02})
        events, truth = generate_events(cfg)
        _CACHE['v3'] = {'2d': make_qvector_data(events, truth, n_pt=30, n_harm=7),
                        '3d': make_qgrid_data(events, truth, n_eta=40, n_pt=30, n_harm=7, eta_half=4.0)}
    return _CACHE['v3']


def test_harmonic_ratio():
    """v3{2}/v2{2} (ALICE O-O): orders [2, 2], harmonics [3, 2], both terms with the gap."""
    comb = {'kind': 'ratio', 'orders': [2, 2], 'harmonics': [3, 2], 'two_particle': SUB}
    for kind in ('2d', '3d'):
        c = _fx_v3()[kind]

        def ev(r):
            return (evaluate(r, qv=c, cent_method='simple', with_jackknife=False) if kind == '2d' else
                    evaluate(r, grid=c, cent_method='simple', with_jackknife=False))[0]
        v3 = ev(dict(rec(2, **SUB), harmonic=3, value_type='v3'))
        v2 = ev(rec(2, **SUB))
        got = ev(dict(rec(2, comb=comb), harmonic=3, value_type='v3{2}/v2{2}'))
        check(f'{kind} v3{{2}}/v2{{2}} = ratio of the separately computed gap terms',
              np.all(np.isfinite(got)) and np.allclose(got, v3 / v2, rtol=1e-10, atol=0),
              f'{np.round(got, 4)} vs {np.round(v3 / v2, 4)}')
        check(f'{kind} v3{{2}}/v2{{2}} near the input 0.5', np.all(np.abs(got - 0.5) < 0.1), f'{np.round(got, 3)}')
    for bad in ({'kind': 'ratio', 'orders': [2, 2]}, {'kind': 'ratio', 'orders': [2, 2], 'harmonics': [3]},
                {'kind': 'gamma1', 'harmonics': [2, 2, 2]}):
        try:
            resolve(rec(comb=bad))
            ok = False
        except ValueError:
            ok = True
        check(f'refused {bad}', ok)


def test_two_particle_reference():
    # CMS 1716441: v2{4}/v2{2}, the v{2} term a pair-gap correlation with its
    # own associated-pT reference (1-3 GeV/c), which the v{4} term does not use
    ref = {'eta': [{'min': -0.8, 'max': 0.8}], 'pT': {'min': 1.0, 'max': 3.0}}
    comb = {'kind': 'ratio', 'orders': [4, 2], 'two_particle': {'eta_gap': 1.0, 'flow_reference': ref}}
    for kind in ('2d', '3d'):
        vr, _ = run(rec(comb=comb), kind)
        v4, _ = run(rec(4), kind)
        v2, _ = run(rec(2, eta_gap=1.0, flow_reference=ref), kind)
        v2_plain, _ = run(rec(2, eta_gap=1.0), kind)
        check(f'{kind}: v2{{4}}/v2{{2, ref}} = the separate terms', np.allclose(vr, v4 / v2, rtol=1e-10, atol=0),
              f'{vr} vs {v4 / v2}')
        check(f'{kind}: the reference changes the v{{2}} term', not np.allclose(v2, v2_plain, rtol=1e-6),
              f'{v2} vs {v2_plain}')


def test_jackknife_correlation():
    vr, er = run(rec(comb={'kind': 'ratio', 'orders': [6, 4]}), '3d', jk=True)
    v4, e4 = run(rec(4), '3d', jk=True)
    v6, e6 = run(rec(6), '3d', jk=True)
    naive = (v6 / v4) * np.sqrt((e4 / v4) ** 2 + (e6 / v6) ** 2)
    check('v6/v4 jackknife error below the uncorrelated propagation',
          np.all(np.isfinite(er)) and np.all(er < naive) and er.min() < 0.5 * naive[er.argmin()], f'{er} vs {naive}')


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
