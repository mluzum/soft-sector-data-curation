"""Two-subevent v_n{2} on a 3D grid whose cells are wider than the subevents.

ALICE's v_n{2, |Delta eta| > 1.4} in |eta| < 0.8 correlates -0.8 < eta < -0.7
with 0.7 < eta < 0.8: half a cell each of a 0.2-wide eta grid (JETSCAPE O+O).
prepare_vn2_gap_3d takes cells cut by a window with their overlap fraction
(each subevent sum enters linearly), and subtracts the expected self-pairs
sum f_A f_B Q0 of cells inside both windows.  Checks (toy events with v2,
v3, flat in eta):
  - the ALICE windows recover the input v2, v3 within errors on 3D (whole
    cells by edge snapping gave 0: one window took a whole cell, the other
    none) and on 2D;
  - windows on cell edges give exactly the whole-cell sums;
  - overlapping windows (cells shared by both) are unbiased: the self-pair
    term is subtracted;
  - an empty subevent gives NaN, not 0.
Run: python3 tests/test_toy_mc_gap_subcell.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import extract_measurement_params  # noqa: E402
from analysis.pipeline_dispatch import PIPELINE_DISPATCH  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, make_qvector_data, evaluate  # noqa: E402

V2, V3 = 0.06, 0.03
results = []
_CACHE = {}


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def fx():
    if not _CACHE:
        cfg = ToyGenConfig(n_events=800, mult_mean=3000, mult_sigma=300, seed=31, rap_max=2.0, pt_min=0.2, pt_max=3.0,
                           flow={2: dict(kind='const', value=V2), 3: dict(kind='const', value=V3)})
        ev, tr = generate_events(cfg)
        _CACHE['3d'] = make_qgrid_data(ev, tr, n_eta=20, n_pt=14, n_harm=4, eta_half=2.0)    # 0.2-wide cells
        _CACHE['2d'] = make_qvector_data(ev, tr, n_pt=14, n_harm=4)
    return _CACHE


def rec(a, b):
    sub = lambda w: {'kinematic_range': {'eta': {'min': w[0], 'max': w[1]}, 'pT': {'min': 0.2, 'max': 3.0}}}  # noqa: E731
    return {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': 2, 'harmonic': 2,
            'value_type': 'v2', 'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1,
            'particle_id': {'species': 'charged'}, 'dimensionality': 'integrated',
            'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.2, 'max': 3.0}},
            'centrality': {'classes': ['0-100%'], 'eta': [{'min': -2.0, 'max': -1.2}, {'min': 1.2, 'max': 2.0}]},
            'particles': 'AB', 'eta_gap': 0.0, 'subevents': {'A': sub(a), 'B': sub(b)}}


def run(r, kind, harmonic=2, jk=True):
    r = dict(r, harmonic=harmonic, value_type=f'v{harmonic}')
    c = fx()[kind]
    v, e = (evaluate(r, grid=c, cent_method='simple', with_jackknife=jk) if kind == '3d' else
            evaluate(r, qv=c, cent_method='simple', with_jackknife=jk))
    return float(np.atleast_1d(v)[0]), float(np.atleast_1d(e)[0]) if jk else np.nan


def test_alice_windows():
    r = rec((-0.8, -0.7), (0.7, 0.8))
    for kind in ('3d', '2d'):
        for n, want in ((2, V2), (3, V3)):
            v, e = run(r, kind, n)
            check(f'{kind} v{n}{{2, |Delta eta| > 1.4}} (half-cell windows) = input {want} within errors',
                  np.isfinite(v) and abs(v - want) < 4 * e, f'{v:.4f} +- {e:.4f}')


def test_edge_aligned_unchanged():
    g = fx()['3d']
    r = rec((-0.8, -0.6), (0.6, 0.8))
    d = PIPELINE_DISPATCH['vn2_gap']['prepare_data_3d'](g.select({'species': 'charged'}), extract_measurement_params(r))
    u = g.select({'species': 'charged'})
    from analysis.pipeline_dispatch import _unit_weights
    u = _unit_weights(u)
    want_A = u.acceptance(-0.8, -0.6, 0.2, 3.0)
    want_B = u.acceptance(0.6, 0.8, 0.2, 3.0)
    check('windows on cell edges: whole-cell sums exactly, no self-pairs',
          np.allclose(d['Qn_A'], want_A, rtol=1e-12, atol=0) and np.allclose(d['Qn_B'], want_B, rtol=1e-12, atol=0)
          and not d['dQ_cross'].any())


def test_overlapping_windows():
    r = rec((-0.5, 0.1), (-0.1, 0.5))          # cells around eta = 0 lie in both windows
    d = PIPELINE_DISPATCH['vn2_gap']['prepare_data_3d'](fx()['3d'].select({'species': 'charged'}),
                                                        extract_measurement_params(r))
    v, e = run(r, '3d', 2)
    check(f'overlapping windows: self-pairs subtracted (dQ_cross > 0), v2 = input within errors',
          d['dQ_cross'].min() > 0 and abs(v - V2) < 4 * e, f'{v:.4f} +- {e:.4f}')


def test_empty_subevent():
    v, _ = run(rec((-0.8, -0.7), (5.0, 5.5)), '3d', 2, jk=False)
    check('an empty subevent: NaN', np.isnan(v), f'{v}')


if __name__ == '__main__':
    for t in (test_alice_windows, test_edge_aligned_unchanged, test_overlapping_windows, test_empty_subevent):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
