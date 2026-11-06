"""Flow relative to a lower-order plane (v_2{Psi_1}: flow_reference.harmonic 1) through the production path.

Toy events with a rapidity-odd v_1 (eta_shape) and Psi_2 = Psi_1 + delta,
delta ~ N(0, sigma) (ToyGenConfig.plane_lock), on 3D species grids:
  - v_2{Psi_1} of pions in -0.5 < y < 0 per pT bin, the plane from a forward
    subevent A, the resolution from A, B (forward) and C (backward),
    recovers v_2 <cos 2(Psi_2 - Psi_1)> = v_2 exp(-2 sigma^2), and v_2 when
    the planes coincide;
  - with independent planes it is consistent with zero;
  - a y-differential record agrees with the pT-integrated value per bin;
  - 2D containers, a POI overlapping subevent A, and m not dividing n are
    refused.
Run: python3 tests/test_toy_mc_mixed_plane.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import resolve  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, ToyGenConfig, evaluate, generate_events  # noqa: E402

V2 = 0.06
SIG = 0.35
BINS = [[0.2, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 2.0]]
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx(lock):
    if lock not in _CACHE:
        cfg = ToyGenConfig(n_events=3000, mult_mean=3000, mult_sigma=100, rap_max=4.5, seed=2504,
                           species_mix={'pion': 0.6, 'kaon': 0.15, 'proton': 0.25}, pt_min=0.05, pt_max=3.0,
                           flow={1: dict(kind='const', value=0.15), 2: dict(kind='const', value=V2)},
                           eta_shape={1: lambda eta: np.tanh(eta / 1.5)},
                           plane_lock={2: (1, lock)} if lock is not None else None)
        events, _ = generate_events(cfg)
        eta_edges = np.round(np.linspace(-5, 5, 101), 10)
        pt_edges = np.round(np.linspace(0.0, 3.0, 31), 10)
        mult = np.array([len(e['pt']) for e in events], float)

        def grid(slot=None):
            evs = events if slot is None else [{k: v[e['species_idx'] == slot] for k, v in e.items()} for e in events]
            return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=5, masses=SPECIES_MASSES, cent_mult=mult)
        _CACHE[lock] = MultiSpeciesGrid({211: grid(0), 321: grid(1), 2212: grid(2)}, combined=grid())
    return _CACHE[lock]


def rec(dim='differential_pT', **extra):
    r = {'observable_type': 'flow_vn', 'method': 'event_plane', 'harmonic': 2, 'value_type': 'v2',
         'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'pi+'},
         'kinematic_range': {'y': {'min': -0.5, 'max': 0.0},
                             'pT': {'min': 0.2, 'max': 2.0, 'bin_ranges': BINS}},
         'dimensionality': dim,
         'flow_reference': {'harmonic': 1, 'eta': [{'min': 2.0, 'max': 3.5}], 'pT': {'min': 0.0, 'max': 100.0},
                            'resolution_subevents': {'B': {'eta': [{'min': 1.0, 'max': 1.8}]},
                                                     'C': {'eta': [{'min': -1.2, 'max': -0.3}],
                                                           'pT': {'min': 0.2, 'max': 2.0}}}},
         'centrality': {'classes': ['0-50%', '50-100%'], 'eta': [{'min': -1.0, 'max': 1.0}]}}
    if dim == 'differential_y':
        r['kinematic_range'] = {'y': {'min': -1.0, 'max': 0.0, 'bin_ranges': [[-1.0, -0.5], [-0.5, 0.0]]},
                                'pT': {'min': 0.2, 'max': 2.0}}
    r.update(extra)
    return r


def run(r, lock, jk=True):
    return evaluate(r, grid=_fx(lock), cent_method='simple', with_jackknife=jk)


def test_dispatch_and_refusals():
    check('dispatches to vn_mixed_plane', resolve(rec())['compute'] == 'vn_mixed_plane')
    bad = rec()
    bad['flow_reference'] = dict(bad['flow_reference'], eta=[{'min': -0.6, 'max': 0.2}])
    for name, r, kw in (('POI overlapping subevent A', bad, {}),
                        ('m not dividing n', rec(harmonic=3, flow_reference=dict(rec()['flow_reference'], harmonic=2)),
                         {})):
        try:
            run(r, 0.0, jk=False)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    from analysis.pipeline_dispatch import get_pipeline_entry
    try:
        get_pipeline_entry('vn_mixed_plane')['prepare_data_2d'](None, {})
        ok = False
    except ValueError:
        ok = True
    check('refused: 2D container', ok)


def test_truth():
    for lock, want in ((0.0, V2), (SIG, V2 * np.exp(-2 * SIG ** 2))):
        v, e = run(rec(), lock)
        ok = np.all(np.abs(v - want) < 3 * e + 0.05 * want)
        check(f'v2{{Psi_1}}(pT), sigma = {lock}: = v2 <cos 2(Psi_2 - Psi_1)> = {want:.4f}', ok,
              f'{v.round(4)} +- {e.round(4)}')


def test_independent_planes():
    v, e = run(rec(), None)
    check('independent planes: v2{Psi_1} consistent with 0', np.all(np.abs(v) < 4 * e + 0.003),
          f'{v.round(4)} +- {e.round(4)}')


def test_y_bins():
    v, e = run(rec('differential_y'), SIG)
    vi, ei = run(rec('integrated', kinematic_range={'y': {'min': -0.5, 'max': 0.0}, 'pT': {'min': 0.2, 'max': 2.0}}),
                 SIG)
    vb = v.reshape(2, -1)[:, 1]
    check('y bin -0.5 < y < 0 = the integrated record', np.allclose(vb, vi, rtol=1e-10), f'{vb} vs {vi}')
    want = V2 * np.exp(-2 * SIG ** 2)
    check('v2{Psi_1}(y) flat at the truth', np.all(np.abs(v - want) < 3 * e + 0.05 * want), f'{v.round(4)}')


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
