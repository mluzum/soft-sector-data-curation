"""Flow from three sets of two-particle correlations (method three_subevent_2pc) through the production path.

  - pions with v_2(pT) rising linearly, references B (forward) and C
    (backward): v_2(pT) = sqrt(c_AB c_AC / c_BC) follows the generated
    profile times sqrt(<s^2>) (the event-by-event scale), 2D and 3D, in an
    eta window and in a rapidity window;
  - the frame mapping of an asymmetric system shifts and flips the
    reference regions with the record (analysis.frames);
  - a POI overlapping a reference region is refused (3D).
Run: python3 tests/test_toy_mc_3x2pc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import resolve  # noqa: E402
from analysis.frames import to_container_frame  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    SPECIES_MASSES, ToyGenConfig, _vn_profile_scale, evaluate, generate_events, make_qvector_data, vn_of_pt,
)

BINS = [[0.2, 0.6], [0.6, 1.0], [1.0, 1.6], [1.6, 2.4]]
SUBS = {'B': {'eta': [{'min': 2.0, 'max': 3.5}], 'pT': {'min': 0.0, 'max': 100.0}},
        'C': {'eta': [{'min': -3.5, 'max': -2.0}], 'pT': {'min': 0.0, 'max': 100.0}}}
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=2500, mult_mean=3000, mult_sigma=100, rap_max=4.5, seed=2411,
                           species_mix={'pion': 0.7, 'kaon': 0.15, 'proton': 0.15}, pt_min=0.05, pt_max=3.0,
                           flow={2: dict(kind='linear', value=0.05, pt0=1.0)}, fluctuate={2: 0.02})
        events, truth = generate_events(cfg)
        eta_edges = np.round(np.linspace(-5, 5, 101), 10)
        pt_edges = np.round(np.linspace(0.0, 3.0, 31), 10)
        mult = np.array([len(e['pt']) for e in events], float)

        def grid(slot=None):
            evs = events if slot is None else [{k: v[e['species_idx'] == slot] for k, v in e.items()} for e in events]
            return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=5, masses=SPECIES_MASSES, cent_mult=mult)
        s = truth['vn_ebe'][2] / _vn_profile_scale(cfg.flow[2])
        _CACHE['fx'] = {'3d': MultiSpeciesGrid({211: grid(0), 321: grid(1), 2212: grid(2)}, combined=grid()),
                        '2d': make_qvector_data(events, truth, n_pt=30, n_harm=5),
                        'rms': np.sqrt(np.mean(s ** 2))}
    return _CACHE['fx']


def rec(window='eta', **extra):
    kr = {'pT': {'min': 0.2, 'max': 2.4, 'bin_ranges': BINS}}
    kr['eta' if window == 'eta' else 'y'] = {'min': -0.8, 'max': 0.8} if window == 'eta' else {'min': -0.5, 'max': 0.5}
    if window == 'y':
        kr['eta'] = {'min': -0.8, 'max': 0.8}
    r = {'observable_type': 'flow_vn', 'method': 'three_subevent_2pc', 'harmonic': 2, 'value_type': 'v2',
         'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'pi+'},
         'kinematic_range': kr, 'dimensionality': 'differential_pT',
         'flow_reference': {'subevents': {k: dict(v) for k, v in SUBS.items()}},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -1.0, 'max': 1.0}]}}
    r.update(extra)
    return r


def run(r, kind, jk=True):
    c = _fx()[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def test_profile():
    want = vn_of_pt(np.array([0.5 * (a + b) for a, b in BINS]), 'linear', 0.05, 1.0) * _fx()['rms']
    check('dispatches to vn_3x2pc', resolve(rec())['compute'] == 'vn_3x2pc')
    for kind in ('3d', '2d'):
        for window in ('eta', 'y'):
            v, e = run(rec(window), kind)
            check(f'{kind} {window} window: v2(pT) = generated profile', np.all(np.abs(v - want) < 3 * e + 0.06 * want),
                  f'{v.round(4)} +- {e.round(4)} vs {want.round(4)}')


def test_frames():
    r = rec(collision_system={'energy_gev': 5020.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'lab',
                              'rapidity_shift': -0.465, 'forward_beam': 'Pb'})
    m, why = to_container_frame(r, {'frame': 'nn_cms', 'forward_beam': 'p'})
    b = m['flow_reference']['subevents']['B']['eta'][0]
    c = m['flow_reference']['subevents']['C']['eta'][0]
    ok = why is None and np.isclose(b['min'], -3.965) and np.isclose(b['max'], -2.465) and \
        np.isclose(c['min'], 1.535) and np.isclose(c['max'], 3.035)
    check('frame mapping shifts and flips the reference regions', ok, f'B {b}, C {c}')


def test_overlap_refused():
    r = rec()
    r['flow_reference']['subevents']['B'] = {'eta': [{'min': 0.4, 'max': 1.5}], 'pT': {'min': 0.0, 'max': 100.0}}
    try:
        run(r, '3d', jk=False)
        ok = False
    except ValueError:
        ok = True
    check('POI overlapping reference B refused (3D)', ok)


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
