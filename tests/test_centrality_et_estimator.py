"""Centrality estimators that sum transverse mass (centrality.estimator.sum:
transverse_mass; ALICE's TPC E_T, sum sqrt(pT^2 + m_pi^2) over the charged
tracks; analysis.pipeline_dispatch.estimator_transverse_mass /
estimator_variable).

Toy: events of Poisson multiplicity, each with its own spectrum slope (an
event-by-event temperature independent of the multiplicity), particles
uniform in |eta| < 1.

Checks:
  - 3D: sum m_T and sum m_T^2 in a window that cuts cells (|eta| < 0.75,
    0.15 < pT < 50 on 0.1-wide cells) equal the particle-level sums to the
    cell approximation (0.3%);
  - 2D: E_T over the charged multiplicity in the same window is the
    particles' mean m_T (1%);
  - the variable ranks events as E_T does and is in units of m~ = sum m_T^2
    / sum m_T;
  - run through the production path (toy harness): classes of E_T select
    events with a higher <pT> than classes of N_ch in the same acceptance
    (ALICE's energy bias), and E_T classes of a record without the field
    stay N_ch classes;
  - refusals: an unknown sum, transverse mass with classes in track count.
Run: python3 tests/test_centrality_et_estimator.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data, particles_to_qvector_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.pipeline_dispatch import (PION_MASS, estimator_multiplicity_3d,  # noqa: E402
                                        estimator_transverse_mass, estimator_variable)
from analysis.toy_mc import SPECIES_MASSES, SPECIES_NAMES  # noqa: E402

NE = 3000
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fixture():
    if 'fx' in _CACHE:
        return _CACHE['fx']
    rng = np.random.default_rng(4)
    events, T = [], []
    for _ in range(NE):
        m = rng.poisson(400)
        t = rng.normal(0.45, 0.05)
        pt = rng.gamma(2.0, t / 2.0 * 1.2, m) + 0.02
        events.append({'phi': rng.uniform(0, 2 * np.pi, m), 'eta': rng.uniform(-1, 1, m), 'pt': pt,
                       'species_idx': np.zeros(m, int), 'mass': np.full(m, SPECIES_MASSES[0])})
        T.append(t)
    eta_edges = np.round(np.linspace(-2, 2, 41), 10)
    pt_edges = np.round(np.linspace(0.0, 6.0, 61), 10)
    mult = np.array([len(e['phi']) for e in events], float)
    grid = particles_to_qgrid_data(events, eta_edges, pt_edges, n_harm=3, cent_mult=mult, masses=SPECIES_MASSES)
    qv = particles_to_qvector_data(events, pt_edges, rap_max=1.0, species_names=list(SPECIES_NAMES),
                                   masses=SPECIES_MASSES, n_harm=3)
    _CACHE['fx'] = dict(events=events, grid=grid, qv=qv, T=np.array(T))
    return _CACHE['fx']


def _particle_sums(fx, eta, pt):
    t1, t2 = [], []
    for e in fx['events']:
        sel = (np.abs(e['eta']) < eta) & (e['pt'] >= pt)
        mt = np.sqrt(e['pt'][sel] ** 2 + PION_MASS ** 2)
        t1.append(mt.sum())
        t2.append((mt ** 2).sum())
    return np.array(t1), np.array(t2)


def test_sums():
    print('test_sums')
    fx = _fixture()
    cf = 2.0       # the toy grid holds positives only: charged = x2
    # a window on cell edges: per event, up to m_T at a cell's mean pT^2
    s1, s2 = estimator_transverse_mass(fx['grid'], [(-0.8, 0.8)], (0.2, 50.0), is_2d=False)
    t1, t2 = _particle_sums(fx, 0.8, 0.2)
    r1, r2 = s1 / (cf * t1), s2 / (cf * t2)
    for name, r in (('sum m_T', r1), ('sum m_T^2', r2)):
        check(f'3D {name} per event = particles (cells on the window edges): mean, rms',
              abs(r.mean() - 1) < 2e-3 and r.std() < 6e-3, f'mean {r.mean():.5f}, rms {r.std():.1e}')
    # a window that cuts cells: their overlap share, exact on the sample
    s1, _ = estimator_transverse_mass(fx['grid'], [(-0.75, 0.75)], (0.15, 50.0), is_2d=False)
    t1, _ = _particle_sums(fx, 0.75, 0.15)
    r = np.sum(s1) / (cf * np.sum(t1))
    check('3D sum m_T in a window cutting cells, over the sample', abs(r - 1) < 3e-3, f'{r:.5f}')
    # 2D: a window over the whole plateau (|y| < 1; the toy is flat in eta,
    # which the boost-invariant container maps through dy/deta) and pT on bin
    # edges (the multiplicity helper takes whole bins)
    q1, _ = estimator_transverse_mass(fx['qv'], [(-5.0, 5.0)], (0.2, 50.0), is_2d=True)
    from scripts.compute_flow_vn import compute_charged_mult
    n2 = compute_charged_mult(fx['qv'], -5.0, 5.0, pt_min=0.2)
    mt_all = np.concatenate([np.sqrt(e['pt'][e['pt'] >= 0.2] ** 2 + PION_MASS ** 2) for e in fx['events']])
    ratio = np.sum(q1) / np.sum(n2) / mt_all.mean()
    check('2D E_T / N = <m_T>', abs(ratio - 1) < 0.01, f'{ratio:.4f}')


def test_variable():
    print('test_variable')
    fx = _fixture()
    params = {'cent_estimator': {'sum': 'transverse_mass'}, 'cent_eta_ranges': [(-0.8, 0.8)],
              'cent_pT_range': (0.15, 50.0)}
    x = estimator_variable(fx['grid'], params, False)
    s1, s2 = estimator_transverse_mass(fx['grid'], [(-0.8, 0.8)], (0.15, 50.0))
    check('ranks as E_T', np.array_equal(np.argsort(x), np.argsort(s1)))
    check('units of sum m_T^2 / sum m_T', abs(np.sum(x) - np.sum(s1) ** 2 / np.sum(s2)) < 1e-9 * np.sum(x))
    check('a counting estimator gives None', estimator_variable(fx['grid'], {'cent_estimator': {'sum': 'count'}},
                                                                 False) is None)


def _record(sum_=None):
    est = {'name': 'TPC', 'observable': 'ET' if sum_ else 'Nch'}
    if sum_:
        est['sum'] = sum_
    return dict(observable_type='mean_pT', inspire_id='0', version=1, hepdata_file='t.yaml', value_type='mean_pT',
                collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                centrality={'classes': ['0-5%', '5-50%', '50-100%'], 'estimator': est,
                            'eta': [{'min': -0.8, 'max': 0.8}], 'pT': {'min': 0.15, 'max': 50.0}},
                particle_id={'species': 'charged'},
                kinematic_range={'eta': {'min': -0.8, 'max': 0.8}, 'pT': {'min': 0.0, 'max': 10.0}},
                dimensionality='integrated')


def test_energy_bias():
    print('test_energy_bias')
    from analysis.toy_mc import evaluate
    fx = _fixture()
    for dim, kw in (('3D', {'grid': fx['grid']}), ('2D', {'qv': fx['qv']})):
        v_et, e_et = evaluate(_record('transverse_mass'), with_jackknife=True, n_blocks=10, **kw)
        v_n, e_n = evaluate(_record(), with_jackknife=True, n_blocks=10, **kw)
        d = v_et[0] - v_n[0]
        check(f'{dim} 0-5%: <pT> of E_T classes above N_ch classes', d > 3 * np.hypot(e_et[0], e_n[0]),
              f'{v_et[0]:.4f} vs {v_n[0]:.4f} (+- {np.hypot(e_et[0], e_n[0]):.4f})')
    rec = _record()
    rec['centrality']['estimator']['observable'] = 'ET'           # a calorimeter E_T label: still the count
    p = extract_measurement_params(rec)
    x = estimator_variable(fx['grid'], p, False)
    check('observable ET without sum: the charged multiplicity', x is None)
    m = estimator_multiplicity_3d(fx['grid'], p['cent_eta_ranges'], p['cent_pT_range'])
    check('  (estimator_multiplicity_3d serves it)', m.shape == (NE,))


def test_refusals():
    print('test_refusals')
    bad = _record('energy')
    try:
        resolve(bad)
        ok = False
    except ValueError:
        ok = True
    check('refused: sum energy', ok)
    mb = _record('transverse_mass')
    mb['centrality'].update(classes=['0-100', '100-200'], multiplicity_bins={'efficiency': 1.0})
    try:
        resolve(mb)
        ok = False
    except ValueError:
        ok = True
    check('refused: transverse mass with classes in track count', ok)


if __name__ == '__main__':
    tests = [test_sums, test_variable, test_energy_bias, test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
