"""Symmetric cumulants SC(m, n) and NSC(m, n) (analysis.correlators, sc_mn).

SC(m, n) = <<4>>_{m,n,-m,-n} - <<2>>_m <<2>>_n over distinct particles;
NSC divides by <<2>>_m <<2>>_n (with a normalization gap: from the two
halves of the window).

Toy: v2 and v3 fluctuate event by event and are correlated (v3 grows with
v2), so SC(3,2) > 0; truth SC = <v2^2 v3^2> - <v2^2><v3^2>.

Checks:
  - 3D: production equals the closed-form unit-weight Q-cumulant
    (Bilandzic et al., PRC 89, 064904, Eq. for <4>_{m,n,-m,-n}) on the same
    particles, to machine precision;
  - physics: SC(3,2) and NSC(3,2) recover the truth within errors, in 3D and
    on the 2D container (Jacobian weights), with and without the
    normalization gap; the old |Q_n|^2/M^2 estimator would be biased by the
    1/M self-correlations;
  - two subevents (particles AB; ALICE 2825785's |Delta eta| > 0.8): <<m,
    n | -m, -n>> with m, n in one subevent and -m, -n in the other, both
    orientations pooled: equal on 3D to the closed form from the subevents'
    Q-vectors, and SC, NSC recover the truth on 2D and 3D;
  - refusals: SC of three harmonics, a record without 'normalized', a
    harmonic the container does not store (SC(4,2) needs Q6), and subevents
    with a normalization gap.
Until November 2026 sc_mn used |Q_n|^2/M^2 (no self-correlation removal),
ignored the pT window and the weights in 2D, never normalized, and read
three-harmonic SC as SC of the first two.
Run: python3 tests/test_toy_mc_sc.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data, particles_to_qvector_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, SPECIES_NAMES  # noqa: E402

NE, M_MEAN = 3000, 500
ETA = (-0.8, 0.8)
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _events():
    if 'ev' in _CACHE:
        return _CACHE['ev']
    rng = np.random.default_rng(7)
    events, v2s, v3s = [], [], []
    for _ in range(NE):
        v2 = abs(rng.normal(0.07, 0.03))
        v3 = abs(0.02 + 0.3 * (v2 - 0.07) + rng.normal(0, 0.008))
        psi2, psi3 = rng.uniform(0, 2 * np.pi, 2)
        m = rng.poisson(M_MEAN)
        phi = []
        while len(phi) < m:          # accept-reject from 1 + 2 v2 cos 2(phi-psi2) + 2 v3 cos 3(phi-psi3)
            c = rng.uniform(0, 2 * np.pi, 2 * m)
            f = 1 + 2 * v2 * np.cos(2 * (c - psi2)) + 2 * v3 * np.cos(3 * (c - psi3))
            phi.extend(c[rng.uniform(0, 1 + 2 * (v2 + v3), c.size) < f].tolist())
        phi = np.array(phi[:m])
        eta = rng.uniform(-1, 1, m)
        pt = rng.exponential(0.5, m) + 0.2
        events.append({'phi': phi, 'eta': eta, 'pt': pt, 'species_idx': np.zeros(m, int),
                       'mass': np.full(m, SPECIES_MASSES[0])})
        v2s.append(v2)
        v3s.append(v3)
    v2s, v3s = np.array(v2s), np.array(v3s)
    truth_sc = np.mean(v2s ** 2 * v3s ** 2) - np.mean(v2s ** 2) * np.mean(v3s ** 2)
    eta_edges = np.round(np.linspace(-6, 6, 121), 10)
    pt_edges = np.round(np.linspace(0.0, 10.0, 101), 10)
    mult = np.array([len(e['phi']) for e in events], float)
    grid = particles_to_qgrid_data(events, eta_edges, pt_edges, n_harm=9, cent_mult=mult, masses=SPECIES_MASSES)
    qv = particles_to_qvector_data(events, pt_edges, rap_max=1.0, species_names=list(SPECIES_NAMES),
                                   masses=SPECIES_MASSES, n_harm=9)
    _CACHE['ev'] = dict(events=events, grid=grid, qv=qv, sc=truth_sc,
                        nsc=truth_sc / (np.mean(v2s ** 2) * np.mean(v3s ** 2)))
    return _CACHE['ev']


SUB_A, SUB_B = (-1.0, -0.2), (0.2, 1.0)    # the toy fills |eta| < 1


def record_sub(m=3, n=2, normalized=False):
    r = record(m, n, normalized)
    r.update(particles='AB', subevents={lab: {'kinematic_range': {'eta': {'min': w[0], 'max': w[1]},
                                                                  'pT': {'min': 0.2, 'max': 5.0}}}
                                        for lab, w in (('A', SUB_A), ('B', SUB_B))})
    return r


def closed_form_sub(events, m, n):
    """Two-subevent SC(m,n) from the subevents' Q-vectors (disjoint windows)."""
    acc = np.zeros(5)
    for e in events:
        q = {}
        for lab, w in (('A', SUB_A), ('B', SUB_B)):
            s = (e['eta'] >= w[0]) & (e['eta'] < w[1]) & (e['pt'] >= 0.2) & (e['pt'] < 5.0)
            phi = e['phi'][s]
            q[lab] = {h: np.exp(1j * h * phi).sum() for h in (m, n, m + n)}
            q[lab]['M'] = float(len(phi))
        a, b = q['A'], q['B']
        four = (a[m] * a[n] - a[m + n]) * np.conj(b[m] * b[n] - b[m + n])
        acc += [2 * four.real, 2 * a['M'] * (a['M'] - 1) * b['M'] * (b['M'] - 1),
                2 * np.real(a[m] * np.conj(b[m])), 2 * np.real(a[n] * np.conj(b[n])), 2 * a['M'] * b['M']]
    return acc[0] / acc[1] - (acc[2] / acc[4]) * (acc[3] / acc[4])


def record(m=3, n=2, normalized=False, gap=None):
    r = dict(observable_type='flow_harmonic_correlations', correlation_type='symmetric_cumulant',
             harmonics=[m, n], observable='SC', value_type='SC', hepdata_file='t.yaml', inspire_id='0', version=1,
             collision_system={'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
             centrality={'classes': ['0-100%'], 'eta': [{'min': 3.0, 'max': 5.0}]},
             kinematic_range={'eta': {'min': ETA[0], 'max': ETA[1]}, 'pT': {'min': 0.2, 'max': 5.0}},
             normalized=normalized)
    if gap is not None:
        r['normalization_eta_gap'] = gap
    return r


def production(rec, dim, w=None):
    fx = _events()
    p = extract_measurement_params(rec)
    ent = get_pipeline_entry(resolve(rec)['compute'])
    data = ent['prepare_data_2d'](fx['qv'], p) if dim == '2D' else ent['prepare_data_3d'](fx['grid'], p)
    return float(ent['core_fn'](data, np.ones(NE) if w is None else w)[0])


def closed_form(events, m, n):
    """Unit-weight SC(m,n) from Q-vectors of the particles in the window."""
    num4, den4, n2m, n2n, den2 = [], [], [], [], []
    for e in events:
        s = (e['eta'] >= ETA[0]) & (e['eta'] < ETA[1]) & (e['pt'] >= 0.2) & (e['pt'] < 5.0)
        phi = e['phi'][s]
        M = float(len(phi))
        Q = {h: np.exp(1j * h * phi).sum() for h in (m, n, m + n, m - n)}
        a = (abs(Q[m]) ** 2 * abs(Q[n]) ** 2 - 2 * np.real(Q[m + n] * np.conj(Q[m]) * np.conj(Q[n]))
             - 2 * np.real(Q[m] * np.conj(Q[m - n]) * np.conj(Q[n])) + abs(Q[m + n]) ** 2 + abs(Q[m - n]) ** 2
             - (M - 4) * (abs(Q[m]) ** 2 + abs(Q[n]) ** 2) + M * (M - 6))
        num4.append(a)
        den4.append(M * (M - 1) * (M - 2) * (M - 3))
        n2m.append(abs(Q[m]) ** 2 - M)
        n2n.append(abs(Q[n]) ** 2 - M)
        den2.append(M * (M - 1))
    four = np.sum(num4) / np.sum(den4)
    return four - np.sum(n2m) / np.sum(den2) * np.sum(n2n) / np.sum(den2)


def jk_error(fn, n_blocks=10):
    blocks = np.array_split(np.arange(NE), n_blocks)
    vals = []
    for b in blocks:
        w = np.ones(NE)
        w[b] = 0
        vals.append(fn(w))
    vals = np.array(vals)
    return np.sqrt((n_blocks - 1) / n_blocks * np.sum((vals - vals.mean()) ** 2))


def test_engine_matches_closed_form():
    print('test_engine_matches_closed_form')
    fx = _events()
    got, want = production(record(), '3D'), closed_form(fx['events'], 3, 2)
    check('3D SC(3,2) == closed-form Q-cumulant', abs(got - want) < 1e-9 * abs(want), f'{got:.6e} vs {want:.6e}')


def test_physics():
    print('test_physics')
    fx = _events()
    for dim in ('3D', '2D'):
        for norm, gap, truth in ((False, None, fx['sc']), (True, None, fx['nsc']), (True, 1.0, fx['nsc'])):
            rec = record(normalized=norm, gap=gap)
            v = production(rec, dim)
            e = jk_error(lambda w, rec=rec, dim=dim: production(rec, dim, w))
            name = ('NSC' if norm else 'SC') + ('(gap 1)' if gap else '')
            check(f'{dim} {name}(3,2) = {truth:.3e}', abs(v - truth) < 3 * e + 0.05 * abs(truth), f'{v:.3e} +- {e:.1e}')
    # what the old estimator gave: |Q|^2/M^2 keeps the 1/M self-correlations
    old = []
    for e in fx['events']:
        s = (np.abs(e['eta']) < 0.8) & (e['pt'] < 5.0)
        phi = e['phi'][s]
        M = len(phi)
        old.append((abs(np.exp(2j * phi).sum()) ** 2 / M ** 2, abs(np.exp(3j * phi).sum()) ** 2 / M ** 2))
    old = np.array(old)
    sc_old = np.mean(old[:, 0] * old[:, 1]) - old[:, 0].mean() * old[:, 1].mean()
    print(f'      old estimator: SC = {sc_old:.3e} (truth {fx["sc"]:.3e})')


def test_subevents():
    print('test_subevents')
    fx = _events()
    got, want = production(record_sub(), '3D'), closed_form_sub(fx['events'], 3, 2)
    check('3D two-subevent SC(3,2) == closed form from the subevent Q-vectors', abs(got - want) < 1e-9 * abs(want),
          f'{got:.6e} vs {want:.6e}')
    for dim in ('3D', '2D'):
        for norm, truth in ((False, fx['sc']), (True, fx['nsc'])):
            rec = record_sub(normalized=norm)
            v = production(rec, dim)
            e = jk_error(lambda w, rec=rec, dim=dim: production(rec, dim, w))
            check(f"{dim} two-subevent {'NSC' if norm else 'SC'}(3,2) = {truth:.3e}",
                  abs(v - truth) < 3 * e + 0.05 * abs(truth), f'{v:.3e} +- {e:.1e}')
    r = record_sub(normalized=True)
    r['normalization_eta_gap'] = 0.8
    try:
        production(r, '3D')
        check('subevents with a normalization gap refused', False)
    except ValueError:
        check('subevents with a normalization gap refused', True)


def test_refusals():
    print('test_refusals')
    r3 = record()
    r3['harmonics'] = [2, 3, 4]
    for name, rec in (('three harmonics', r3), ('no normalized field', {k: v for k, v in record().items()
                                                                        if k != 'normalized'})):
        try:
            resolve(rec)
            check(f'refused: {name}', False)
        except ValueError:
            check(f'refused: {name}', True)
    fx = _events()
    import copy
    small = copy.copy(fx['qv'])
    small.Qn = fx['qv'].Qn[:, :, :6, :]
    p = extract_measurement_params(record(4, 2))
    try:
        get_pipeline_entry('sc_mn')['prepare_data_2d'](small, p)
        check('SC(4,2) without Q6 refused', False)
    except ValueError:
        check('SC(4,2) without Q6 refused', True)


if __name__ == '__main__':
    tests = [test_engine_matches_closed_form, test_physics, test_subevents, test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            print(f'  PASS  {t.__name__}\n')
        except AssertionError as ex:
            print(f'  FAIL  {t.__name__}: {ex}\n')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
