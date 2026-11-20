"""Higher moments of the event-by-event mean pT (analysis.pt_moments, key
pT_moments; ATLAS PRL 133 (2024) 252301, ALICE JHEP 11 (2025) 076).

Toy: each event's spectrum has its own scale T (pT = T x a Gamma(2)
variable), T drawn from a Gamma distribution (skewed), the multiplicity
Poisson; particles uniform in |eta| < 1.  The dynamical moments of [pT]
are those of <pT>_e = 2 T_e (the spectrum mean): c_n = 2^n times the n-th
central moment of T (c4: the fourth central moment), so k2, k3, gamma,
Gamma and kappa are known.

Checks:
  - exact against explicit sums over distinct particle tuples (3D; pT on
    the cell centres, so the cells' powers are exact): c2, c3, c4;
  - physics, 2D and 3D: k2, k3, gamma, Gamma, kappa of the toy within
    errors; ALICE's normalization (k_n over <[pT]^(n)>) differs from
    ATLAS's (over <[pT]>^n) by the expected <[pT]^(n)>/<[pT]>^n;
  - multiplicity_baseline: with <pT>_e rising with the multiplicity inside
    a class, M at the event's multiplicity removes that spread from c2;
  - refusals: an unknown moment, a differential record.
Run: python3 tests/test_toy_mc_pt_moments.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data, particles_to_qvector_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, SPECIES_NAMES  # noqa: E402

NE = 4000
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _events(seed=2, ne=NE, m_mean=300, trend=0.0, discrete=False):
    rng = np.random.default_rng(seed)
    events, T = [], []
    for _ in range(ne):
        m = int(rng.integers(m_mean // 2, 3 * m_mean // 2)) if trend else rng.poisson(m_mean)
        t = 0.25 * rng.gamma(9.0, 1.0 / 9.0) * (1.0 + trend * (m - m_mean) / m_mean)
        pt = t * rng.gamma(2.0, 1.0, m) + 0.0
        if discrete:
            pt = np.floor(pt / 0.1) * 0.1 + 0.05          # on the cell centres
        events.append({'phi': rng.uniform(0, 2 * np.pi, m), 'eta': rng.uniform(-1, 1, m), 'pt': pt,
                       'species_idx': np.zeros(m, int), 'mass': np.full(m, SPECIES_MASSES[0])})
        T.append(t)
    return events, np.array(T)


def _fixture(key='base', **kw):
    if key not in _CACHE:
        events, T = _events(**kw)
        eta_edges = np.round(np.linspace(-2, 2, 41), 10)
        pt_edges = np.round(np.linspace(0.0, 8.0, 81), 10)
        mult = np.array([len(e['pt']) for e in events], float)
        grid = particles_to_qgrid_data(events, eta_edges, pt_edges, n_harm=2, cent_mult=mult, masses=SPECIES_MASSES,
                                       with_sum_pt2=True)
        qv = particles_to_qvector_data(events, pt_edges, rap_max=1.0, species_names=list(SPECIES_NAMES),
                                       masses=SPECIES_MASSES, n_harm=2, with_sum_pt2=True)
        _CACHE[key] = dict(events=events, T=T, grid=grid, qv=qv)
    return _CACHE[key]


def record(q, norm=None, baseline=False, pt=(0.0, 8.0)):
    r = dict(observable_type='mean_pT_fluctuations', inspire_id='0', version=1, hepdata_file='t.yaml',
             value_type=q, pT_moment=q, dimensionality='integrated',
             collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             centrality={'classes': ['0-100%'], 'eta': [{'min': -1.0, 'max': 1.0}]},
             particle_id={'species': 'charged'},
             # the window holds every particle (2D: the whole plateau; the toy is flat in eta)
             kinematic_range={'eta': {'min': -5.0, 'max': 5.0}, 'pT': {'min': pt[0], 'max': pt[1]}})
    if norm:
        r['moment_normalization'] = norm
    if baseline:
        r['multiplicity_baseline'] = {'eta': [{'min': -5.0, 'max': 5.0}], 'pT': {'min': pt[0], 'max': pt[1]}}
    return r


_PREP = {}


def production(rec, dim, w=None, fx=None):
    import json
    fx = fx or _fixture()
    key = (json.dumps(rec, sort_keys=True), dim, id(fx))
    ent = get_pipeline_entry(resolve(rec)['compute'])
    if key not in _PREP:
        p = extract_measurement_params(rec)
        _PREP[key] = ent['prepare_data_2d'](fx['qv'], p) if dim == '2D' else ent['prepare_data_3d'](fx['grid'], p)
    n = len(fx['events'])
    return float(ent['core_fn'](_PREP[key], np.ones(n) if w is None else w)[0])


def jk_error(fn, n, n_blocks=10):
    vals = []
    for b in np.array_split(np.arange(n), n_blocks):
        w = np.ones(n)
        w[b] = 0
        vals.append(fn(w))
    vals = np.array(vals)
    return np.sqrt((n_blocks - 1) / n_blocks * np.sum((vals - vals.mean()) ** 2))


def test_exact():
    print('test_exact')
    fx = _fixture('disc', seed=3, ne=60, m_mean=8, discrete=True)
    ev = fx['events']
    M = np.sum([e['pt'].sum() for e in ev]) / np.sum([len(e['pt']) for e in ev])
    for q, n in (('c2', 2), ('c3', 3), ('c4', 4)):
        num = den = 0.0
        for e in ev:
            d = e['pt'] - M
            for idx in itertools.permutations(range(len(d)), n):
                num += np.prod(d[list(idx)])
                den += 1.0
        want = num / den
        got = production(record(q), '3D', fx=fx)
        check(f'3D {q} == explicit distinct tuples', abs(got - want) < 1e-9 * abs(want), f'{got:.6e} vs {want:.6e}')


def truth(T):
    m = 2.0 * T                      # <pT>_e
    mu = m.mean()
    c = {n: np.mean((m - mu) ** n) for n in (2, 3, 4)}
    return {'k2': c[2] / mu ** 2, 'k3': c[3] / mu ** 3, 'gamma': c[3] / c[2] ** 1.5, 'Gamma': c[3] * mu / c[2] ** 2,
            'kappa': c[4] / c[2] ** 2}


def test_physics():
    print('test_physics')
    fx = _fixture()
    t = truth(fx['T'])
    for dim in ('3D', '2D'):
        for q in ('k2', 'k3', 'gamma', 'Gamma', 'kappa'):
            rec = record(q)
            v = production(rec, dim)
            e = jk_error(lambda w, rec=rec, dim=dim: production(rec, dim, w), NE)
            check(f'{dim} {q} = {t[q]:.4g}', abs(v - t[q]) < 3.5 * e + 0.02 * abs(t[q]), f'{v:.4g} +- {e:.2g}')
    a, b = production(record('k2', 'moment'), '3D'), production(record('k2'), '3D')
    ev = fx['events']
    M = np.sum([e['pt'].sum() for e in ev]) / np.sum([len(e['pt']) for e in ev])
    c2 = b * M ** 2
    # <[pT]^(2)>: sum over distinct pairs of pT_i pT_j, pair-weighted over the events
    num = np.sum([e['pt'].sum() ** 2 - (e['pt'] ** 2).sum() for e in ev])
    den = np.sum([len(e['pt']) * (len(e['pt']) - 1) for e in ev])
    want = c2 / (num / den)
    check("ALICE's k2 = c2 / <[pT]^(2)> (pair-weighted)", abs(a - want) < 1e-3 * a, f'{a:.6e} vs {want:.6e}')


def test_baseline():
    print('test_baseline')
    fx = _fixture('trend', seed=7, ne=4000, m_mean=300, trend=0.6)
    T, mult = fx['T'], np.array([len(e['pt']) for e in fx['events']], float)
    # the events' dynamical <pT> and its mean at fixed multiplicity (exactly
    # linear in the toy: E[2T | N] = 2 x 0.25 (1 + 0.6 (N - 300)/300)); c2
    # weights the events by their pairs, the class mean by their particles
    m = 2.0 * T
    cond = 0.5 * (1.0 + 0.6 * (mult - 300) / 300)
    P = mult * (mult - 1)
    M = np.sum(mult * m) / np.sum(mult)
    fixed = np.sum(P * (m - cond) ** 2) / np.sum(P)
    full = np.sum(P * (m - M) ** 2) / np.sum(P)
    vb = production(record('c2', baseline=True), '3D', fx=fx)
    vc = production(record('c2'), '3D', fx=fx)
    eb = jk_error(lambda w: production(record('c2', baseline=True), '3D', w, fx=fx), 4000)
    ec = jk_error(lambda w: production(record('c2'), '3D', w, fx=fx), 4000)
    check(f'c2 at fixed multiplicity = {fixed:.4e}', abs(vb - fixed) < 3.5 * eb, f'{vb:.4e} +- {eb:.1e}')
    check(f'c2 of the class = {full:.4e}', abs(vc - full) < 3.5 * ec, f'{vc:.4e} +- {ec:.1e}')
    check('the two differ (> 5 sigma)', abs(full - fixed) > 5 * max(eb, ec))


def test_refusals():
    print('test_refusals')
    for name, rec in (('unknown moment', record('skew')), ('differential', dict(record('k2'), dimensionality='differential_pT')),
                      ('unknown normalization', record('k2', 'tuple'))):
        try:
            resolve(rec)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)


if __name__ == '__main__':
    tests = [test_exact, test_physics, test_baseline, test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
