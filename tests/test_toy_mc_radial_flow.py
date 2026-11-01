"""Radial-flow fluctuations: v0, v0(pT), normalized covariance (analysis.radial_flow).

Toy: every event's pT are rescaled by a common k_e ~ N(1, sigma) (a
collective radial-flow fluctuation, the same at all rapidities).  Two
subevents a: -2.5 < eta < -0.5 and b: 0.5 < eta < 2.5, spectrum
0.5 < pT < 2.5 GeV, reference [pT] in 0.5 < pT < 2 GeV.

Checks:
  - routing: radial_flow records resolve to the 'radial_flow' key, one
    value per pT bin (SINGLE_ROW_KEYS);
  - 3D: production equals the particle-level evaluation of the same
    estimator on the same particles (disjoint subevents: no self pairs),
    for every quantity;
  - 2D boost-invariant container: production equals the particle-level
    weighted estimator with the pairs of a particle with itself removed
    (every particle is in both windows with its Jacobian weight);
  - physics: v0 and v0(pT) agree with their expectation from the k_e
    distribution (computed from a large unscaled particle sample), in 3D and
    in 2D (where without the self-pair removal v0 comes out 40% high here,
    0.0198 for 0.0141: the pairs of a particle with itself double v0^2);
  - multiplicity_baseline: a [pT] trend with the event multiplicity inflates
    the class-mean v0; deviations at fixed multiplicity remove it.
Run: python3 tests/test_toy_mc_radial_flow.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data, particles_to_qvector_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.observables_2d import _make_jacobian_list  # noqa: E402
from analysis.pipeline_dispatch import SINGLE_ROW_KEYS, container_for_prepare, get_pipeline_entry  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, SPECIES_NAMES, ToyGenConfig, generate_events  # noqa: E402

NE = 1500
SIGMA = 0.04
A_ETA, B_ETA = (-2.5, -0.5), (0.5, 2.5)
SPEC, REF = (0.5, 2.5), (0.5, 2.0)
PT_BINS = [0.5, 0.7, 0.9, 1.2, 1.6, 2.5]
ETA_EDGES = np.round(np.linspace(-6, 6, 121), 10)       # 0.1 cells: window edges on cell edges
PT_EDGES = np.round(np.linspace(0.0, 4.0, 81), 10)      # 0.05 cells
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _events(sigma=SIGMA, seed=5, trend=0.0):
    key = (sigma, seed, trend)
    if key not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=1500, mult_sigma=150, rap_max=3.0, seed=seed,
                           species_mix={'pion': 0.8, 'kaon': 0.12, 'proton': 0.08}, pt_rescale_sigma=sigma)
        events, _ = generate_events(cfg)
        if trend:
            # a [pT] trend with multiplicity: pT x (1 + trend (M - <M>)/<M>)
            mult = np.array([len(e['pt']) for e in events], float)
            for e, m in zip(events, mult):
                e['pt'] = e['pt'] * (1 + trend * (m - mult.mean()) / mult.mean())
        mult = np.array([len(e['pt']) for e in events], float)
        grid = particles_to_qgrid_data(events, ETA_EDGES, PT_EDGES, n_harm=3, masses=SPECIES_MASSES,
                                       cent_mult=mult, with_sum_pt=True, with_sum_pt2=True)
        qv = particles_to_qvector_data(events, PT_EDGES, rap_max=cfg.rap_max, species_names=list(SPECIES_NAMES),
                                       masses=SPECIES_MASSES, n_harm=3, with_sum_pt=True, with_sum_pt2=True)
        _CACHE[key] = dict(events=events, grid=grid, qv=qv, cfg=cfg)
    return _CACHE[key]


def record(quantity, baseline=None):
    dim = 'integrated' if quantity == 'v0' else 'differential_pT'
    pt = {'min': SPEC[0], 'max': SPEC[1]}
    if dim != 'integrated':
        pt['bins'] = PT_BINS
    r = dict(observable_type='radial_flow', quantity=quantity, particles='AB',
             subevents={'A': {'kinematic_range': {'eta': {'min': A_ETA[0], 'max': A_ETA[1]}}},
                        'B': {'kinematic_range': {'eta': {'min': B_ETA[0], 'max': B_ETA[1]}}}},
             reference_pT={'min': REF[0], 'max': REF[1]},
             kinematic_range={'eta': {'min': -2.5, 'max': 2.5}, 'pT': pt},
             particle_id={'species': 'charged'}, dimensionality=dim, value_type='v0', hepdata_file='t.yaml',
             inspire_id='0', version=1,
             centrality={'classes': ['0-100%'], 'eta': [{'min': -0.5, 'max': 0.5}]})
    if baseline:
        r['multiplicity_baseline'] = baseline
    return r


def production(rec, dim, fx):
    params = extract_measurement_params(rec)
    ent = get_pipeline_entry(resolve(rec)['compute'])
    if dim == '2D':
        data = ent['prepare_data_2d'](fx['qv'], params)
    else:
        data = ent['prepare_data_3d'](container_for_prepare(fx['grid'], params, ent), params)
    return np.asarray(ent['core_fn'](data, np.ones(NE)))


def _per_event(fx, weights_of):
    """Per-event particle sums.  weights_of(e, X) -> per-particle weight of
    subevent X (0 outside)."""
    rows = []
    for e in fx['events']:
        pt = e['pt']
        r = {}
        for X in 'AB':
            w = weights_of(e, X)
            in_s, in_r = (pt >= SPEC[0]) & (pt < SPEC[1]), (pt >= REF[0]) & (pt < REF[1])
            r[X] = dict(w=w, N=(w * in_s).sum(), Nr=(w * in_r).sum(), P=(w * in_r * pt).sum(),
                        Nb=np.array([(w * ((pt >= lo) & (pt < hi))).sum()
                                     for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])]),
                        in_s=in_s, in_r=in_r)
        r['pt'] = pt
        rows.append(r)
    return rows


def particle_level(rows, quantity):
    """The estimator from the particles, pairs of a particle with itself
    removed explicitly (zero for disjoint subevents)."""
    pA = np.array([r['A']['P'] / r['A']['Nr'] for r in rows])
    pB = np.array([r['B']['P'] / r['B']['Nr'] for r in rows])
    mA, mB = pA.mean(), pB.mean()
    nA = np.array([r['A']['Nb'] / r['A']['N'] for r in rows])
    nB = np.array([r['B']['Nb'] / r['B']['N'] for r in rows])
    nbA, nbB = nA.mean(axis=0), nB.mean(axis=0)
    pp, cov = [], []
    for r, a, b, na, nb in zip(rows, pA, pB, nA, nB):
        wa, wb, pt = r['A']['w'], r['B']['w'], r['pt']
        da = wa * r['A']['in_r'] * (pt - mA) / r['A']['Nr']        # each particle's share of d[pT]_a
        db = wb * r['B']['in_r'] * (pt - mB) / r['B']['Nr']
        pp.append(da.sum() * db.sum() - (da * db).sum())
        edges = list(zip(PT_BINS[:-1], PT_BINS[1:]))
        ind = np.array([(pt >= lo) & (pt < hi) for lo, hi in edges], float)    # (n_bins, n_particles)
        ua = wa * r['A']['in_s'] * (ind - nbA[:, None]) / r['A']['N']          # share of dn_a(b)
        ub = wb * r['B']['in_s'] * (ind - nbB[:, None]) / r['B']['N']
        cov.append(0.5 * (ua.sum(1) * db.sum() - (ua * db).sum(1) + ub.sum(1) * da.sum() - (ub * da).sum(1)))
    v0sq = np.mean(pp) / (mA * mB)
    v0 = np.sqrt(v0sq)
    ncov = np.mean(cov, axis=0) / (0.5 * (nbA + nbB) * np.sqrt(mA * mB))
    return {'v0': np.array([v0]), 'normalized_covariance': ncov, 'v0_pT': ncov / v0,
            'v0_pT_over_v0': ncov / v0sq}[quantity]


def jackknife(rows, quantity, n_blocks=20):
    """Delete-block jackknife error of the particle-level estimator (which
    production equals): the statistical tolerance of the physics checks."""
    blocks = np.array_split(np.arange(len(rows)), n_blocks)
    est = np.array([particle_level([r for i, r in enumerate(rows) if i not in set(b)], quantity) for b in blocks])
    return np.sqrt((n_blocks - 1) / n_blocks * np.sum((est - est.mean(0)) ** 2, axis=0))


def rows_3d(fx):
    def w(e, X):
        lo, hi = A_ETA if X == 'A' else B_ETA
        return ((e['eta'] >= lo) & (e['eta'] < hi)).astype(float)
    return _per_event(fx, w)


def rows_2d(fx):
    qv = fx['qv']
    jac = {X: _make_jacobian_list(qv.pt_centers, qv.masses, *(A_ETA if X == 'A' else B_ETA)) / (2 * qv.rap_max)
           for X in 'AB'}

    def w(e, X):
        cell = np.clip(np.searchsorted(PT_EDGES, e['pt'], side='right') - 1, 0, len(PT_EDGES) - 2)
        return jac[X][e['species_idx'], cell]
    return _per_event(fx, w)


def expectation(fx):
    """v0 and v0(pT) from the k_e distribution: [pT](k) and n(pT)(k) from a
    large unscaled sample of the same spectra, averaged over k ~ N(1, sigma)."""
    if 'truth' not in _CACHE:
        base = _events(sigma=0.0, seed=99)['events']
        pt = np.concatenate([e['pt'][(np.abs(e['eta']) > 0.5) & (np.abs(e['eta']) < 2.5)] for e in base])
        ks = np.maximum(np.random.RandomState(1).normal(1, SIGMA, 4000), 0.05)
        P, N = [], []
        for k in ks:
            q = k * pt
            r, s = q[(q >= REF[0]) & (q < REF[1])], (q >= SPEC[0]) & (q < SPEC[1])
            P.append(r.mean())
            N.append([np.mean(((q >= lo) & (q < hi))[s]) for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])])
        P, N = np.array(P), np.array(N)
        v0 = P.std() / P.mean()
        ncov = ((N - N.mean(0)) * (P - P.mean())[:, None]).mean(0) / (N.mean(0) * P.mean())
        _CACHE['truth'] = {'v0': v0, 'v0_pT': ncov / v0}
    return _CACHE['truth']


def test_routing():
    print('test_routing')
    for q in ('v0', 'v0_pT', 'normalized_covariance', 'v0_pT_over_v0'):
        key = resolve(record(q))['compute']
        check(f'{q} -> radial_flow', key == 'radial_flow', key)
    check('one value per pT bin (single-row layout)', 'radial_flow' in SINGLE_ROW_KEYS)


def test_3d_equals_particle_level():
    print('test_3d_equals_particle_level')
    fx = _events()
    rows = rows_3d(fx)
    for q in ('v0', 'v0_pT', 'normalized_covariance', 'v0_pT_over_v0'):
        got, want = production(record(q), '3D', fx), particle_level(rows, q)
        check(f'3D {q}', np.allclose(got, want, rtol=1e-9, atol=1e-12), f'{np.round(got, 5)} vs {np.round(want, 5)}')


def test_2d_equals_particle_level():
    print('test_2d_equals_particle_level')
    fx = _events()
    rows = rows_2d(fx)
    for q in ('v0', 'v0_pT', 'normalized_covariance'):
        got, want = production(record(q), '2D', fx), particle_level(rows, q)
        check(f'2D {q}', np.allclose(got, want, rtol=1e-8, atol=1e-11), f'{np.round(got, 5)} vs {np.round(want, 5)}')


def test_physics():
    """Within 3 sigma (jackknife) plus 2% for the expectation's approximations
    (unscaled sample, eta windows applied before the rescale)."""
    print('test_physics')
    fx = _events()
    t = expectation(fx)
    for dim, rows in (('3D', rows_3d(fx)), ('2D', rows_2d(fx))):
        v0, e0 = production(record('v0'), dim, fx)[0], jackknife(rows, 'v0')[0]
        check(f'{dim} v0 = {t["v0"]:.4f}', abs(v0 - t['v0']) < 3 * e0 + 0.02 * t['v0'], f'{v0:.4f} +- {e0:.4f}')
        v, e = production(record('v0_pT'), dim, fx), jackknife(rows, 'v0_pT')
        check(f'{dim} v0(pT)', np.all(np.abs(v - t['v0_pT']) < 3 * e + 0.02 * np.abs(t['v0_pT'])),
              f'{np.round(v, 4)} vs {np.round(t["v0_pT"], 4)} (3 sigma {np.round(3 * e, 4)})')


def test_multiplicity_baseline():
    print('test_multiplicity_baseline')
    fx = _events(trend=0.5)
    t = expectation(fx)
    base = {'observable': 'N', 'eta': [{'min': -2.5, 'max': 2.5}], 'pT': {'min': 0.0, 'max': 100.0}}
    for dim, rows in (('3D', rows_3d(fx)), ('2D', rows_2d(fx))):
        e0 = jackknife(rows, 'v0')[0]
        plain = production(record('v0'), dim, fx)[0]
        fixed = production(record('v0', base), dim, fx)[0]
        check(f'{dim} class mean includes the multiplicity trend', plain > t['v0'] + 5 * e0, f'{plain:.4f}')
        check(f'{dim} fixed multiplicity recovers v0 = {t["v0"]:.4f}',
              abs(fixed - t['v0']) < 3 * e0 + 0.02 * t['v0'], f'{fixed:.4f} +- {e0:.4f}')


if __name__ == '__main__':
    tests = [test_routing, test_3d_equals_particle_level, test_2d_equals_particle_level, test_physics,
             test_multiplicity_baseline]
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
