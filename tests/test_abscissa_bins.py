"""Classes of a forward estimator labelled on another count's scale
(centrality.abscissa_bins; CMS 2747107: 50 GeV intervals of HF E_T reported
at N_ch / N_ch^0 of |eta| < 0.5; analysis.weights.AbscissaBinWeights).

Toy: per event a "source" s (the collision's size), a forward count F and a
midrapidity count A, both Poisson around multiples of s; the classes are
intervals of F, labelled by A's conditional mean over its 0-5% mean.

Checks:
  - 'simple': a class holds exactly the events whose F lies between the
    edges the scale maps its labels to, i.e. intervals of the estimator;
  - the class mean of A over A's 0-5% mean falls inside the label, and the
    scale is monotone;
  - the reference: A's mean in the estimator's 0-5% class;
  - 'poisson': weights are probabilities of the estimator interval;
  - a jackknife replicate rebuilds the scale; a class beyond the scale's
    reach (labels above the largest conditional mean) is empty;
  - run through the toy harness (a mean-pT record), and refusals.
Run: python3 tests/test_abscissa_bins.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.weights import AbscissaBinWeights, CentralityWeights, conditional_mean  # noqa: E402

N = 40000


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _toy(seed=3):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, N) ** 0.5 * 400.0      # roughly flat in centrality
    F = rng.poisson(3.0 * s).astype(float)
    A = rng.poisson(1.0 * s).astype(float)
    return F, A


def test_intervals():
    print('test_intervals')
    F, A = _toy()
    aw = AbscissaBinWeights('simple', F, A, {'f': 1.0, 'n_ref': float(N)}, reference_class=(0.0, 5.0))
    edges = [(0.9, 0.95), (0.95, 1.0), (1.0, 1.05)]
    W = aw.weights(edges, raw=True)
    lims = aw.edges_in_estimator(edges)
    ok = all(np.array_equal(W[:, k] > 0, (F >= a) & (F < b)) for k, (a, b) in enumerate(lims))
    check('classes are intervals of the estimator', ok)
    check('contiguous labels give contiguous intervals', all(abs(lims[k][1] - lims[k + 1][0]) < 1e-9
                                                             for k in range(len(lims) - 1)))
    c, scale = aw.scale()
    check('the scale is monotone', np.all(np.diff(scale) >= 0) and np.all(np.diff(c) > 0))
    wref = CentralityWeights('simple', F, {'f': 1.0, 'n_ref': float(N)}).weights([(0.0, 5.0)], raw=True)[:, 0]
    a0 = np.sum(wref * A) / np.sum(wref)
    for k, (lo, hi) in enumerate(edges):
        m = np.sum(W[:, k] * A) / np.sum(W[:, k]) / a0
        check(f'class {lo}-{hi}: mean A / A^0 = {m:.4f} inside the label', lo - 0.01 < m < hi + 0.01)


def test_poisson_and_replicate():
    print('test_poisson_and_replicate')
    F, A = _toy()
    aw = AbscissaBinWeights('poisson', F, A, {'f': 1.0, 'n_ref': float(N)}, reference_class=(0.0, 5.0))
    W = aw.weights([(0.9, 1.0), (1.0, 1.1)], raw=True)
    check('poisson weights are probabilities', np.all(W >= -1e-12) and np.all(W.sum(axis=1) <= 1 + 1e-9))
    mask = np.ones(N, bool)
    mask[np.argsort(-F)[:400]] = False
    s_full, s_rep = aw.scale()[1], aw.scale(mask)[1]
    check('a replicate rebuilds the scale', not np.array_equal(s_full, s_rep))
    c, m = conditional_mean(F, A)
    check('conditional_mean: increasing centres and means', np.all(np.diff(c) > 0) and np.all(np.diff(m) >= 0))
    top = float(aw.scale()[1][-1])
    for method in ('poisson', 'simple'):
        a2 = AbscissaBinWeights(method, F, A, {'f': 1.0, 'n_ref': float(N)}, reference_class=(0.0, 5.0))
        Wb = a2.weights([(top + 0.05, top + 0.1), (0.0, 0.2)], raw=True)
        check(f'{method}: a class beyond the scale\'s reach is empty', np.all(Wb[:, 0] == 0.0))


def _record(classes=('0.8-0.95', '0.95-1.05', '1.05-1.2')):
    return dict(observable_type='mean_pT', inspire_id='0', version=1, hepdata_file='t.yaml', value_type='mean_pT',
                dimensionality='integrated', collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                centrality={'classes': list(classes), 'estimator': {'name': 'HF', 'observable': 'ET'},
                            'eta': [{'min': -1.0, 'max': -0.6}, {'min': 0.6, 'max': 1.0}],
                            'abscissa_bins': {'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.0, 'max': 100.0},
                                              'reference_class': '0-5%'}},
                particle_id={'species': 'charged'},
                kinematic_range={'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 6.0}})


def test_harness_and_refusals():
    print('test_harness_and_refusals')
    from analysis.converters import particles_to_qgrid_data
    from analysis.toy_mc import SPECIES_MASSES, evaluate
    rng = np.random.default_rng(9)
    events = []
    for _ in range(4000):
        s = rng.uniform(0, 1) ** 0.5
        m = rng.poisson(600 * s) + 2
        events.append({'phi': rng.uniform(0, 2 * np.pi, m), 'eta': rng.uniform(-1, 1, m),
                       'pt': rng.exponential(0.45 + 0.1 * s, m) + 0.1, 'species_idx': np.zeros(m, int),
                       'mass': np.full(m, SPECIES_MASSES[0])})
    grid = particles_to_qgrid_data(events, np.round(np.linspace(-2, 2, 41), 10), np.round(np.linspace(0, 6, 61), 10),
                                   n_harm=2, masses=SPECIES_MASSES)
    vals, errs = evaluate(_record(), grid=grid, n_blocks=10)
    check('harness: three classes, finite', np.all(np.isfinite(vals)) and np.all(errs > 0), f'{vals}')
    check('  <pT> rises with the abscissa (the toy: hotter for larger sources)', vals[2] > vals[0])
    for name, change in (('no reference class', {'reference_class': None}),
                         ('a reference class beyond 100%', {'reference_class': '90-110%'})):
        r = _record()
        r['centrality']['abscissa_bins'] = dict(r['centrality']['abscissa_bins'], **change)
        try:
            resolve(r)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    r = _record()
    r['centrality']['multiplicity_bins'] = {'efficiency': 1.0}
    try:
        resolve(r)
        ok = False
    except ValueError:
        ok = True
    check('refused: with multiplicity_bins', ok)


if __name__ == '__main__':
    tests = [test_intervals, test_poisson_and_replicate, test_harness_and_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
