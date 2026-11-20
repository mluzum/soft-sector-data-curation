"""Classes in units of the estimator's count at a percentile of the cross
section (centrality.multiplicity_bins.normalized_to; ATLAS's N_ch /
N_ch^{5%}, PRL 133 (2024) 252301; analysis.weights.NormalizedMultiplicityBinWeights).

Checks:
  - the reference R is the count above which p percent of the cross
    section lies, with the anchor's coverage (f < 1: n_ref = n / f) and
    importance weights;
  - 'simple': a class 'lo-hi' holds exactly the events with lo <= x/R < hi;
  - 'poisson': the weights are P(lo R <= N_trk < hi R), tiling classes sum
    to the probability of the covered range;
  - a jackknife replicate rebuilds R from the events it keeps;
  - run through the toy harness (a mean-pT record in normalized classes),
    and the refusals (a percentile outside (0, 100), combined_from).
Run: python3 tests/test_normalized_multiplicity_bins.py
"""
import math
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.weights import NormalizedMultiplicityBinWeights, estimator_weights  # noqa: E402


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _sample(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 400.0, n)


def test_reference():
    print('test_reference')
    x = _sample()
    nw = NormalizedMultiplicityBinWeights('simple', x, efficiency=1.0, anchor={'f': 1.0, 'n_ref': float(len(x))},
                                          percentile=5.0)
    R = nw.reference()
    frac = np.mean(x > R)
    check('R: 5% of the events above it', abs(frac - 0.05) < 1.0 / len(x) + 1e-12, f'{frac:.6f}')
    nw8 = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 0.8, 'n_ref': len(x) / 0.8}, percentile=5.0)
    frac8 = np.mean(x > nw8.reference()) * 0.8
    check('coverage f = 0.8: 5% of the cross section above R', abs(frac8 - 0.05) < 2.0 / len(x), f'{frac8:.6f}')
    ew = np.where(x > np.median(x), 2.0, 1.0)
    nww = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 1.0, 'n_ref': float(ew.sum())},
                                           percentile=5.0, event_weights=ew)
    fw = np.sum(ew[x > nww.reference()]) / ew.sum()
    check('importance weights count in the percentile', abs(fw - 0.05) < 3.0 / len(x), f'{fw:.6f}')


def test_classes():
    print('test_classes')
    x = _sample()
    edges = [(0.9, 1.0), (1.0, 1.1), (1.1, 1.3)]
    nw = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 1.0, 'n_ref': float(len(x))}, percentile=5.0)
    R = nw.reference()
    W = nw.weights(edges, raw=True)
    ok = all(np.array_equal(W[:, k] > 0, (x / R >= lo) & (x / R < hi)) for k, (lo, hi) in enumerate(edges))
    check('simple: the events with lo <= x/R < hi', ok)
    xi = np.round(x).astype(float)
    nwp = NormalizedMultiplicityBinWeights('poisson', xi, efficiency=0.6, anchor={'f': 1.0, 'n_ref': float(len(x))},
                                           percentile=5.0)
    Rp = nwp.reference()
    Wp = nwp.weights(edges, raw=True)
    want = (scipy.stats.poisson.cdf(math.ceil(1.3 * Rp) - 1, 0.6 * xi)
            - scipy.stats.poisson.cdf(math.ceil(0.9 * Rp) - 1, 0.6 * xi))
    check('poisson: tiling classes sum to P(0.9 R <= N_trk < 1.3 R)', np.allclose(Wp.sum(axis=1), want, atol=1e-12))
    check('poisson: R in N_trk units (efficiency x N)', abs(Rp - np.quantile(0.6 * xi, 0.95)) < 2.0, f'{Rp:.1f}')
    ew = estimator_weights('simple', x, {'f': 1.0, 'n_ref': float(len(x))},
                           multiplicity_bins={'efficiency': 1.0, 'normalized_to': {'percentile': 5.0}})
    check('estimator_weights builds it from the record', isinstance(ew, NormalizedMultiplicityBinWeights))


def test_replicate():
    print('test_replicate')
    x = _sample()
    nw = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 1.0, 'n_ref': float(len(x))}, percentile=5.0)
    mask = np.ones(len(x), bool)
    mask[np.argsort(-x)[:200]] = False          # drop the 200 highest: R moves down
    R_full, R_rep = nw.reference(), nw.reference(mask)
    check('a replicate rebuilds R', R_rep < R_full, f'{R_rep:.1f} < {R_full:.1f}')
    W = nw.weights([(1.0, 1.2)], mask, raw=True)
    xs = x[mask]
    check('  and its classes from it', np.array_equal(W[:, 0] > 0, (xs / R_rep >= 1.0) & (xs / R_rep < 1.2)))


def test_harness_and_refusals():
    print('test_harness_and_refusals')
    from analysis.converters import particles_to_qgrid_data
    from analysis.toy_mc import SPECIES_MASSES, evaluate
    rng = np.random.default_rng(5)
    events = []
    for _ in range(3000):
        m = int(rng.gamma(3.0, 100.0)) + 5
        events.append({'phi': rng.uniform(0, 2 * np.pi, m), 'eta': rng.uniform(-1, 1, m),
                       'pt': rng.exponential(0.5, m) + 0.1, 'species_idx': np.zeros(m, int),
                       'mass': np.full(m, SPECIES_MASSES[0])})
    grid = particles_to_qgrid_data(events, np.round(np.linspace(-2, 2, 41), 10), np.round(np.linspace(0, 6, 61), 10),
                                   n_harm=2, masses=SPECIES_MASSES)
    rec = dict(observable_type='mean_pT', inspire_id='0', version=1, hepdata_file='t.yaml', value_type='mean_pT',
               dimensionality='integrated', collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
               centrality={'classes': ['0.8-0.95', '0.95-1.05', '1.05-1.3'], 'eta': [{'min': -1.0, 'max': 1.0}],
                           'estimator': {'name': 'ID', 'observable': 'Nch'},
                           'multiplicity_bins': {'efficiency': 1.0, 'normalized_to': {'percentile': 5.0}}},
               particle_id={'species': 'charged'},
               kinematic_range={'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.0, 'max': 6.0}})
    vals, errs = evaluate(rec, grid=grid, n_blocks=10)
    check('harness: three classes, finite', np.all(np.isfinite(vals)) and np.all(errs > 0), f'{vals}')
    for name, mb in (('percentile 0', {'efficiency': 1.0, 'normalized_to': {'percentile': 0.0}}),
                     ('percentile 100', {'efficiency': 1.0, 'normalized_to': {'percentile': 100.0}})):
        r = dict(rec, centrality=dict(rec['centrality'], multiplicity_bins=mb))
        try:
            resolve(r)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    r = dict(rec, centrality=dict(rec['centrality'], combined_from='unit_multiplicity'))
    try:
        resolve(r)
        ok = False
    except ValueError:
        ok = True
    check('refused: combined_from', ok)


if __name__ == '__main__':
    tests = [test_reference, test_classes, test_replicate, test_harness_and_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
