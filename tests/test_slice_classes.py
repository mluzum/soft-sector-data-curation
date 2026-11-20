"""Slices of a second count within a percentile class of the estimator
(centrality.slices; ATLAS PRL 133 (2024) 252301, Fig. 4: N_ch slices of the
FCal 0-1% events, Delta<[pT]>/<[pT]>_{0-1%} against Delta N_ch/<N_ch>_{0-1%};
analysis.weights.SliceWeights), and <[pT]> averaged over events
(mean_pT averaging: events).

Toy: per event a source s, a forward count F (the estimator) and a
midrapidity count A (the slices), both Poisson around multiples of s.
Checks:
  - 'simple': a class holds exactly the events of the estimator's `within`
    class whose A / <A>_within lies in it;
  - 'poisson': an event's weight is its weight in the `within` class times
    P(lo M <= N_trk < hi M), N_trk ~ Poisson(efficiency x A); tiling
    slices sum to the `within` weight times the covered range's probability;
  - a jackknife replicate rebuilds <N_trk>_within;
  - through the toy harness, 3D: <[pT]> averaged over events in each slice
    equals the mean of the events' [pT] (from the particles), and differs
    from the pooled <pT> of the 0-1% class when [pT] rises with A;
  - refusals: slices with abscissa_bins or multiplicity_bins, a `within`
    that is no percentile class, no eta windows; averaging events with
    per-bin <pT>.
Run: python3 tests/test_slice_classes.py
"""
import math
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.weights import CentralityWeights, SliceWeights, estimator_weights  # noqa: E402

N = 40000


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _toy(seed=5):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.0, 1.0, N) ** 0.5 * 400.0
    F = rng.poisson(3.0 * s).astype(float)
    A = rng.poisson(1.0 * s).astype(float)
    return F, A


def test_simple_and_poisson():
    print('test_simple_and_poisson')
    F, A = _toy()
    anchor = {'f': 1.0, 'n_ref': float(N)}
    prim = CentralityWeights('simple', F, anchor)
    sw = SliceWeights('simple', prim, A, (0.0, 1.0))
    inside = prim.weights([(0.0, 1.0)], raw=True)[:, 0] > 0
    M = A[inside].mean()
    check('<N_trk>_within: the mean over the class', abs(sw.mean() - M) < 1e-9 * M, f'{sw.mean():.3f} vs {M:.3f}')
    edges = [(0.9, 0.95), (0.95, 1.0), (1.0, 1.05), (1.05, 1.1)]
    W = sw.weights(edges, raw=True)
    ok = all(np.array_equal(W[:, k] > 0, inside & (A / M >= lo) & (A / M < hi)) for k, (lo, hi) in enumerate(edges))
    check('simple: the within-class events with lo <= A/<A> < hi', ok)
    check('  some events in each slice', np.all((W > 0).sum(axis=0) > 5), f'{(W > 0).sum(axis=0)}')
    eff = 0.6
    primp = CentralityWeights('poisson', F, anchor)
    swp = SliceWeights('poisson', primp, A, (0.0, 1.0), efficiency=eff)
    Wp = swp.weights(edges, raw=True)
    wp = primp.weights([(0.0, 1.0)], raw=True)[:, 0]
    Mp = np.sum(wp * eff * A) / np.sum(wp)
    want = wp * (scipy.stats.poisson.cdf(math.ceil(1.1 * Mp) - 1, eff * A)
                 - scipy.stats.poisson.cdf(math.ceil(0.9 * Mp) - 1, eff * A))
    check('poisson: tiling slices = within weight x P(0.9 M <= N_trk < 1.1 M)',
          np.allclose(Wp.sum(axis=1), want, atol=1e-12))
    mask = np.ones(N, bool)
    mask[np.argsort(-A)[:300]] = False                  # drop the highest A: <N_trk>_within falls
    check('a replicate rebuilds <N_trk>_within', swp.mean(mask) < swp.mean(), f'{swp.mean(mask):.2f} < {swp.mean():.2f}')
    est = estimator_weights('poisson', F, anchor, slices={'within': (0.0, 1.0), 'efficiency': eff}, slice_counts=A)
    check('estimator_weights builds it from the record', isinstance(est, SliceWeights))


def _grid(seed=8):
    """3D toy: [pT] rises with the midrapidity count at fixed forward count."""
    from analysis.converters import particles_to_qgrid_data
    from analysis.toy_mc import SPECIES_MASSES
    rng = np.random.default_rng(seed)
    events, mean_pt = [], []
    for _ in range(3000):
        s = rng.uniform(0.6, 1.0) ** 0.5
        nf = rng.poisson(900 * s) + 2               # forward, 3 < |eta| < 4
        nm = rng.poisson(600 * s) + 2               # midrapidity, |eta| < 1
        T = 0.45 * (nm / (600 * s)) ** 0.3           # hotter for more particles at the same source
        eta = np.concatenate([rng.uniform(3, 4, nf) * rng.choice([-1, 1], nf), rng.uniform(-1, 1, nm)])
        pt = np.concatenate([rng.exponential(0.5, nf), rng.exponential(T, nm)]) + 0.1
        pt = np.minimum(pt, 5.99)
        m = nf + nm
        events.append({'phi': rng.uniform(0, 2 * np.pi, m), 'eta': eta, 'pt': pt,
                       'species_idx': np.zeros(m, int), 'mass': np.full(m, SPECIES_MASSES[0])})
        mean_pt.append(pt[nf:].mean())
    grid = particles_to_qgrid_data(events, np.round(np.linspace(-4, 4, 81), 10), np.round(np.linspace(0, 6, 61), 10),
                                   n_harm=2, masses=SPECIES_MASSES)
    return grid, events, np.array(mean_pt)


def _record(classes, averaging='events', slices=True):
    cent = {'classes': list(classes), 'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
            'eta': [{'min': -4.0, 'max': -3.0}, {'min': 3.0, 'max': 4.0}]}
    if slices:
        cent['slices'] = {'within': '0-10%', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.0, 'max': 6.0},
                          'efficiency': 1.0}
    r = dict(observable_type='mean_pT', inspire_id='0', version=1, hepdata_file='t.yaml', value_type='mean_pT',
             observable='mean_pT', dimensionality='integrated',
             collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             centrality=cent, particle_id={'species': 'charged'},
             kinematic_range={'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.0, 'max': 6.0}})
    if averaging:
        r['averaging'] = averaging
    return r


def test_harness():
    print('test_harness')
    from analysis.toy_mc import evaluate
    grid, events, mpt = _grid()
    classes = ('0.9-0.96', '0.96-1.0', '1.0-1.04', '1.04-1.1')
    v, e = evaluate(_record(classes), grid=grid, n_blocks=10, cent_method='simple')
    F = np.array([np.sum(np.abs(ev['eta']) > 2) for ev in events], float)
    A = np.array([np.sum(np.abs(ev['eta']) < 1) for ev in events], float)
    inside = CentralityWeights('simple', F, {'f': 1.0, 'n_ref': float(len(F))}).weights([(0.0, 10.0)], raw=True)[:, 0] > 0
    M = A[inside].mean()
    want = []
    for c in classes:
        lo, hi = map(float, c.split('-'))
        sel = inside & (A / M >= lo) & (A / M < hi)
        want.append(mpt[sel].mean())
    check('events-averaged <[pT]> per slice = the mean of the events\' [pT]', np.allclose(v, want, rtol=1e-9),
          f'{v} vs {np.array(want)}')
    check('  rises across the slices', v[-1] > v[0], f'{v}')
    vr, _ = evaluate(_record(['0-10%'], slices=False), grid=grid, n_blocks=10, cent_method='simple')
    vp, _ = evaluate(_record(['0-10%'], averaging=None, slices=False), grid=grid, n_blocks=10, cent_method='simple')
    check('0-10%: the events average = the mean of the events\' [pT]', abs(vr[0] - mpt[inside].mean()) < 1e-9,
          f'{vr[0]:.6f} vs {mpt[inside].mean():.6f}')
    pooled = np.sum(A[inside] * mpt[inside]) / np.sum(A[inside])
    check('  the pooled <pT> is sum N [pT] / sum N, higher by cov(N, [pT]) / <N>',
          abs(vp[0] - pooled) < 1e-9 and vp[0] > vr[0], f'pooled {vp[0]:.6f} vs {pooled:.6f}')


def test_refusals():
    print('test_refusals')
    base = _record(('0.9-1.0',))
    cases = []
    r = _record(('0.9-1.0',))
    r['centrality']['abscissa_bins'] = {'eta': [{'min': -1, 'max': 1}], 'reference_class': '0-5%'}
    cases.append(('with abscissa_bins', r))
    r = _record(('0.9-1.0',))
    r['centrality']['multiplicity_bins'] = {'efficiency': 1.0}
    cases.append(('with multiplicity_bins', r))
    r = _record(('0.9-1.0',))
    r['centrality']['slices']['within'] = '0.9-1.1'
    cases.append(('a within that is no percentile class', r))
    r = _record(('0.9-1.0',))
    r['centrality']['slices'].pop('eta')
    cases.append(('no eta windows', r))
    r = _record(('0.9-1.0',), averaging='pairs')
    cases.append(('mean_pT averaging pairs', r))
    r = _record(('0.9-1.0',))
    r['dimensionality'] = 'differential_eta'
    r['kinematic_range']['eta'] = {'min': -1.0, 'max': 1.0, 'bins': [-1.0, 0.0, 1.0]}
    cases.append(('averaging events with per-bin <pT>', r))
    for name, rec in cases:
        try:
            resolve(rec)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    check('dispatched: mean_pT in slices', resolve(base)['compute'] == 'mean_pt')


if __name__ == '__main__':
    tests = [test_simple_and_poisson, test_harness, test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
