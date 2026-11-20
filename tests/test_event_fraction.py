"""The fraction of the cross section in a class (multiplicity_distribution
records with value_type event_fraction; pipeline key event_fraction).

Checks, through the toy harness (the production prepare, weights and
jackknife):
  - percentile classes hold their width, in the full sample and in every
    replicate (a zero error), with a coverage f < 1 too;
  - classes of the estimator on an abscissa scale (centrality.abscissa_bins,
    'simple'): the events whose estimator lies in the class's interval over
    n_ref, with f = 1 and f = 0.8 (n_ref = n / f);
  - classes in units of the count at a percentile (normalized_to): the
    events with lo <= x / R < hi over n_ref;
  - importance weights: each event counts with its weight over the
    campaign's n_ref;
  - refusals: no normalization, classes in absolute track count, a
    threshold normalization, another value_type.
Run: python3 tests/test_event_fraction.py
"""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import evaluate  # noqa: E402


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _qv(seed=4):
    """2D toy container: sub-samples of falling multiplicity (roughly flat
    in centrality), events in random order."""
    from analysis.toy_mc import ToyGenConfig, generate_events, make_qvector_data
    parts = []
    for j, mult in enumerate(np.linspace(450, 30, 15)):
        cfg = ToyGenConfig(n_events=200, flow={2: dict(kind='const', value=0.05)}, fluctuate={2: 0.02},
                           mult_mean=float(mult), mult_sigma=0.15 * float(mult), seed=seed * 100 + j)
        ev, tr = generate_events(cfg)
        parts.append(make_qvector_data(ev, tr))
    qv = parts[0]
    order = np.random.default_rng(seed).permutation(sum(p.n_events for p in parts))
    cat = {f: np.concatenate([getattr(p, f) for p in parts])[order]
           for f in ('Qn', 'nsamples', 'sum_pt', 'sum_pt2', 'sum_mT') if getattr(qv, f) is not None}
    return dataclasses.replace(qv, **cat)


def _record(classes, centrality_extra=None, **kw):
    cent = {'classes': list(classes), 'estimator': {'name': 'ID', 'observable': 'Nch'},
            'eta': [{'min': -0.5, 'max': 0.5}]}
    cent.update(centrality_extra or {})
    rec = dict(observable_type='multiplicity_distribution', inspire_id='0', version=1, hepdata_file='t.yaml',
               observable='event_fraction', value_type='event_fraction', normalization='cross_section',
               efficiency_corrected=True, dimensionality='integrated',
               collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
               centrality=cent, particle_id={'species': 'charged'},
               kinematic_range={'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 6.0}})
    rec.update(kw)
    return rec


def _estimator(qv, eta=(-0.5, 0.5), pt=(None, None)):
    from analysis.toy_mc import _cent_mult_2d
    return _cent_mult_2d(qv, [eta], pt)


def test_percentile_classes():
    print('test_percentile_classes')
    qv = _qv()
    classes = ('0-5%', '5-10%', '10-30%', '30-60%')
    for anchor in (1.0, 0.8):
        v, e = evaluate(_record(classes), qv=qv, n_blocks=10, cent_anchor=anchor)
        check(f'f = {anchor}: fractions are the widths', np.allclose(v, [0.05, 0.05, 0.2, 0.3], atol=1e-12),
              f'{v}')
        check(f'f = {anchor}: zero error (every replicate holds the width)', np.all(e < 1e-10), f'{e}')


def test_abscissa_classes():
    print('test_abscissa_classes')
    from analysis.weights import AbscissaBinWeights, resolve_anchor
    qv = _qv()
    ab = {'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.0, 'max': 100.0}, 'reference_class': '0-5%'}
    classes = ('0.6-0.8', '0.8-0.95', '0.95-1.05', '1.05-1.2')
    est = _estimator(qv)
    absc = _estimator(qv, (-1.0, 1.0), (0.0, 100.0))
    for f in (1.0, 0.8):
        rec = _record(classes, {'abscissa_bins': ab})
        v, e = evaluate(rec, qv=qv, n_blocks=10, cent_method='simple', cent_anchor=f)
        a = resolve_anchor(f, qv.n_events)
        aw = AbscissaBinWeights('simple', est, absc, a, reference_class=(0.0, 5.0))
        lims = aw.edges_in_estimator([tuple(map(float, c.split('-'))) for c in classes])
        want = np.array([np.sum((est >= lo) & (est < hi)) for lo, hi in lims]) / a['n_ref']
        check(f'f = {f}: events in the estimator intervals over n_ref', np.allclose(v, want, rtol=1e-12),
              f'{v} vs {want}')
        check(f'f = {f}: finite errors', np.all(np.isfinite(e)) and np.all(e > 0), f'{e}')
    vp, _ = evaluate(_record(classes, {'abscissa_bins': ab}), qv=qv, n_blocks=10, cent_method='poisson')
    check('poisson: fractions close to the sharp ones', np.allclose(vp, want / 0.8 * 1.0, rtol=0.15), f'{vp}')


def test_normalized_classes_and_weights():
    print('test_normalized_classes_and_weights')
    qv = _qv()
    mb = {'efficiency': 1.0, 'normalized_to': {'percentile': 5.0}}
    classes = ('0.7-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.5')
    rec = _record(classes, {'multiplicity_bins': mb})
    v, _ = evaluate(rec, qv=qv, n_blocks=10, cent_method='simple')
    from analysis.weights import NormalizedMultiplicityBinWeights
    x = _estimator(qv)
    R = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 1.0, 'n_ref': float(len(x))},
                                         percentile=5.0).reference()
    check('  R: 5% of the events above it', abs(np.mean(x > R) - 0.05) < 2.0 / len(x), f'{np.mean(x > R):.4f}')
    want = np.array([np.mean((x / R >= float(c.split('-')[0])) & (x / R < float(c.split('-')[1])))
                     for c in classes])
    check('normalized_to: events with lo <= x/R < hi over n', np.allclose(v, want, atol=1e-12),
          f'{v} vs {want}')
    # importance weights: each event counts with its weight, over the
    # campaign's n_ref (here the sum of the weights: f = 1)
    rng = np.random.default_rng(2)
    k = rng.integers(1, 4, qv.n_events).astype(float)
    weighted = dataclasses.replace(qv, event_weights=k)
    vw, _ = evaluate(rec, qv=weighted, n_blocks=10, cent_method='simple', cent_anchor={'n_ref': float(k.sum())})
    Rw = NormalizedMultiplicityBinWeights('simple', x, anchor={'f': 1.0, 'n_ref': float(k.sum())}, percentile=5.0,
                                          event_weights=k).reference()
    want_w = np.array([np.sum(k * ((x / Rw >= float(c.split('-')[0])) & (x / Rw < float(c.split('-')[1]))))
                       for c in classes]) / k.sum()
    check('importance weights: sum of the weights in the class over n_ref', np.allclose(vw, want_w, atol=1e-12),
          f'{vw} vs {want_w}')
    vp, _ = evaluate(_record(('0-5%', '5-20%', '20-50%')), qv=weighted, n_blocks=10,
                     cent_anchor={'n_ref': float(k.sum())})
    check('importance weights: percentile classes still hold their widths', np.allclose(vp, [0.05, 0.15, 0.3],
                                                                                       atol=1e-12), f'{vp}')


def test_refusals():
    print('test_refusals')
    for name, rec in (
            ('no normalization', _record(('0-5%',), normalization=None)),
            ('classes in absolute track count', _record(('100-200',), {'multiplicity_bins': {'efficiency': 1.0}})),
            ('a threshold normalization', _record(('0-5%',), {'normalization': {
                'estimator_threshold': 10, 'fraction_above': 0.9}})),
            ('another value_type', _record(('0-5%',), value_type='P(N_trk)'))):
        try:
            resolve(rec)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    check('dispatched: percentile classes', resolve(_record(('0-5%',)))['compute'] == 'event_fraction')


if __name__ == '__main__':
    tests = [test_percentile_classes, test_abscissa_classes, test_normalized_classes_and_weights, test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
