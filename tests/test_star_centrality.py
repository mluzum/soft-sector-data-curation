"""Experiment-defined centrality (STAR isobars, arXiv:2109.00131 Sec. III).

- threshold_anchor: the coverage f = F / <P(Ntrk >= threshold)> makes the
  most central fraction F of the experiment's total hold exactly the model
  events above the threshold: sharp ('simple') and Poisson weights;
- centrality.actual_classes replace the nominal labels as class edges
  (and must match them in number);
- a detector-level yield (efficiency_corrected: false) with
  detector_efficiency is dispatched and computed as efficiency x N.
Run: python3 tests/test_star_centrality.py
"""
import os
import sys

import numpy as np
import scipy.stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.weights import cent_weights, threshold_anchor  # noqa: E402


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _mult(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    return rng.exponential(120.0, n) + 2.0          # steeply falling, like Ntrk


def test_threshold_anchor_simple():
    print('test_threshold_anchor_simple')
    m, eff, thr, F = _mult(), 0.87, 50, 0.4712
    a = threshold_anchor(m, 'simple', thr, F, efficiency=eff)
    p = np.mean(eff * m >= thr)
    check('f = F / p', abs(a['f'] - F / p) < 1e-12, f"{a['f']:.5f} vs {F / p:.5f}")
    w = cent_weights('simple', eff * m, [(0.0, 100 * F)], n_ref=a['n_ref'])[:, 0]
    above = eff * m >= thr
    check('class 0-100F holds the events above threshold', abs(w.sum() - above.sum()) <= 1
          and w[above].sum() >= above.sum() - 1, f'{w.sum():.0f} vs {above.sum()}')


def test_threshold_anchor_poisson():
    print('test_threshold_anchor_poisson')
    m, eff, thr, F = _mult(), 0.87, 50, 0.4590
    a = threshold_anchor(m, 'poisson', thr, F, efficiency=eff)
    p = np.mean(scipy.stats.poisson.sf(thr - 1, eff * m))
    check('f = F / <P(N >= thr)>', abs(a['f'] - F / p) < 1e-12, f"{a['f']:.5f}")
    w = cent_weights('poisson', m, [(0.0, 100 * F)], efficiency=eff, anchor_cent=a['anchor_cent'])[:, 0]
    check('class 0-100F weight = sum P(N >= thr)', abs(w.sum() / (len(m) * p) - 1) < 1e-6,
          f'{w.sum():.2f} vs {len(m) * p:.2f}')
    try:
        threshold_anchor(m, 'poisson', 1e6, 0.5)
        check('unreachable threshold refused', False)
    except ValueError:
        check('unreachable threshold refused', True)


def _record(**cent):
    c = {'classes': ['40-50%', '50-60%'], 'estimator': {'name': 'RefMult', 'observable': 'Nch'},
         'eta': [{'min': -0.5, 'max': 0.5}], 'pT': {'min': 0.2, 'max': 100.0}}
    c.update(cent)
    return {'observable_type': 'particle_yields', 'inspire_id': '0', 'version': 1, 'hepdata_file': 't.yaml',
            'value_type': 'Nch', 'collision_system': {'energy_gev': 200.0, 'reaction': 'RU RU --> CHARGED X'},
            'centrality': c, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.2, 'max': 100.0}}}


def test_actual_classes_and_normalization_params():
    print('test_actual_classes_and_normalization_params')
    norm = {'estimator_threshold': 50, 'fraction_above': 0.4712, 'efficiency': 0.87}
    p = extract_measurement_params(_record(actual_classes=['39.89-49.86%', '49.86-60.29%'], normalization=norm))
    check('edges from actual_classes', p['cent_edges'] == [(39.89, 49.86), (49.86, 60.29)], str(p['cent_edges']))
    check('labels stay nominal', p['cent_classes'] == ['40-50%', '50-60%'])
    check('normalization passed on', p['cent_normalization'] == norm)
    try:
        extract_measurement_params(_record(actual_classes=['39.89-49.86%']))
        check('length mismatch refused', False)
    except ValueError:
        check('length mismatch refused', True)


def test_detector_efficiency_yield():
    print('test_detector_efficiency_yield')
    from analysis.pipeline_dispatch import get_pipeline_entry
    from analysis.qvector import QGridData
    rec = _record(classes=['0-100%'])
    rec['kinematic_range'] = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
    det = dict(rec, efficiency_corrected=False)
    try:
        resolve(det)
        check('detector-level without efficiency refused', False)
    except ValueError:
        check('detector-level without efficiency refused', True)
    det['detector_efficiency'] = 0.87
    resolve(det)
    check('detector-level with efficiency dispatched', True)
    eta_edges = np.round(np.linspace(-2, 2, 41), 10)
    Qn = np.zeros((3, 40, 1, 2), complex)
    Qn[:, :, 0, 0] = 10.0                                   # 100 per unit eta
    grid = QGridData(Qn=Qn, eta_edges=eta_edges, pt_edges=np.array([0.0, 100.0]), cent_mult=np.ones(3),
                     nsamples=np.ones(3), positives_only=False)
    out = {}
    for name, r in (('corrected', rec), ('detector', det)):
        ent = get_pipeline_entry(resolve(r)['compute'])
        p = extract_measurement_params(r)
        out[name] = float(np.asarray(ent['core_fn'](ent['prepare_data_3d'](grid, p), np.ones(3)))[0])
    check('detector-level = efficiency x corrected', abs(out['detector'] / out['corrected'] - 0.87) < 1e-12,
          f"{out['detector']:.4f} / {out['corrected']:.4f}")


if __name__ == '__main__':
    tests = [test_threshold_anchor_simple, test_threshold_anchor_poisson,
             test_actual_classes_and_normalization_params, test_detector_efficiency_yield]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            print(f'  PASS  {t.__name__}\n')
        except Exception as ex:
            print(f'  FAIL  {t.__name__}: {ex}\n')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
