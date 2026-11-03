"""Records needing a harmonic the 3D container does not store.

A container with n_harm = 4 (slots 0..3: counts and n = 1..3, as the O+O
JETSCAPE-framework QnVector output): v2{4}, v2{6} and v2{8} are computed
with the self-correlation terms that need Q_4, Q_6, Q_8 dropped (relative
order 1/M_eff, a printed note; until November 2026 the v_n{6,8} 3D prepare
returned None and the core then crashed the group's jackknife), SC(m, n)
with m + n > 3 is refused, and a v_n{2} record of a harmonic the container
lacks gives NaN.
Run: python3 tests/test_missing_harmonics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, evaluate, record_flow_vn2, SPECIES_MASSES  # noqa: E402

_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def grid():
    if 'g' not in _CACHE:
        cfg = ToyGenConfig(n_events=400, mult_mean=800, mult_sigma=80, seed=99,
                           flow={2: dict(kind='const', value=0.08), 3: dict(kind='const', value=0.03)},
                           fluctuate={2: 0.02})
        events, _ = generate_events(cfg)
        _CACHE['g'] = particles_to_qgrid_data(events, np.linspace(-3, 3, 31), np.linspace(0.0, 3.0, 31), n_harm=4,
                                              masses=SPECIES_MASSES)
    return _CACHE['g']


def rec(order, harmonic=2):
    r = record_flow_vn2(eta=(-1.0, 1.0), pt=(0.2, 3.0), cent_classes=('0-100%',))
    r.update(method='cumulant', cumulant_order=order, harmonic=harmonic)
    return r


def test_vn6_vn8_computed():
    v4 = evaluate(rec(4), grid=grid(), cent_method='simple', with_jackknife=False)[0][0]
    for order in (6, 8):
        v, _ = evaluate(rec(order), grid=grid(), cent_method='simple', with_jackknife=False)
        check(f'v2{{{order}}} computed without Q_{order} (close to v2{{4}})',
              np.isfinite(v[0]) and abs(v[0] / v4 - 1) < 0.1, f'{v[0]:.4f} vs {v4:.4f}')


def test_sc_refused():
    r = dict(observable_type='flow_harmonic_correlations', correlation_type='symmetric_cumulant',
             harmonics=[2, 3], normalized=True, value_type='NSC(3,2)', hepdata_file='toy.yaml', inspire_id='0',
             version=1, particle_id={'species': 'charged'},
             kinematic_range={'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 3.0}},
             centrality={'classes': ['0-100%'], 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]})
    try:
        get_pipeline_entry('sc_mn')['prepare_data_3d'](grid(), extract_measurement_params(r))
        ok = False
    except ValueError:
        ok = True
    check('SC(3,2) refused without Q_5', ok)


def test_vn4_computed():
    v, _ = evaluate(rec(4), grid=grid(), cent_method='simple', with_jackknife=False)
    check('v2{4} computed without Q_4 (self-correlation terms dropped)', np.isfinite(v[0]) and 0.05 < v[0] < 0.1,
          f'{v[0]:.4f}')


def test_missing_vn2_harmonic_nan():
    v, _ = evaluate(rec(2, harmonic=4), grid=grid(), cent_method='simple', with_jackknife=False)
    check('v4{2} without Q_4 gives NaN', not np.isfinite(v[0]), str(v))


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
