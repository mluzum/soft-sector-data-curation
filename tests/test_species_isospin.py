"""Neutral pions and kaons by isospin (analysis.species).

pi0 ~ (pi+ + pi-)/2 and K0_S ~ (K+ + K-)/2: on a positives-only container
pi0 = pi+ and K0_S = K+ (charge factor 1, not 2); on an exact-charge PDG
axis half the sum of both charges; an explicit 111 / 310 axis is used when
present.  pi+ + pi- + pi0 is three pions.
Run: python3 tests/test_species_isospin.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.species import resolve_species, resolve_species_on_axis, species_known  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, make_qvector_data, evaluate, record_spectra  # noqa: E402

AXIS = [211, -211, 321, -321, 2212, -2212, 3122]
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def test_resolution():
    check('pi0 legacy = pi+', resolve_species('pi0') == ([0], [211], 1.0))
    check('K0_S legacy = K+', resolve_species('K0_S') == ([1], [321], 1.0))
    check('pi+pi-pi0 legacy = 3 pi+', resolve_species('pi+ + pi- + pi0')[2] == 3.0)
    check('pi0 exact = half both', resolve_species_on_axis('pi0', AXIS) == ([0, 1], 0.5, True))
    check('K0_S exact = half both', resolve_species_on_axis('K0_S', AXIS) == ([2, 3], 0.5, True))
    check('pi+pi-pi0 exact', resolve_species_on_axis('pi+ + pi- + pi0', AXIS) == ([0, 1], 1.5, True))
    check('pi0 positives-only axis', resolve_species_on_axis('pi0', [211, 321, 2212]) == ([0], 1.0, True))
    check('explicit pi0 axis wins', resolve_species_on_axis('pi0', [211, -211, 111]) == ([2], 1.0, True))
    check('known', species_known('pi0') and species_known('K0_S'))


def test_dispatch():
    check('pi0 spectra dispatch', resolve(record_spectra(species='pi0'))['compute'] == 'spectra_invariant_yield')


def _qv():
    if 'qv' not in _CACHE:
        cfg = ToyGenConfig(n_events=300, mult_mean=200, mult_sigma=30, seed=77)
        events, _ = generate_events(cfg)
        _CACHE['qv'] = make_qvector_data(events, {'cfg': cfg}, n_pt=40)
    return _CACHE['qv']


def test_spectra_on_positives_only_container():
    qv = _qv()
    kw = dict(qv=qv, cent_method='simple', with_jackknife=False)
    pi0, _ = evaluate(record_spectra(species='pi0', cent_classes=('0-100%',)), **kw)
    pip, _ = evaluate(record_spectra(species='pi+', cent_classes=('0-100%',)), **kw)
    pipm, _ = evaluate(record_spectra(species='pi+ + pi-', cent_classes=('0-100%',)), **kw)
    k0s, _ = evaluate(record_spectra(species='K0_S', cent_classes=('0-100%',)), **kw)
    kp, _ = evaluate(record_spectra(species='K+', cent_classes=('0-100%',)), **kw)
    check('pi0 = pi+', np.allclose(pi0, pip) and pi0.sum() > 0, f'{pi0[:2]} {pip[:2]}')
    check('pi+ + pi- = 2 pi0', np.allclose(pipm, 2 * pi0))
    check('K0_S = K+', np.allclose(k0s, kp) and k0s.sum() > 0, f'{k0s[:2]} {kp[:2]}')


if __name__ == '__main__':
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith('test_')]
    n_pass = 0
    for fn in tests:
        print(fn.__name__)
        try:
            fn()
            n_pass += 1
        except AssertionError as err:
            print(f'  FAILED: {err}')
    print(f'\n{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
