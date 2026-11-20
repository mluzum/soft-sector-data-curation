"""Interval edges from the intervals' means and event fractions
(analysis.interval_edges; the classes of ALICE's and CMS's event-fraction
distributions, which tabulate each interval's mean N_ch / N_ch^0 and
fraction, not its edges).

Toy: a knee-shaped distribution (a falling plateau times an erfc tail, the
fixed-b Gaussian of the experiments' fits), cut into equal intervals; the
means and fractions from a fine integration.
Checks:
  - the mean of an exponential density on [0, w] (centroid_offset) against
    numerical integration;
  - noise-free: the edges within 1e-4 of the truth (midpoints 7e-4 off),
    the fractions in the reconstructed intervals within 0.3% of the truth in
    the tail (midpoints 5% high at the end);
  - means with 1e-4 noise (the spectra fits'): the tail fractions within 2%
    (midpoints 4.5% off), every interval mean within 3e-4 of the tabulated;
  - refusals: fewer than three intervals, decreasing means.
Run: python3 tests/test_interval_edges.py
"""
import os
import sys

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.interval_edges import centroid_offset, edges_from_means, interval_means  # noqa: E402

GRID = np.linspace(0.3, 1.3, 400001)
DENS = (1 - 0.6 * (GRID - 1)) * 0.5 * erfc((GRID - 1.137) / (np.sqrt(2) * 0.0348))
CDF = np.concatenate([[0.0], np.cumsum(DENS)])
XCDF = np.concatenate([[0.0], np.cumsum(DENS * GRID)])


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _toy(width=0.0098):
    E = np.arange(0.39, 1.24, width)
    i = np.searchsorted(GRID, E)
    y = np.diff(CDF[i])
    x = np.diff(XCDF[i]) / y
    total = CDF[i[-1]] - CDF[i[0]]
    return E, x, y / total, total


def _fractions(edges, total):
    i = np.searchsorted(GRID, edges)
    return np.diff(CDF[i]) / total


def _midpoints(x):
    m = 0.5 * (x[:-1] + x[1:])
    return np.concatenate([[x[0] - (m[0] - x[0])], m, [x[-1] + (x[-1] - m[-1])]])


def test_centroid_offset():
    print('test_centroid_offset')
    t = (np.arange(200000) + 0.5) / 200000
    worst = 0.0
    for u in (-3.0, -0.7, -1e-5, 0.0, 1e-5, 0.4, 2.5):
        d = np.exp(u * t)
        worst = max(worst, abs(np.sum(t * d) / np.sum(d) - 0.5 - float(centroid_offset(np.array([u]))[0])))
    check('the mean of exp(u t) on [0, 1] (small u by its series)', worst < 1e-6, f'{worst:.1e}')


def test_noise_free():
    print('test_noise_free')
    E, x, y, total = _toy()
    e, worst = edges_from_means(x, y)
    m = _midpoints(x)
    check('edges within 1e-4 of the truth', np.max(np.abs(e - E)) < 1e-4,
          f'{np.max(np.abs(e - E)):.1e} (midpoints {np.max(np.abs(m - E)):.1e})')
    r, r0 = _fractions(e, total) / y, _fractions(m, total) / y
    check('tail fractions within 0.3% of the truth', np.max(np.abs(r[-10:] - 1)) < 0.003,
          f'{np.max(np.abs(r[-10:] - 1)):.4f} (midpoints {np.max(np.abs(r0[-10:] - 1)):.4f})')
    check('  midpoints are worse than 3% at the end', np.max(np.abs(r0[-10:] - 1)) > 0.03)
    check('interval means reproduced', worst < 5e-5, f'{worst:.1e}')
    check('interval_means of the true edges = the toy means to 1e-4',
          np.max(np.abs(interval_means(E, y) - x)) < 1e-4, f'{np.max(np.abs(interval_means(E, y) - x)):.1e}')


def test_noisy_means_and_refusals():
    print('test_noisy_means_and_refusals')
    E, x, y, total = _toy()
    xn = x + np.random.default_rng(3).normal(0, 1e-4, len(x))
    e, worst = edges_from_means(xn, y)
    r, r0 = _fractions(e, total) / y, _fractions(_midpoints(xn), total) / y
    check('noise 1e-4: tail fractions within 2%', np.max(np.abs(r[-10:] - 1)) < 0.02,
          f'{np.max(np.abs(r[-10:] - 1)):.4f} (midpoints {np.max(np.abs(r0[-10:] - 1)):.4f})')
    check('noise 1e-4: interval means within 3e-4 of the tabulated', worst < 3e-4, f'{worst:.1e}')
    for name, args in (('two intervals', (x[:2], y[:2])), ('decreasing means', (x[::-1], y[::-1]))):
        try:
            edges_from_means(*args)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)


if __name__ == '__main__':
    tests = [test_centroid_offset, test_noise_free, test_noisy_means_and_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
