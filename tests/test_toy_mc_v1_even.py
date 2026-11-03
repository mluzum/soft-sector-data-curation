"""Rapidity-even directed flow v1^even(pT) (v1_even: factorization of v11) through the production path.

Toy events carry an even v1(pT) = a (pT/p0 - 1) that changes sign at p0
(as measured: negative at low pT), a random Psi_1 per event and no
momentum-conservation correlation, so v11(pTa, pTb) = v1(pTa) v1(pTb).
Checks:
  - the pair-weighted rank-1 fit recovers an exact factorization for any
    weights and is the weighted least-squares stationary point otherwise;
  - 3D, no gap: the pipeline equals the particle-level rank-1 fit of the
    pair matrix (pT bins and eta window on grid edges): machine precision;
  - 2D and 3D (with and without |Delta eta| > 0.7): v1 per bin equals the
    particle average of the input profile within errors; the sign follows
    sign_convention (positive at high pT), the opposite convention flips it;
  - an integrated window: |v1| (absolute) and the signed value projected on
    a fit grid (fit_pT_bins) equal the input;
  - bins without pairs (beyond the container's pT range) give NaN and leave
    the rest of the fit unchanged;
  - records without a sign convention, or of another fit, are refused.
Run: python3 tests/test_toy_mc_v1_even.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, make_qgrid_data, evaluate, vn_of_pt,
)
from analysis.v1_even import v1_from_matrix  # noqa: E402

A, P0 = 0.06, 0.8                      # v1(pT) = A (pT/P0 - 1)
BINS = [[0.2, 0.5], [0.5, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2.2], [2.2, 3.0]]
ETA = {'min': -1.0, 'max': 1.0}
SIGN = {'pT': {'min': 1.6, 'max': 3.0}, 'sign': 'positive'}
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=1500, mult_mean=1500, mult_sigma=100, seed=2024, rap_max=1.0,
                           pt_min=0.2, pt_max=3.0, flow={1: dict(kind='crossing', value=A, pt0=P0),
                                                         2: dict(kind='const', value=0.06)})
        events, truth = generate_events(cfg)
        _CACHE['fx'] = (events, {'2d': make_qvector_data(events, truth, n_pt=28, n_harm=4),
                                 '3d': make_qgrid_data(events, truth, n_eta=30, n_pt=28, n_harm=4, eta_half=1.5)})
    return _CACHE['fx']


def rec(bins=BINS, v1_even=None, gap=None, integrated=None, **extra):
    r = {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': 2, 'harmonic': 1,
         'value_type': 'v1even', 'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1,
         'particle_id': {'species': 'charged'},
         'v1_even': v1_even if v1_even is not None else {'fit': 'factorization', 'sign_convention': SIGN},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -0.9, 'max': 0.9}]}}
    if integrated:
        r['dimensionality'] = 'integrated'
        r['kinematic_range'] = {'eta': ETA, 'pT': {'min': integrated[0], 'max': integrated[1]}}
    else:
        r['dimensionality'] = 'differential_pT'
        r['kinematic_range'] = {'eta': ETA, 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
    if gap:
        r['eta_gap'] = gap
    r.update(extra)
    return r


def run(r, kind, jk=True):
    c = _fx()[1][kind]
    kw = dict(cent_method='simple', with_jackknife=jk)
    return evaluate(r, qv=c, **kw) if kind == '2d' else evaluate(r, grid=c, **kw)


def truth(bins):
    """Particle average of the input v1(pT) per bin (|eta| < 1)."""
    events = _fx()[0]
    pt = np.concatenate([e['pt'][np.abs(e['eta']) < 1.0] for e in events])
    v = vn_of_pt(pt, 'crossing', A, P0)
    return np.array([v[(pt >= lo) & (pt < hi)].mean() for lo, hi in bins])


def particle_level(bins, sign_win):
    """Pair-weighted rank-1 fit of the distinct-pair v11 matrix from the particles."""
    events = _fx()[0]
    allb = list(bins) + [sign_win]
    K = len(allb)
    num, den = np.zeros((K, K)), np.zeros((K, K))
    for e in events:
        sel = np.abs(e['eta']) < 1.0
        pt, phi = e['pt'][sel], e['phi'][sel]
        q = np.array([np.exp(1j * phi[(pt >= lo) & (pt < hi)]).sum() for lo, hi in allb])
        m = np.array([((pt >= lo) & (pt < hi)).sum() for lo, hi in allb], float)
        shared = np.array([[((pt >= max(a[0], b[0])) & (pt < min(a[1], b[1]))).sum() for b in allb] for a in allb])
        num += np.real(q[:, None] * np.conj(q[None, :])) - shared
        den += m[:, None] * m[None, :] - shared
    V = num / den
    n = len(bins)
    return v1_from_matrix(V, list(range(n)), list(range(n)), n, 1.0, weights=den)


def test_fit():
    rng = np.random.default_rng(1)
    v = np.array([-0.03, -0.01, 0.02, 0.05, 0.09])
    W = rng.uniform(1e3, 1e5, (5, 5))
    W = W + W.T
    got = v1_from_matrix(np.outer(v, v), range(5), range(5), 4, 1.0, weights=W)
    check('exact rank-1 matrix: v recovered for any weights', np.allclose(got, v, rtol=1e-10))
    V = np.outer(v, v) + rng.normal(0, 1e-4, (5, 5)) / np.sqrt(W / 1e4)
    V = 0.5 * (V + V.T)
    got = v1_from_matrix(V, range(5), range(5), 4, 1.0, weights=W)
    grad = ((W * (V - np.outer(got, got))) @ got)
    check('noisy matrix: weighted least-squares stationary point', np.abs(grad).max() < 1e-9 * np.abs(W @ got ** 2).max())
    check('no positive eigenvalue: NaN', np.all(np.isnan(v1_from_matrix(-np.outer(v, v), range(5), range(5), 4))))


def test_empty_bins():
    """Bins beyond the container's pT range (ATLAS fits v_1,1 to 10 GeV) leave
    the fit: their v1 is NaN, the other bins equal the fit without them."""
    v_in, _ = run(rec(), '3d', jk=False)
    v_out, _ = run(rec(BINS + [[5.0, 7.0], [7.0, 10.0]]), '3d', jk=False)
    check('empty high-pT bins: NaN there, the rest unchanged',
          np.all(np.isnan(v_out[-2:])) and np.allclose(v_out[:-2], v_in, rtol=1e-12),
          f'{np.round(v_out, 4)}')


def test_dispatch():
    check('v1_even dispatches', resolve(rec())['compute'] == 'v1_even_factorization')
    check('integrated |v1| dispatches',
          resolve(rec(integrated=(0.4, 0.7), v1_even={'fit': 'factorization', 'absolute': True}))['compute']
          == 'v1_even_factorization')
    for name, v in (('no sign convention', {'fit': 'factorization'}),
                    ('another fit', {'fit': 'scalar_product', 'sign_convention': SIGN}),
                    ('bad sign', {'fit': 'factorization', 'sign_convention': {'pT': SIGN['pT'], 'sign': 'up'}})):
        try:
            resolve(rec(v1_even=v))
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    r = rec()
    del r['v1_even']
    try:
        resolve(r)
        ok = False
    except ValueError:
        ok = True
    check('harmonic 1 without v1_even or an odd reference still refused', ok)


def test_particle_level_3d():
    got, _ = run(rec(), '3d', jk=False)
    want = particle_level(BINS, (1.6, 3.0))
    check('3D (no gap) = particle-level rank-1 fit', np.allclose(got, want, rtol=1e-9, atol=1e-12),
          f'{np.round(got, 5)} vs {np.round(want, 5)}')


def test_truth():
    t = truth(BINS)
    for kind, gap in (('2d', None), ('3d', None), ('3d', 0.7)):
        v, e = run(rec(gap=gap), kind)
        pull = (v - t) / e
        check(f'{kind} gap={gap}: v1(pT) = input within errors', np.all(np.isfinite(v)) and np.all(np.abs(pull) < 4),
              f'v {np.round(v, 4)} truth {np.round(t, 4)} pull {np.round(pull, 1)}')
        check(f'{kind} gap={gap}: negative at low pT, positive at high pT', v[0] < 0 < v[-1])
    flip, _ = run(rec(v1_even={'fit': 'factorization', 'sign_convention': {'pT': SIGN['pT'], 'sign': 'negative'}}),
                  '3d', jk=False)
    same, _ = run(rec(), '3d', jk=False)
    check('sign: negative convention flips every bin', np.allclose(flip, -same))


def test_integrated():
    t = truth([[0.4, 0.7]])[0]
    for kind in ('2d', '3d'):
        v, e = run(rec(integrated=(0.4, 0.7), v1_even={'fit': 'factorization', 'absolute': True}), kind)
        check(f'{kind} |v1| in 0.4-0.7 = |input| within errors', abs(v[0] - abs(t)) < 4 * e[0],
              f'{v[0]:.5f} +- {e[0]:.5f} vs {abs(t):.5f}')
        v, e = run(rec(integrated=(0.4, 0.7), v1_even={'fit': 'factorization', 'fit_pT_bins': BINS,
                                                       'sign_convention': SIGN}), kind)
        check(f'{kind} signed v1 in 0.4-0.7 on a fit grid = input within errors', abs(v[0] - t) < 4 * e[0] and v[0] < 0,
              f'{v[0]:.5f} +- {e[0]:.5f} vs {t:.5f}')


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
