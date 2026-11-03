"""QGridData window sums: summed-area table, separable fractional sums, bounding boxes.

acceptance() sums Qn over a window from a lazily built summed-area table;
acceptance_fractional() on eta grids adds the whole cells from window sums
and the partial edge cells as strips; _weighted() (rapidity-binned grids)
sums on the bounding box of its weights.  Checks against the direct
formulas on a random grid with on- and off-grid window edges:
  - complex64 storage: table sums equal direct sums exactly (24-bit
    mantissas sum exactly in double precision);
  - complex128 storage: equal to rounding;
  - fractional and weighted sums equal the full einsum to rounding;
  - the table is built only after a grid's worth of direct work, and a
    copy with another Qn (a masked jackknife copy) does not reuse it;
  - summed_area_tables = False sums directly.
Run: python3 tests/test_qgrid_window_sums.py
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.qvector import QGridData, _overlap_fractions, _pt_squared, _snap_searchsorted  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def grid(dtype, n=300, A=40, P=25, H=4, seed=5):
    rng = np.random.default_rng(seed)
    Q = rng.poisson(2.0, (n, A, P, H)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (n, A, P, H)))
    Q[..., 0] = rng.poisson(2.0, (n, A, P))
    return QGridData(Qn=Q.astype(dtype), eta_edges=np.linspace(-4, 4, A + 1), pt_edges=np.linspace(0.0, 2.5, P + 1),
                     cent_mult=np.ones(n), nsamples=np.ones(n))


def windows(seed=9, k=200):
    rng = np.random.default_rng(seed)
    for t in range(k):
        lo, hi = np.sort(rng.uniform(-4.5, 4.5, 2))
        plo, phi = np.sort(rng.uniform(-0.2, 2.8, 2))
        if t % 3 == 0:
            lo, hi, plo, phi = (round(x, 1) for x in (lo, hi, plo, phi))
        yield lo, hi, plo, phi


def direct(g, lo, hi, plo, phi):
    ei = _snap_searchsorted(g.eta_edges, lo, hi)
    pi = _snap_searchsorted(g.pt_edges, plo, phi)
    return g.Qn[:, ei[0]:ei[1], pi[0]:pi[1], :].sum(axis=(1, 2), dtype=np.complex128)


def test_acceptance():
    for dtype, tol in ((np.complex64, 0.0), (np.complex128, 1e-13)):
        g = grid(dtype)
        scale = np.abs(g.Qn).sum(axis=(1, 2)).max()
        worst = max(np.abs(g.acceptance(*w) - direct(g, *w)).max() for w in windows())
        check(f'{np.dtype(dtype).name}: table window sums = direct sums (worst {worst / scale:.1e} of the totals)',
              g._sat_cache['table'] is not None and worst <= tol * scale)


def test_fractional_and_weighted():
    g = grid(np.complex64)
    Q = g.Qn.astype(np.complex128)
    scale = np.abs(Q).sum(axis=(1, 2)).max()
    rng = np.random.default_rng(2)
    wf = ww = 0.0
    for lo, hi, plo, phi in windows(seed=4):
        fe = _overlap_fractions(g.eta_edges, lo, hi)
        fp = _overlap_fractions(g.pt_edges, plo, phi, _pt_squared)
        wf = max(wf, np.abs(g.acceptance_fractional(lo, hi, plo, phi) - np.einsum('ijkh,j,k->ih', Q, fe, fp)).max())
        W = rng.uniform(0, 1, fe.shape + fp.shape) * (rng.uniform(0, 1, fe.shape + fp.shape) < 0.1)
        ww = max(ww, np.abs(g._weighted(W) - np.einsum('ijkh,jk->ih', Q, W)).max())
    check(f'fractional sums = full einsum (worst {wf / scale:.1e})', wf <= 1e-13 * scale)
    check(f'weighted sums on the bounding box = full einsum (worst {ww / scale:.1e})', ww <= 1e-13 * scale)
    z = g._weighted(np.zeros(fe.shape + fp.shape))
    check('all-zero weights: zero sums', z.shape == (g.n_events, g.Qn.shape[3]) and not z.any())


def test_cache_rules():
    g = grid(np.complex64)
    g.acceptance(-0.5, 0.5, 0.2, 0.5)
    check('a small first query does not build the table', g._sat_cache['table'] is None)
    for _ in range(3):
        g.acceptance(-4, 4, 0, 2.5)
    check('built after a grid of direct work', g._sat_cache['table'] is not None)
    m = copy.copy(g)
    m.Qn = g.Qn[::3].copy()
    w = (-1.0, 1.0, 0.3, 1.7)
    check('a copy with another Qn sums its own Qn', np.array_equal(m.acceptance(*w), direct(m, *w))
          and m._sat_cache is not g._sat_cache and g._sat_cache['table'] is not None)
    try:
        QGridData.summed_area_tables = False
        h = grid(np.complex64)
        for _ in range(5):
            h.acceptance(-4, 4, 0, 2.5)
        check('summed_area_tables = False: no table', h._sat_cache['table'] is None
              and np.array_equal(h.acceptance(*w), direct(h, *w)))
    finally:
        QGridData.summed_area_tables = True


if __name__ == '__main__':
    for t in (test_acceptance, test_fractional_and_weighted, test_cache_rules):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
