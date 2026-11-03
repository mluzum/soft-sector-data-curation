"""pT bins narrower than a container cell: a local thermal shape of the spectrum.

analysis.spectra_shape shares a cell's count among the bins that cut it by a
local model of the class-averaged spectrum, exp(s m_T) times the window's
dy/deta Jacobian, the slopes fitted to neighbouring cells (each cell keeps
its count; bins of whole cells unchanged).  Checks:
  - shares of a cell tiled by bins sum to 1; an empty neighbour gives the
    flat shape (uniform in pT^2);
  - an exactly exponential invariant yield (m = 0, no Jacobian) is
    reproduced exactly in sub-cell bins;
  - toy events with a thermal spectrum on 0.2 GeV cells, ATLAS-like bins
    inside the first cells (0.21-0.225, 0.225-0.24, ...): on 3D and 2D the
    pipeline agrees with the particle-level spectrum far better than the
    flat share did (3D: within 3%, from up to 30%; 2D within 6%, limited by
    its lowest cell), and bins of whole cells are unchanged.
Run: python3 tests/test_toy_mc_spectra_subcell.py
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.spectra_shape import partial_cells, shares  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, make_qvector_data, evaluate  # noqa: E402
from analysis import pipeline_dispatch as pd  # noqa: E402

results = []
_CACHE = {}
SUBBINS = [[0.21, 0.225], [0.225, 0.24], [0.24, 0.27], [0.27, 0.3], [0.3, 0.33], [0.33, 0.36], [0.36, 0.4],
           [0.4, 0.45], [0.45, 0.5], [0.5, 0.6], [0.6, 0.8], [0.8, 1.0], [1.0, 1.4], [1.4, 2.0]]
ALIGNED = [[0.2, 0.4], [0.4, 0.6], [0.6, 1.0], [1.0, 1.4], [1.4, 2.0]]


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def test_shares():
    e = np.linspace(0.0, 2.0, 11)
    y = np.exp(-np.linspace(0.1, 1.9, 10) / 0.3) * np.diff(e ** 2)
    parts = [(3, 0.6, 0.65), (3, 0.65, 0.7), (3, 0.7, 0.8)]
    check('shares of a tiled cell sum to 1', abs(shares(e, y, parts).sum() - 1) < 1e-12)
    y0 = y.copy()
    y0[2] = 0.0
    y0[4] = 0.0
    f = shares(e, y0, [(3, 0.6, 0.7)])
    check('empty neighbours: the flat shape (uniform in pT^2)', abs(f[0] - (0.7 ** 2 - 0.6 ** 2) / (0.8 ** 2 - 0.6 ** 2)) < 1e-12)
    # exactly exponential invariant yield: counts_c = int Y 2 pi p dp; sub-bins exact
    T = 0.25

    def integral(a, b):            # int exp(-p/T) p dp
        F = lambda p: -T * np.exp(-p / T) * (p + T)  # noqa: E731
        return F(b) - F(a)
    counts = np.array([integral(a, b) for a, b in zip(e[:-1], e[1:])])
    bins = [(0.2, 0.23), (0.23, 0.31), (0.31, 0.4), (0.45, 0.62), (1.05, 1.3)]
    pc = partial_cells(e, bins)
    f = shares(e, counts, [(c, lo, hi) for _, c, lo, hi in pc])
    est = np.zeros(len(bins))
    for (b, c, lo, hi), fk in zip(pc, f):
        est[b] += fk * counts[c]
    for b, (lo, hi) in enumerate(bins):        # whole cells
        est[b] += sum(counts[c] for c in range(10) if e[c] >= lo - 1e-12 and e[c + 1] <= hi + 1e-12)
    want = np.array([integral(a, b) for a, b in bins])
    check(f'exponential invariant yield exact in sub-cell bins (m = 0, no Jacobian; worst {np.max(np.abs(est / want - 1)):.1e})',
          np.allclose(est, want, rtol=1e-10))


def fx():
    if not _CACHE:
        cfg = ToyGenConfig(n_events=800, mult_mean=2000, mult_sigma=200, seed=13, rap_max=1.0, pt_min=0.0,
                           pt_max=3.0, flow={2: dict(kind='const', value=0.05)})
        ev, tr = generate_events(cfg)
        _CACHE['ev'] = ev
        _CACHE['3d'] = make_qgrid_data(ev, tr, n_eta=20, n_pt=15, n_harm=3, eta_half=2.0)     # 0.2 GeV cells
        _CACHE['2d'] = make_qvector_data(ev, tr, n_pt=15, n_harm=3)
    return _CACHE


def rec(bins):
    return {'observable_type': 'particle_spectra', 'inspire_id': '0', 'version': 1, 'hepdata_file': 'toy.yaml',
            'value_type': 'invariant_yield', 'observable': 'invariant_yield', 'particle_id': {'species': 'charged'},
            'centrality': {'classes': ['0-100%'], 'eta': [{'min': -2.0, 'max': -1.5}, {'min': 1.5, 'max': 2.0}]},
            'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}}


def truth(bins):
    """Particle-level invariant yield per bin, |eta| < 0.5 (both charges: the toy is positives x 2)."""
    ev = _CACHE['ev']
    pt = np.concatenate([e['pt'][np.abs(e['eta']) < 0.5] for e in ev])
    out = []
    for lo, hi in bins:
        n = ((pt >= lo) & (pt < hi)).sum() / len(ev)
        out.append(n / (np.pi * (hi * hi - lo * lo) * 1.0))
    return np.array(out)


def run(r, kind, flat=False):
    c = fx()[kind]
    saved = pd._wrap_spectra
    if flat:                           # the shares of the previous code: drop the subcell block
        pd.PIPELINE_DISPATCH['spectra_invariant_yield']['core_fn'] = _flat_core(pd.PIPELINE_DISPATCH['spectra_invariant_yield']['core_fn'])
    try:
        v, _ = (evaluate(r, grid=c, cent_method='simple', with_jackknife=False) if kind == '3d' else
                evaluate(r, qv=c, cent_method='simple', with_jackknife=False))
    finally:
        if flat:
            pd.PIPELINE_DISPATCH['spectra_invariant_yield']['core_fn'] = _CACHE['core']
    return np.asarray(v)


def _flat_core(core):
    _CACHE['core'] = core

    def flat(data, w):
        d = dict(data)
        d.pop('subcell', None)
        return core(d, w)
    return flat


def test_toy():
    fx()
    t = truth(SUBBINS)
    for kind in ('3d', '2d'):
        new = run(rec(SUBBINS), kind)
        old = run(rec(SUBBINS), kind, flat=True)
        scale = new[-1] / t[-1]          # 2D/3D normalization of the toy (charge factor, eta Jacobian)
        dn, do = new / (scale * t) - 1, old / (scale * t) - 1
        check(f'{kind} sub-cell bins: shape share closer to the particle-level spectrum '
              f'(worst {np.max(np.abs(dn)):.3f} vs flat {np.max(np.abs(do)):.3f})',
              np.max(np.abs(dn)) < 0.5 * np.max(np.abs(do)) and np.max(np.abs(dn[:9])) < (0.03 if kind == '3d' else 0.06),
              f'\n      new {np.round(dn[:9], 3)}\n      old {np.round(do[:9], 3)}')
        a_new, a_old = run(rec(ALIGNED), kind), run(rec(ALIGNED), kind, flat=True)
        check(f'{kind} bins of whole cells unchanged', np.allclose(a_new, a_old, rtol=1e-12))


if __name__ == '__main__':
    for t in (test_shares, test_toy):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
