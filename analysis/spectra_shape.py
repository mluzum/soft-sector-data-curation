"""Sharing a grid cell's count among measurement bins that cut it.

A pT bin narrower than a container cell (ATLAS O+O spectra: three bins
inside the 0.2-0.4 GeV cell) gets part of the cell's count.  With a flat
invariant yield inside the cell (uniform in pT^2) every such bin repeats the
cell average, and the model/data ratio shows a sawtooth where the data
errors are small.  Here the share follows a local model of the
class-averaged spectrum:

    Y(p) = A exp(s m_T(p)) J(p),   m_T = sqrt(p^2 + m^2),

thermal in the rapidity density (m: the species' mass, the pion's for
charged hadrons, the assumed mass of a y_pi window), times, for a spectrum
per unit pseudorapidity, the window-averaged Jacobian
J(p) = < dy/deta > = < p cosh(eta) / sqrt(m^2 + p^2 cosh(eta)^2) > (1 for a
rapidity window).  The Jacobian matters at low pT: per unit eta, dN/dpT is
nearly flat across 0.2-0.4 GeV/c for pions, where a pure exponential gave
bins at the lower edge of the cell 8% low.

    s_{c,c+1}  the slope whose cell averages, int Y p dp / int p dp over
               cells c and c+1, are in the observed ratio (count / pT^2
               width): exact for spectra of this form;
    inside cell c: s = s_{c-1,c} below the cell's centre, s_{c,c+1} above
               (the one available slope at the ends; 0 next to an empty
               cell: the flat-in-m_T shape);
    share of cell c in [a, b] = int_a^b Y p dp / int_cell Y p dp.

Each cell keeps its count exactly (the shares of a cell tiled by bins sum
to 1) and bins made of whole cells are unchanged.  Computed per centrality
class (and jackknife replicate) from the class-averaged counts.  Integrals
by 24-point Gauss-Legendre (smooth integrands; |s width| capped at 40).
"""
import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_ETA_NODES = np.linspace(0.0, 1.0, 9)


def jacobian(mass, eta_window=None):
    """J(p) (vectorized): the window-averaged dy/deta, or 1 for eta_window None."""
    if eta_window is None:
        return lambda p: np.ones_like(np.asarray(p, float))
    lo, hi = float(eta_window[0]), float(eta_window[1])
    if lo < 0 < hi:                       # symmetric about 0: |eta| in [0, max]
        a, b = 0.0, max(-lo, hi)
    else:
        a, b = min(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    ch = np.cosh(a + (b - a) * _ETA_NODES)
    m2 = float(mass) ** 2

    def J(p):
        p = np.asarray(p, float)[..., None]
        return np.mean(p * ch / np.sqrt(m2 + (p * ch) ** 2), axis=-1)
    return J


def _integral(a, b, s, mref, mass, J):
    """int_a^b exp(s (m_T(p) - mref)) J(p) p dp, vectorized over a, b, s, mref."""
    a, b, s, mref = (np.asarray(x, float) for x in (a, b, s, mref))
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    p = mid[..., None] + half[..., None] * _GL_X
    mt = np.sqrt(p * p + float(mass) ** 2)
    val = half * np.sum(_GL_W * np.exp(s[..., None] * (mt - mref[..., None])) * J(p) * p, axis=-1)
    return np.where(b > a, val, 0.0)


def _pair_slopes(e, y, mass, J):
    """s per neighbouring pair of cells (bisection; 0 where a cell is empty)."""
    n = len(y) - 1
    s = np.zeros(max(n, 0))
    ok = (y[:-1] > 0) & (y[1:] > 0)
    if not ok.any():
        return s
    a0, a1, a2 = e[:-2][ok], e[1:-1][ok], e[2:][ok]
    target = np.log(y[1:][ok]) - np.log(y[:-1][ok])
    m2 = float(mass) ** 2
    ref = np.sqrt(a1 ** 2 + m2)
    lim = 40.0 / np.maximum(np.sqrt(a2 ** 2 + m2) - np.sqrt(a0 ** 2 + m2), 1e-9)
    w1, w0 = 0.5 * (a2 ** 2 - a1 ** 2), 0.5 * (a1 ** 2 - a0 ** 2)

    def g(x):
        return (np.log(_integral(a1, a2, x, ref, mass, J) / w1)
                - np.log(_integral(a0, a1, x, ref, mass, J) / w0) - target)
    lo, hi = -lim, lim
    for _ in range(80):                   # g increases with s
        mid = 0.5 * (lo + hi)
        up = g(mid) < 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    s[ok] = 0.5 * (lo + hi)
    return s


def slopes(edges, mean_counts, mass=0.0, eta_window=None):
    """(s_below, s_above) per cell (edges in pT)."""
    e = np.asarray(edges, float)
    y = np.asarray(mean_counts, float) / np.diff(e ** 2)
    n = len(y)
    s = _pair_slopes(e, y, mass, jacobian(mass, eta_window))
    below = np.concatenate([[s[0] if n > 1 else 0.0], s]) if n else np.zeros(0)
    above = np.concatenate([s, [s[-1] if n > 1 else 0.0]]) if n else np.zeros(0)
    return below, above


def shares(edges, mean_counts, parts, mass=0.0, eta_window=None):
    """Share of each part's cell inside its pT range: parts = ((cell, lo, hi), ...)."""
    if not parts:
        return np.zeros(0)
    e = np.asarray(edges, float)
    J = jacobian(mass, eta_window)
    below, above = slopes(e, mean_counts, mass, eta_window)
    out = np.zeros(len(parts))
    for k, (c, lo, hi) in enumerate(parts):
        c = int(c)
        e0, e1 = e[c], e[c + 1]
        pc = 0.5 * (e0 + e1)
        ref = np.sqrt(pc * pc + float(mass) ** 2)

        def cell(a, b):
            return (float(_integral(a, min(b, pc), below[c], ref, mass, J))
                    + float(_integral(max(a, pc), b, above[c], ref, mass, J)))
        full = cell(e0, e1)
        out[k] = cell(max(float(lo), e0), min(float(hi), e1)) / full if full > 0 else 0.0
    return out


def partial_cells(edges, bins):
    """((bin, cell, lo, hi), ...) for every cell that a bin covers only in part."""
    e = np.asarray(edges, float)
    out = []
    for b, (lo, hi) in enumerate(bins):
        for c in range(len(e) - 1):
            ov = min(hi, e[c + 1]) - max(lo, e[c])
            if ov > 1e-12 and ov < (e[c + 1] - e[c]) - 1e-12:
                out.append((b, c, float(lo), float(hi)))
    return tuple(out)
