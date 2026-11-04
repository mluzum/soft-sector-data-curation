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


def _cosh_nodes(eta_window):
    """cosh(eta) at the nodes of |eta| over the window (J averages over them),
    or None for a rapidity window (J = 1)."""
    if eta_window is None:
        return None
    lo, hi = float(eta_window[0]), float(eta_window[1])
    if lo < 0 < hi:                       # symmetric about 0: |eta| in [0, max]
        a, b = 0.0, max(-lo, hi)
    else:
        a, b = min(abs(lo), abs(hi)), max(abs(lo), abs(hi))
    return np.cosh(a + (b - a) * _ETA_NODES)


def _jac(p, ch, m2):
    """J(p) on nodes p (shape S + (24,)); ch: cosh nodes broadcastable to S + (9,), or None."""
    if ch is None:
        return np.ones_like(p)
    pc = p[..., None] * ch[..., None, :]
    return np.mean(pc / np.sqrt(m2 + pc * pc), axis=-1)


def jacobian(mass, eta_window=None):
    """J(p) (vectorized): the window-averaged dy/deta, or 1 for eta_window None."""
    ch = _cosh_nodes(eta_window)
    m2 = float(mass) ** 2
    return lambda p: _jac(np.asarray(p, float)[..., None], ch, m2)[..., 0] if ch is not None \
        else np.ones_like(np.asarray(p, float))


def _integral(a, b, s, mref, mass, ch, weight='count'):
    """int_a^b exp(s (m_T(p) - mref)) J(p) p x(p) dp, x = 1 ('count'), p ('pT') or
    m_T ('mT'); vectorized over a, b, s, mref (shape S; ch broadcastable to S + (9,))."""
    a, b, s, mref = np.broadcast_arrays(*(np.asarray(x, float) for x in (a, b, s, mref)))
    half, mid = 0.5 * (b - a), 0.5 * (a + b)
    p = mid[..., None] + half[..., None] * _GL_X
    m2 = float(mass) ** 2
    mt = np.sqrt(p * p + m2)
    f = np.exp(s[..., None] * (mt - mref[..., None])) * _jac(p, ch, m2) * p
    if weight == 'pT':
        f = f * p
    elif weight == 'mT':
        f = f * mt
    elif weight != 'count':
        raise ValueError(f'unknown share weight {weight!r}')
    return np.where(b > a, half * np.sum(_GL_W * f, axis=-1), 0.0)


def _pair_slopes(e, y, mass, ch, left):
    """s of the cell pairs (left, left + 1) in every window: y (n_win, n_cells)
    cell averages (count / pT^2 width), ch (n_win, 9) or None; (n_win,
    len(left)), bisection; 0 where a cell of the pair is empty."""
    left = np.asarray(left, int)
    yl, yr = y[:, left], y[:, left + 1]
    ok = np.isfinite(yl) & np.isfinite(yr) & (yl > 0) & (yr > 0)
    s = np.zeros(yl.shape)
    if not ok.any():
        return s
    shape = yl.shape
    a0, a1, a2 = (np.broadcast_to(e[left + k], shape) for k in range(3))
    target = np.where(ok, np.log(np.where(ok, yr, 1.0)) - np.log(np.where(ok, yl, 1.0)), 0.0)
    m2 = float(mass) ** 2
    ref = np.sqrt(a1 ** 2 + m2)
    lim = 40.0 / np.maximum(np.sqrt(a2 ** 2 + m2) - np.sqrt(a0 ** 2 + m2), 1e-9)
    w1, w0 = 0.5 * (a2 ** 2 - a1 ** 2), 0.5 * (a1 ** 2 - a0 ** 2)
    chb = None if ch is None else ch[:, None, :]

    def g(x):
        return (np.log(_integral(a1, a2, x, ref, mass, chb) / w1)
                - np.log(_integral(a0, a1, x, ref, mass, chb) / w0) - target)
    lo, hi = -lim, lim
    for _ in range(80):                   # g increases with s
        mid = 0.5 * (lo + hi)
        up = g(mid) < 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), 0.0)


def slopes(edges, mean_counts, mass=0.0, eta_window=None):
    """(s_below, s_above) per cell (edges in pT)."""
    e = np.asarray(edges, float)
    y = (np.asarray(mean_counts, float) / np.diff(e ** 2))[None, :]
    n = y.shape[1]
    if n < 2:
        return np.zeros(n), np.zeros(n)
    ch = _cosh_nodes(eta_window)
    s = _pair_slopes(e, y, mass, None if ch is None else ch[None, :], np.arange(n - 1))[0]
    return np.concatenate([[s[0]], s]), np.concatenate([s, [s[-1]]])


def window_shares(edges, mean_counts, parts, mass=0.0, eta_windows=None, weights=('count',)):
    """Shares of cut cells in every window: {weight: (n_win, n_parts)}.

    mean_counts (n_win, n_cells): the class-averaged count of each cell in
    each window (only the cells next to the parts' cells are read); parts
    ((cell, lo, hi), ...), the same in every window; eta_windows: one eta
    window per window (the Jacobian of a density per unit eta), or None for
    rapidity windows.  weight 'count' is the share of the cell's count,
    'pT' (or 'mT') the share of its sum of pT (m_T): int_part Y p x dp /
    int_cell Y p x dp, the model of the module doc."""
    e = np.asarray(edges, float)
    y = np.atleast_2d(np.asarray(mean_counts, float)) / np.diff(e ** 2)
    n_win, n = y.shape
    if not parts:
        return {w: np.zeros((n_win, 0)) for w in weights}
    cells = np.array([int(c) for c, _, _ in parts])
    ch = None if eta_windows is None else np.stack([_cosh_nodes(w) for w in eta_windows])
    if n > 1:
        # the pair below and above each part's cell (the one available pair at the ends)
        lb = np.where(cells > 0, cells - 1, 0)
        la = np.where(cells < n - 1, cells, n - 2)
        left = np.unique(np.concatenate([lb, la]))
        s = _pair_slopes(e, y, mass, ch, left)
        sb, sa = s[:, np.searchsorted(left, lb)], s[:, np.searchsorted(left, la)]
    else:
        sb = sa = np.zeros((n_win, len(cells)))
    e0, e1 = e[cells], e[cells + 1]
    pc = 0.5 * (e0 + e1)
    ref = np.sqrt(pc * pc + float(mass) ** 2)
    lo = np.maximum(np.array([float(x) for _, x, _ in parts]), e0)
    hi = np.minimum(np.array([float(x) for _, _, x in parts]), e1)
    chp = None if ch is None else ch[:, None, :]
    out = {}
    for wt in weights:
        def cell(a, b):
            return (_integral(a, np.minimum(b, pc), sb, ref, mass, chp, wt)
                    + _integral(np.maximum(a, pc), b, sa, ref, mass, chp, wt))
        full, part = cell(e0, e1), cell(lo, hi)
        out[wt] = np.where(full > 0, part / np.where(full > 0, full, 1.0), 0.0)
    return out


def shares(edges, mean_counts, parts, mass=0.0, eta_window=None):
    """Share of each part's cell inside its pT range: parts = ((cell, lo, hi), ...)."""
    if not parts:
        return np.zeros(0)
    return window_shares(edges, np.asarray(mean_counts, float)[None, :], parts, mass,
                         None if eta_window is None else [eta_window])['count'][0]


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
