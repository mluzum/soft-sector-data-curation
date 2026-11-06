"""Correlations between anisotropic flow and the mean transverse momentum.

The modified Pearson coefficient (P. Bozek, PRC 93 (2016) 044908; ATLAS,
PRC 107 (2023) 054910 Eq. 4; ALICE, PLB 834 (2022) 137393 Eq. 2):

    rho_n = cov_n / sqrt(var(v_n^2) c_k)

    cov_n      = sum_{i != j != k} w_i w_j w_k e^{in(phi_i - phi_j)} (pT_k - <[pT]>)
                 / sum_{i != j != k} w_i w_j w_k,
                 i in window X, j in Y, k in Z, averaged over the events of a
                 class with the triplet counts as weights;
    var(v_n^2) = c_n{4} + c_n{2}^2,   c_n{4} = <<4>> - 2 <<2>>^2,
                 the two terms possibly from different windows (ATLAS: c_n{4}
                 in the full window, c_n{2} from two subevents; ALICE:
                 v_n{2}^4 - v_n{4}^4 with both from subevents A and C);
    c_k        = <Delta pT_i Delta pT_j> over the pairs of window K
                 (analysis.observables_2d.compute_pT_fluct, pooled pairs).

Windows: 'full' (the record's kinematic_range) and the subevents 'A', 'B',
'C'.  cov methods: standard (X = Y = Z = full), two_subevent (X = A, Y = C,
Z = A + C: the pT particle from either), three_subevent (X = A, Y = C,
Z = B), combined_subevent (the mean of the two- and three-subevent cov).

Sums over distinct particles are the inclusion-exclusion over set
partitions of the members (as analysis.correlators.distinct_sum), each
member with its own window, harmonic and power of pT: a block contributes
the single-particle sum over the particles of all its members' windows,
weighted by the product of the members' window weights.  A particle's
window weight is its probability to be in the window (2D: the eta-window
Jacobian over the flat-y plateau, the pT-cell share, 1/nsamples; 3D: the
cell's overlap share, 1/nsamples), so windows that share no particles (3D
subevents on cell edges) give vanishing coincidence terms, and on the
boost-invariant 2D container, where every window holds every particle with
its weight, the coincidences are removed exactly.  A pT-weighted flow
vector, which the containers do not store, enters only through
coincidences of the pT particle with a flow particle (standard and
two-subevent methods), as the cell's flow vector times its mean pT.
"""
import itertools
from math import factorial

import numpy as np

from analysis.correlators import _PARTITIONS


class CellSums:
    """Single-particle sums S(h, windows, k) = sum_i prod_W a_W(i) e^{i h phi_i} pT_i^k.

    Qn    : (n_ev, ..., H) complex flow vectors per cell (cells: the
            leading axes after the event axis, h = 0..H-1);
    c, s1, s2 : (n_ev, ...) counts, sum pT, sum pT^2 per cell;
    windows : {name: weights broadcastable to a cell array (the cells)};
    ns    : (n_ev,) particlizations per event (weights a/ns).
    Sums are taken over views of the container's arrays and cached.
    """

    def __init__(self, Qn, c, s1, s2, windows, ns):
        self.Qn, self.c, self.s1, self.s2 = Qn, c, s1, s2
        self.win = {k: np.asarray(v, dtype=float) for k, v in windows.items()}
        self.ns = np.asarray(ns, dtype=float)
        self.H = Qn.shape[-1]
        self.cache = {}
        self.missing = set()
        self._inv_c = None
        axes = ''.join('ijklm'[:c.ndim - 1])
        self._sum = f'e{axes},{axes}->e'
        self._sum3 = f'e{axes},e{axes},{axes}->e'

    def weights(self, wins):
        a = 1.0
        for w in wins:
            a = a * self.win[w]
        return a

    def __call__(self, h, wins, k=0):
        key = (abs(h), tuple(sorted(wins)), k)
        if key not in self.cache:
            self.cache[key] = self._compute(*key)
        v = self.cache[key]
        return np.conj(v) if h < 0 else v

    def _compute(self, h, wins, k):
        a = self.weights(wins)
        n_ev = self.c.shape[0]
        if not np.any(a):
            return np.zeros(n_ev, complex)
        norm = self.ns ** len(wins)
        if h == 0:
            base = {0: self.c, 1: self.s1, 2: self.s2}[k]
            return np.einsum(self._sum, base, a).astype(complex) / norm
        if h >= self.H:
            self.missing.add(h)
            return np.zeros(n_ev, complex)
        q = self.Qn[..., h]
        if k == 0:
            return np.einsum(self._sum, q, a) / norm
        if k != 1:
            raise ValueError('pT-weighted flow vectors of power > 1 are not needed here')
        if self._inv_c is None:   # the cell's mean pT: s1 / c
            with np.errstate(divide='ignore', invalid='ignore'):
                self._inv_c = np.where(self.c > 0, self.s1 / self.c, 0.0)
        return np.einsum(self._sum3, q, self._inv_c, a) / norm


def distinct(S, members):
    """Sum over distinct particles of prod_m a_{W_m} e^{i h_m phi} pT^{k_m};
    members: (window, h, k) tuples (up to 4)."""
    total = 0.0
    for part in _PARTITIONS[len(members)]:
        coef, term = 1, 1.0
        for b in part:
            coef *= (-1) ** (len(b) - 1) * factorial(len(b) - 1)
            term = term * S(sum(members[m][1] for m in b), [members[m][0] for m in b],
                            sum(members[m][2] for m in b))
        total = total + coef * term
    return total


def _multinomial(alpha):
    k = sum(alpha)
    out = factorial(k)
    for a in alpha:
        out //= factorial(a)
    return out


def distinct_gf(S, types):
    """Ordered sum over distinct particles of prod over members of
    a_W(i) e^{i h phi_i}; types: [(window, h, count)], the members of a type
    interchangeable.  The coefficient of prod s_t^{count_t} of
    prod_i (1 + sum_t s_t a_{W_t}(i) e^{i h_t phi_i}) = exp(sum_alpha
    (-1)^{k+1}/k multinomial(alpha) p_alpha s^alpha), p_alpha the power sums
    S(sum alpha_t h_t, windows repeated alpha_t times), times prod count_t!."""
    dmax = tuple(int(c) for _, _, c in types)
    grid = list(itertools.product(*[range(c + 1) for c in dmax]))
    zero = tuple(0 for _ in dmax)
    L = {}
    for a in grid:
        k = sum(a)
        if k == 0:
            continue
        h = sum(at * types[t][1] for t, at in enumerate(a))
        wins = [types[t][0] for t, at in enumerate(a) for _ in range(at)]
        L[a] = ((-1) ** (k + 1) / k) * _multinomial(a) * S(h, wins)
    E = {zero: 1.0}
    for a in sorted(grid, key=sum):
        if a == zero:
            continue
        t = next(i for i, x in enumerate(a) if x > 0)
        acc = 0.0
        for b in itertools.product(*[range(x + 1) for x in a]):
            if b[t] == 0:
                continue
            rest = tuple(x - y for x, y in zip(a, b))
            acc = acc + b[t] * L[b] * E[rest]
        E[a] = acc / a[t]
    norm = 1
    for c in dmax:
        norm *= factorial(c)
    return E[dmax] * norm


COV_WINDOWS = {'standard': (('full',), ('full',), ('full',)),
               'two_subevent': (('A',), ('C',), ('A', 'C')),
               'three_subevent': (('A',), ('C',), ('B',))}


def _member(win, h, k):
    """A member whose window is a union of disjoint windows: their weights
    add.  Returned as a list of (window, h, k) alternatives."""
    return [(w, h, k) for w in win]


def _distinct_union(S, members):
    """distinct() with members on unions of windows (sum over the choices)."""
    total = 0.0
    for choice in np.ndindex(*[len(m) for m in members]):
        total = total + distinct(S, [m[i] for m, i in zip(members, choice)])
    return total


def event_sums(S, n, cov_methods, var_terms, ck_window):
    """Per-event sums for the core.

    cov_methods: names in COV_WINDOWS; var_terms: {'c4': 'standard' |
    'two_subevent', 'c2': ...}; ck_window: a window name."""
    out = {}
    for m in cov_methods:
        X, Y, Z = COV_WINDOWS[m]
        flow = [_member(X, n, 0), _member(Y, -n, 0)]
        out[f'Tp_{m}'] = np.real(_distinct_union(S, flow + [_member(Z, 0, 1)]))
        out[f'T1_{m}'] = np.real(_distinct_union(S, flow + [_member(Z, 0, 0)]))
        out[f'W_{m}'] = np.real(_distinct_union(S, [_member(X, 0, 0), _member(Y, 0, 0), _member(Z, 0, 0)]))
        out[f'PZ_{m}'] = sum(np.real(S(0, [w], 1)) for w in Z)
        out[f'NZ_{m}'] = sum(np.real(S(0, [w], 0)) for w in Z)
    for term in sorted(set(var_terms.values())):
        if term == 'standard':
            a, c = 'full', 'full'
        elif term == 'two_subevent':
            a, c = 'A', 'C'
        else:
            raise ValueError(f'var(v_n^2) term {term!r} (standard | two_subevent)')
        out[f'N2_{term}'] = np.real(distinct(S, [(a, n, 0), (c, -n, 0)]))
        out[f'D2_{term}'] = np.real(distinct(S, [(a, 0, 0), (c, 0, 0)]))
        if term == var_terms.get('c4'):
            out[f'N4_{term}'] = np.real(distinct(S, [(a, n, 0), (a, n, 0), (c, -n, 0), (c, -n, 0)]))
            out[f'D4_{term}'] = np.real(distinct(S, [(a, 0, 0), (a, 0, 0), (c, 0, 0), (c, 0, 0)]))
    K = ck_window
    out['ck_n'] = np.real(S(0, [K], 0))
    out['ck_s1'] = np.real(S(0, [K], 1))
    out['ck_s2'] = np.real(S(0, [K, K], 2))
    out['ck_t0'] = np.real(S(0, [K, K], 0))
    out['ck_t1'] = np.real(S(0, [K, K], 1))
    return out


def _wavg(x, u):
    s = u.sum()
    return np.sum(u * x) / s if s > 0 else np.nan


def _trend_var(y, mult, u):
    """slope^2 Var(mult) of a weighted linear regression of y on mult: the
    spread of the mean of y across the class that narrow multiplicity bins
    remove."""
    if mult is None or u.sum() <= 0:
        return 0.0
    dm = mult - _wavg(mult, u)
    var = _wavg(dm ** 2, u)
    if not var > 0:
        return 0.0
    slope = _wavg(dm * (y - _wavg(y, u)), u) / var
    return slope ** 2 * var


def core(d, w):
    """One value per class: rho_n (value_type rho_n), cov_n (cov_n), or
    cov_n / <[pT]> (cov_n_over_pT: over the class mean of the [pT] window,
    triplet-weighted as in cov_n; the mean of the methods' ratios for
    combined_subevent)."""
    from analysis.observables_2d import compute_pT_fluct
    nan = np.array([np.nan])
    w = np.asarray(w, dtype=float)
    if w.sum() < 2:
        return nan
    mult = d.get('mult')
    covs, means = [], []
    for m in d['cov_methods']:
        W = d[f'W_{m}']
        u = w * W
        if u.sum() <= 0:
            return nan
        pz = np.divide(d[f'PZ_{m}'], d[f'NZ_{m}'], out=np.zeros_like(W), where=d[f'NZ_{m}'] > 0)
        M = _wavg(pz, u)
        means.append(M)
        if mult is not None:
            dm = mult - _wavg(mult, u)
            var_m = _wavg(dm ** 2, u)
            if var_m > 0:
                M = M + _wavg(dm * (pz - M), u) / var_m * dm
        covs.append(np.sum(w * (d[f'Tp_{m}'] - M * d[f'T1_{m}'])) / u.sum())
    cov = float(np.mean(covs))
    if d['quantity'] == 'cov_n':
        return np.array([cov])
    if d['quantity'] == 'cov_n_over_pT':
        if min(means) <= 0:
            return nan
        return np.array([float(np.mean([c / mz for c, mz in zip(covs, means)]))])
    vt = d['var_terms']
    t4, t2 = vt['c4'], vt['c2']

    def k2(term):
        return np.sum(w * d[f'N2_{term}']) / np.sum(w * d[f'D2_{term}'])

    k4 = np.sum(w * d[f'N4_{t4}']) / np.sum(w * d[f'D4_{t4}'])
    var = k4 - 2.0 * k2(t4) ** 2 + k2(t2) ** 2
    if mult is not None:
        for term, f in ((t4, -2.0), (t2, 1.0)):
            D = d[f'D2_{term}']
            e2 = np.divide(d[f'N2_{term}'], D, out=np.zeros_like(D), where=D > 0)
            var += f * _trend_var(e2, mult, w * D)
    ck = compute_pT_fluct(d['ck_s1'], d['ck_s2'], d['ck_n'], w, d['ck_t0'], d['ck_t1'], mult, raw=True)[0]
    P = var * ck
    if not np.isfinite(P) or P == 0:
        return nan
    return np.array([cov / (np.sign(P) * np.sqrt(abs(P)))])
