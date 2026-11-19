"""Multi-particle cumulants with the standard and the three-subevent method,
normalized by two-subevent two-particle cumulants, in narrow reference
event classes (ATLAS, JHEP 01 (2020) 051, arXiv:1904.04808, Secs. 4-5).

A record states `cumulant_method` (standard | three_subevent) and is one of

  flow_vn, harmonic n, cumulant_order 2k (4 or 6; three_subevent: 4),
  `cumulant_quantity`:
    cumulant     c_n{2k}                    (Eq. 3)
    normalized   nc_n{4} = c_n{4} / c_n{2}^2, nc_n{6} = c_n{6} / (4 c_n{2}^3)
                                            (Eqs. 8, 9)
    coefficient  v_n{2k}, the signed root   (Eq. 4)
    ratio        v_n{2k} / v_n{ratio_order}: ratio_order 2 divides by the
                 two-subevent v_n{2} (Eq. 12), ratio_order 4 by v_n{4} of
                 the same method (v_2{6}/v_2{4})
  flow_harmonic_correlations, `correlation_type`:
    symmetric_cumulant, harmonics [n, m]: sc_{n,m}{4} = <<4>>_{n,m} -
        <<2>>_n <<2>>_m; normalized: nsc = sc / (c_n{2} c_m{2})  (Eq. 10)
    asymmetric_cumulant, harmonics [n]: ac_n{3} = <<e^{i(n phi_1 + n phi_2
        - 2n phi_3)}>>; normalized: nac = ac / sqrt((2 c_n{2}^2 + c_n{4})
        c_{2n}{2}), c_n{4} of the record's method       (Eq. 11)

The standard method takes every particle from the record's kinematic_range
('full').  The three-subevent method takes them from subevents A, B, C
(particles ABC; ATLAS's a, b, c), and each cumulant is the mean of the
three measurements obtained by interchanging the subevents (Sec. 4.2):

    c_n{4}      <<4>>_{2X|Y,Z} - 2 <<2>>_{X|Y} <<2>>_{X|Z}, two particles
                (+n, +n) in X, -n in Y and in Z;
    sc_{n,m}{4} <<e^{in(phi_X - phi_Y) + im(phi_X' - phi_Z)}>>
                - <<2>>_{n, X|Y} <<2>>_{m, X|Z};
                (X, Y, Z) = (A, B, C), (B, A, C), (C, B, A);
    ac_n{3}     <<e^{i(n phi_X + n phi_Y - 2n phi_Z)}>>, Z = A, B, C.

Every two-particle cumulant of a normalization (and v_n{2} of a ratio) is
c_n^{a|c}{2} between two subevents (Eq. 7): `normalization_subevents` {A,
C}, by default the three-subevent record's subevents A and C (a
standard-method record that normalizes states them).  `particles` assigns
the correlator's particles as everywhere: AABC (c_n{4}, sc of the
three-subevent method), ABC (ac_n{3}); none for the standard method.

Reference event classes (Sec. 5): ATLAS averages the correlators over the
events of a narrow interval of the estimator (FCal Sigma E_T in 0.002 TeV
steps, or the track count N_ch^rec in unit steps), forms each cumulant
there, and combines the intervals weighted by their numbers of events;
normalizations and ratios are formed from the combined cumulants.  A record
with `multiplicity_baseline` {eta, pT} (the estimator's acceptance) gets the
same: each correlator's interval average is its class average along a
weighted linear regression on that multiplicity within the class, the
cumulant is formed at each event's multiplicity and averaged over the
class's events.  That is exact for narrow intervals whenever the interval
averages are linear in the multiplicity across the class, and needs no
model intervals (which would hold a handful of events each: an O(1/N) bias
of every product of averages).  Without the baseline, every average is the
class average.

Every correlator is a sum over distinct particles (analysis.flow_pt's
set partitions, distinct, up to four members, the generating function
distinct_gf beyond), events weighted by their numbers of tuples.  The
single-particle sums come from SharedSums, cached per container by the
windows' ranges, so the tables of one window set (a family's four pT
ranges, both quantities) compute them once.  A harmonic the
container does not store is refused where it enters a leading term (sc
needs Q_{n+m}, ac Q_{2n}) and dropped, with a note, where it enters only
coincidence terms (c_n{4}: Q_{2n}; c_n{6}: Q_{3n}).
"""
import collections
import weakref

import numpy as np

from analysis.flow_pt import distinct, distinct_gf

METHODS = ('standard', 'three_subevent')
QUANTITIES = ('cumulant', 'normalized', 'coefficient', 'ratio')
CORRELATION_TYPES = ('symmetric_cumulant', 'asymmetric_cumulant')
# three-subevent measurements: the subevent with two particles first (c_n{4},
# sc), and for ac the subevent of the -2n particle last
TRIPLES = (('A', 'B', 'C'), ('B', 'A', 'C'), ('C', 'B', 'A'))
AC_TRIPLES = (('A', 'B', 'C'), ('B', 'C', 'A'), ('C', 'A', 'B'))


def _window(r, label):
    eta, pt = tuple(r.get('eta') or (None, None)), tuple(r.get('pT') or (None, None))
    if None in eta or None in pt:
        raise ValueError(f"multi-particle cumulants: {label} needs eta and pT")
    return eta, pt


def settings(params):
    """What a record computes: {kind: single | sc | ac, method, harmonics,
    order, quantity, ratio_order, normalized, windows {name: (eta, pT)}}.
    Raises ValueError for a combination the module does not compute."""
    method = params.get('cumulant_method')
    if method not in METHODS:
        raise ValueError(f"cumulant_method {method!r} (one of {', '.join(METHODS)})")
    ctype = params.get('correlation_type')
    st = {'method': method, 'normalized': bool(params.get('normalized'))}
    if ctype is None:
        n = int(params.get('harmonic') or 0)
        order = int(params.get('cumulant_order') or 0)
        q = params.get('cumulant_quantity')
        if q not in QUANTITIES:
            raise ValueError(f"cumulant_quantity {q!r} (one of {', '.join(QUANTITIES)})")
        if order not in (4, 6) or (method == 'three_subevent' and order != 4):
            raise ValueError(f"multi-particle cumulants: cumulant_order {order} (4 or 6; three_subevent: 4)")
        ro = params.get('ratio_order')
        if q == 'ratio':
            ro = int(ro or 0)
            if ro not in (2, 4) or ro >= order:
                raise ValueError(f"multi-particle cumulants: ratio_order {ro} (2, or 4 below order {order})")
        elif ro is not None:
            raise ValueError("ratio_order belongs to cumulant_quantity ratio")
        st.update(kind='single', harmonics=[n], order=order, quantity=q, ratio_order=ro)
        needs_ac = q == 'normalized' or (q == 'ratio' and ro == 2)
    elif ctype in CORRELATION_TYPES:
        harms = [int(h) for h in (params.get('harmonics') or [])]
        if ctype == 'symmetric_cumulant' and (len(harms) != 2 or harms[0] == harms[1]):
            raise ValueError(f"symmetric cumulant: two distinct harmonics, not {harms}")
        if ctype == 'asymmetric_cumulant' and len(harms) != 1:
            raise ValueError(f"asymmetric cumulant ac_n{{3}}: one harmonic n, not {harms}")
        if params.get('normalization_eta_gap') is not None:
            raise ValueError("multi-particle cumulants: the normalization is c_n{2} between subevents A and C, "
                             "not normalization_eta_gap")
        st.update(kind='sc' if ctype == 'symmetric_cumulant' else 'ac', harmonics=harms, order=4,
                  quantity=None, ratio_order=None)
        needs_ac = st['normalized']
    else:
        raise ValueError(f"multi-particle cumulants: correlation_type {ctype!r} "
                         f"(one of {', '.join(CORRELATION_TYPES)}, or a flow_vn record)")
    if min(st['harmonics']) < 2:
        raise ValueError("multi-particle cumulants: harmonics >= 2 (c_1{4} is not computed)")
    full = params.get('poi_range') or {}
    windows = {'full': _window(full, 'kinematic_range')}
    sr = params.get('subevent_ranges') or {}
    want = {'A', 'B', 'C'} if method == 'three_subevent' else set()
    if sorted(sr) != sorted(want):
        raise ValueError(f"multi-particle cumulants ({method}): subevents {''.join(sorted(want)) or 'none'} "
                         f"(particles), not {''.join(sorted(sr)) or 'none'}")
    for label in sorted(sr):
        windows[label] = _window(sr[label], f'subevent {label}')
    ns = params.get('normalization_subevents') or {}
    if needs_ac:
        if not ns and method == 'three_subevent':
            ns = {lab: sr[lab] for lab in ('A', 'C')}
        if sorted(ns) != ['A', 'C']:
            raise ValueError("multi-particle cumulants: the normalization c_n^(a|c){2} needs "
                             "normalization_subevents A and C (default: the three-subevent record's A and C)")
        for label in ('A', 'C'):
            windows['n' + label] = _window(ns[label], f'normalization subevent {label}')
    elif ns:
        raise ValueError("normalization_subevents without a normalization (normalized, or ratio_order 2)")
    st['windows'] = windows
    return st


def check_record(record):
    """resolve()'s check: the fields a multi-cumulant record needs."""
    kr = record.get('kinematic_range') or {}

    def rng(x):
        return (x.get('min'), x.get('max')) if isinstance(x, dict) else (None, None)

    def ranges(sub, labels):
        out = {}
        for lab in labels:
            k = (sub.get(lab) or {}).get('kinematic_range') or {}
            out[lab] = {'eta': rng(k.get('eta')), 'pT': rng(k.get('pT') or kr.get('pT'))}
        return out
    sub = record.get('subevents') or {}
    ns = record.get('normalization_subevents') or {}
    params = {
        'cumulant_method': record.get('cumulant_method'),
        'correlation_type': (record.get('correlation_type')
                             if record.get('observable_type') == 'flow_harmonic_correlations' else None),
        'harmonic': record.get('harmonic'), 'harmonics': record.get('harmonics'),
        'cumulant_order': record.get('cumulant_order'), 'cumulant_quantity': record.get('cumulant_quantity'),
        'ratio_order': record.get('ratio_order'), 'normalized': record.get('normalized'),
        'normalization_eta_gap': record.get('normalization_eta_gap'),
        'poi_range': {'eta': rng(kr.get('eta')), 'pT': rng(kr.get('pT'))},
        'subevent_ranges': ranges(sub, sorted(set(record.get('particles') or ''))),
        'normalization_subevents': ranges(ns, sorted(ns)),
    }
    if record.get('cumulant_method') == 'three_subevent':
        want = 'ABC' if record.get('correlation_type') == 'asymmetric_cumulant' else 'AABC'
        if record.get('particles') != want:
            raise ValueError(f"three-subevent method: particles {want!r}, not {record.get('particles')!r}")
    elif record.get('particles'):
        raise ValueError("standard method: no particles (the normalization's windows are "
                         "normalization_subevents)")
    if record.get('observable_type') not in ('flow_vn', 'flow_harmonic_correlations'):
        raise ValueError("cumulant_method: flow_vn or flow_harmonic_correlations records")
    if record.get('dimensionality', 'integrated') != 'integrated':
        raise ValueError("multi-particle cumulants are implemented for integrated records")
    return settings(params)


def needed_harmonics(st):
    """(leading, coincidence-only) harmonics a record's sums use."""
    h = st['harmonics']
    if st['kind'] == 'single':
        n = h[0]
        return {n}, {k * n for k in range(2, st['order'] // 2 + 1)}
    if st['kind'] == 'sc':
        n, m = h
        return {n, m, n + m}, {2 * n, 2 * m, abs(n - m)}
    n = h[0]
    return {n, 2 * n}, set()


# ── single-particle sums, shared by the records on one container ────────────

_STORES = {}


def _store(owner, tag):
    """The cache of one container (owner: its Q-vector array, kept by weak
    reference; tag: the cells' layout), emptied when the array goes."""
    key = (id(owner), tag)
    st = _STORES.get(key)
    if st is None or st['ref']() is not owner:
        st = {'ref': weakref.ref(owner), 'sums': {}, 'mats': collections.OrderedDict()}
        _STORES[key] = st
        weakref.finalize(owner, _STORES.pop, key, None)
    return st


def cached(owner, tag, key, fn):
    """fn() computed once per container (owner, tag) and key."""
    st = _store(owner, tag)
    vals = st.setdefault('values', {})
    if key not in vals:
        vals[key] = fn()
    return vals[key]


class SharedSums:
    """S(h, windows) = sum_i prod_W a_W(i) e^{i h phi_i} / ns^len(windows)
    per event (the interface of analysis.flow_pt.CellSums, pT powers 0).

    Qn: (n_ev, *cells, H); windows {name: weights over the cells}; ranges
    {name: the window's (eta, pT) ranges}.  A sum depends on the windows
    only through their ranges, so it is cached per container under them and
    every record with the same windows (a table family's four pT ranges)
    reuses it.  Each product of window weights gives the sums of every
    harmonic at once, one matrix product over the slice of the first cell
    axis (eta) that the windows reach -- a view of the container's array,
    upcast to double precision once when it is stored in single.  The
    container's arrays must not change while it is in use (they never do).
    """

    SUBS = 2      # upcast slices kept per container

    def __init__(self, Qn, windows, ranges, ns, owner=None, tag=''):
        self.Qn, self.H = Qn, Qn.shape[-1]
        self.n_ev = Qn.shape[0]
        cells = Qn.shape[1:-1]
        W = {k: np.broadcast_to(np.asarray(v, dtype=float), cells) for k, v in windows.items()}
        rows = np.zeros(cells[0], bool)
        for w in W.values():
            rows |= (w != 0).reshape(cells[0], -1).any(axis=1)
        nz = np.flatnonzero(rows)
        self.i0, self.i1 = (int(nz[0]), int(nz[-1]) + 1) if nz.size else (0, 0)
        self.W = {k: np.ascontiguousarray(w[self.i0:self.i1]).reshape(-1) for k, w in W.items()}
        self.ranges = dict(ranges)
        self.ns = np.asarray(ns, dtype=float)
        self.store = _store(Qn if owner is None else owner, tag)
        self.missing = set()

    def _sub(self):
        subs = self.store['mats']
        key = (self.i0, self.i1)
        if key not in subs:
            sub = self.Qn[:, self.i0:self.i1].reshape(self.n_ev, -1, self.H)
            if sub.dtype != np.complex128:
                sub = sub.astype(np.complex128)
            subs[key] = sub
            while len(subs) > self.SUBS:
                subs.popitem(last=False)
        subs.move_to_end(key)
        return subs[key]

    def __call__(self, h, wins, k=0):
        if k:
            raise ValueError('multi-particle cumulants use no pT-weighted sums')
        hh = abs(int(h))
        if hh >= self.H:
            self.missing.add(hh)
            return np.zeros(self.n_ev, complex)
        key = tuple(sorted(self.ranges[w] for w in wins))
        sums = self.store['sums']
        if key not in sums:
            a = 1.0
            for w in wins:
                a = a * self.W[w]
            a = np.asarray(a, dtype=float)
            if self.i1 > self.i0 and np.any(a):     # (disjoint 3D windows: zero)
                sums[key] = np.matmul(a, self._sub()) / (self.ns ** len(wins))[:, None]
            else:
                sums[key] = np.zeros((self.n_ev, self.H), complex)
        v = sums[key][:, hh]
        if hh == 0:
            v = v.real.astype(complex)
        return np.conj(v) if h < 0 else v


# ── per-event sums ──────────────────────────────────────────────────────────

def _sym(S, win, h, k):
    """Sum over 2k distinct particles of one window, k at +h and k at -h."""
    if 2 * k <= 4:
        return np.real(distinct(S, [(win, h, 0)] * k + [(win, -h, 0)] * k))
    return np.real(distinct_gf(S, [(win, h, k), (win, -h, k)] if h else [(win, 0, 2 * k)]))


def _pair(S, X, Y, h):
    return np.real(distinct(S, [(X, h, 0), (Y, -h, 0)]))


def _c4_sums(S, method, n, out, tag=''):
    """The sums of c_n{4} of a method into out (keys suffixed by tag)."""
    if method == 'standard':
        out[f'N2{tag}'], out['D2'] = _sym(S, 'full', n, 1), _sym(S, 'full', 0, 1)
        out[f'N4{tag}'], out['D4'] = _sym(S, 'full', n, 2), _sym(S, 'full', 0, 2)
        return
    for XY in ('AB', 'AC', 'BC'):
        out[f'P{XY}{tag}'], out[f'PD{XY}'] = _pair(S, XY[0], XY[1], n), _pair(S, XY[0], XY[1], 0)
    for X, Y, Z in TRIPLES:
        out[f'N4{X}{tag}'] = np.real(distinct(S, [(X, n, 0), (X, n, 0), (Y, -n, 0), (Z, -n, 0)]))
        out[f'D4{X}'] = np.real(distinct(S, [(X, 0, 0), (X, 0, 0), (Y, 0, 0), (Z, 0, 0)]))


def event_sums(S, st):
    """Per-event numerators and tuple counts for the core."""
    method, h = st['method'], st['harmonics']
    out = {}
    if st['kind'] == 'single':
        n = h[0]
        _c4_sums(S, method, n, out)
        if st['order'] == 6:
            out['N6'], out['D6'] = _sym(S, 'full', n, 3), _sym(S, 'full', 0, 3)
        if 'nA' in st['windows']:
            out['G2'], out['GD'] = _pair(S, 'nA', 'nC', n), _pair(S, 'nA', 'nC', 0)
    elif st['kind'] == 'sc':
        n, m = h
        if method == 'standard':
            out['N4'] = np.real(distinct(S, [('full', n, 0), ('full', m, 0), ('full', -n, 0), ('full', -m, 0)]))
            out['D4'] = _sym(S, 'full', 0, 2)
            out['N2n'], out['N2m'], out['D2'] = _sym(S, 'full', n, 1), _sym(S, 'full', m, 1), _sym(S, 'full', 0, 1)
        else:
            for XY in ('AB', 'AC', 'BC'):
                out[f'Pn{XY}'], out[f'Pm{XY}'] = _pair(S, XY[0], XY[1], n), _pair(S, XY[0], XY[1], m)
                out[f'PD{XY}'] = _pair(S, XY[0], XY[1], 0)
            for X, Y, Z in TRIPLES:
                out[f'N4{X}'] = np.real(distinct(S, [(X, n, 0), (Y, -n, 0), (X, m, 0), (Z, -m, 0)]))
                out[f'D4{X}'] = np.real(distinct(S, [(X, 0, 0), (Y, 0, 0), (X, 0, 0), (Z, 0, 0)]))
        if st['normalized']:
            out['G2n'], out['G2m'] = _pair(S, 'nA', 'nC', n), _pair(S, 'nA', 'nC', m)
            out['GD'] = _pair(S, 'nA', 'nC', 0)
    else:
        n = h[0]
        if method == 'standard':
            out['N3'] = np.real(distinct(S, [('full', n, 0), ('full', n, 0), ('full', -2 * n, 0)]))
            out['D3'] = np.real(distinct(S, [('full', 0, 0)] * 3))
        else:
            for X, Y, Z in AC_TRIPLES:
                out[f'N3{Z}'] = np.real(distinct(S, [(X, n, 0), (Y, n, 0), (Z, -2 * n, 0)]))
                out[f'D3{Z}'] = np.real(distinct(S, [(X, 0, 0), (Y, 0, 0), (Z, 0, 0)]))
        if st['normalized']:
            _c4_sums(S, method, n, out, tag='n')
            out['G2n'], out['GD'] = _pair(S, 'nA', 'nC', n), _pair(S, 'nA', 'nC', 0)
            out['G22n'] = _pair(S, 'nA', 'nC', 2 * n)
    out['settings'] = {k: v for k, v in st.items() if k != 'windows'}
    return out


# ── the core ────────────────────────────────────────────────────────────────

class Averager:
    """Class averages <<k>> = sum w N / sum w D, or, with a reference
    multiplicity per event, their values at each event's multiplicity along a
    weighted linear regression within the class (weights w D); collapse()
    then averages a quantity formed from them over the class's events
    (weights w).  Without multiplicity, collapse() is the identity."""

    def __init__(self, w, mult=None):
        self.w = np.asarray(w, dtype=float)
        self.on = mult is not None
        if self.on:
            sel = self.w > 0
            self.sel, self.ws = sel, self.w[sel]
            self.m = np.asarray(mult, dtype=float)[sel]

    def __call__(self, num, den):
        if not self.on:
            d = np.sum(self.w * den)
            return np.sum(self.w * np.real(num)) / d if d > 0 else np.nan
        n, d = np.real(num)[self.sel], np.asarray(den)[self.sel]
        u = self.ws * d
        su = u.sum()
        if not su > 0:
            return np.nan
        mu = np.sum(self.ws * n) / su
        dm = self.m - np.sum(u * self.m) / su
        var = np.sum(u * dm ** 2)
        slope = np.sum(self.ws * n * dm) / var if var > 0 else 0.0
        return mu + slope * dm

    def collapse(self, x):
        if not self.on:
            return x
        s = self.ws.sum()
        return np.sum(self.ws * np.broadcast_to(x, self.ws.shape)) / s if s > 0 else np.nan


def _c4(d, A, method, tag=''):
    """c_n{4} of the method (per event when A regresses), and <<2>>."""
    if method == 'standard':
        two = A(d[f'N2{tag}'], d['D2'])
        return A(d[f'N4{tag}'], d['D4']) - 2.0 * two ** 2, two
    p = {XY: A(d[f'P{XY}{tag}'], d[f'PD{XY}']) for XY in ('AB', 'AC', 'BC')}

    def pr(X, Y):
        return p[''.join(sorted(X + Y))]
    c4 = sum(A(d[f'N4{X}{tag}'], d[f'D4{X}']) - 2.0 * pr(X, Y) * pr(X, Z) for X, Y, Z in TRIPLES) / 3.0
    return c4, None


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def core(d, w):
    """One value per class."""
    from analysis.correlators import vn_from_cumulant
    st = d['settings']
    A = Averager(w, d.get('mult'))
    method = st['method']
    with np.errstate(divide='ignore', invalid='ignore'):
        if st['kind'] == 'single':
            order, q, ro = st['order'], st['quantity'], st['ratio_order']
            c4, two = _c4(d, A, method)
            if order == 6:
                c = A.collapse(A(d['N6'], d['D6']) - 9.0 * A(d['N4'], d['D4']) * two + 12.0 * two ** 3)
            else:
                c = A.collapse(c4)
            if q == 'cumulant':
                return np.array([c])
            if q == 'coefficient':
                return np.array([vn_from_cumulant(c, order)])
            if q == 'normalized' or ro == 2:
                c2 = A.collapse(A(d['G2'], d['GD']))
                if q == 'normalized':
                    return np.array([c / c2 ** 2 if order == 4 else c / (4.0 * c2 ** 3)])
                return np.array([vn_from_cumulant(c, order) / _sroot(c2)])
            return np.array([vn_from_cumulant(c, order) / vn_from_cumulant(A.collapse(c4), 4)])
        if st['kind'] == 'sc':
            if method == 'standard':
                sc = A(d['N4'], d['D4']) - A(d['N2n'], d['D2']) * A(d['N2m'], d['D2'])
            else:
                pn = {XY: A(d[f'Pn{XY}'], d[f'PD{XY}']) for XY in ('AB', 'AC', 'BC')}
                pm = {XY: A(d[f'Pm{XY}'], d[f'PD{XY}']) for XY in ('AB', 'AC', 'BC')}

                def k(X, Y):
                    return ''.join(sorted(X + Y))
                sc = sum(A(d[f'N4{X}'], d[f'D4{X}']) - pn[k(X, Y)] * pm[k(X, Z)] for X, Y, Z in TRIPLES) / 3.0
            sc = A.collapse(sc)
            if not st['normalized']:
                return np.array([sc])
            return np.array([sc / (A.collapse(A(d['G2n'], d['GD'])) * A.collapse(A(d['G2m'], d['GD'])))])
        if method == 'standard':
            ac = A(d['N3'], d['D3'])
        else:
            ac = sum(A(d[f'N3{Z}'], d[f'D3{Z}']) for _X, _Y, Z in AC_TRIPLES) / 3.0
        ac = A.collapse(ac)
        if not st['normalized']:
            return np.array([ac])
        c2n = A.collapse(A(d['G2n'], d['GD']))
        c4n = A.collapse(_c4(d, A, method, tag='n')[0])
        c22n = A.collapse(A(d['G22n'], d['GD']))
        return np.array([ac / _sroot((2.0 * c2n ** 2 + c4n) * c22n)])
