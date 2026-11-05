"""Multi-particle correlators of particles of interest (POI) and a reference (REF).

A k-particle correlator sums prod_m w_m exp(i h_m phi_m) over k DISTINCT
particles, member 1 (or none) from the POI sample and the others from the
reference sample.  The sum over distinct particles is the inclusion-exclusion
over set partitions of the members:

    sum_distinct = sum_{partitions pi} mu(pi) prod_{blocks B} S(B)
    mu(pi)       = prod_B (-1)^{|B|-1} (|B|-1)!

where S(B) is a single-particle sum over the particles that belong to every
member's sample in B (POI and REF overlap when both are in B), weighted by
the product of the members' weights, at the summed harmonic.  This holds
for any POI/REF overlap -- identified POI inside a charged reference, POI
outside it, or POI = REF -- and for particle weights (the 2D boost-invariant
container weights every particle by its window Jacobian).

Backends give S(h, n_poi, n_ref):
- Container2D: QVectorData species/pT slices and window weights;
- Container3D: species-selected unit-weight grids and eta/pT windows.

Differential and integrated v_n{2}, v_n{4} of the POI (Bilandzic et al.,
PRC 83, 044913; PRC 89, 064904):
    d2 = <<2'>>,  c2 = <<2>>,                      v'{2} = d2 / sqrt(c2)
    d4 = <<4'>> - 2 <<2'>><<2>>,  c4 = <<4>> - 2 <<2>>^2,  v'{4} = -d4 / (-c4)^(3/4)
With POI = REF they reduce to v{2} = sqrt(c2) and v{4} = (-c4)^(1/4).  Signs
follow the existing convention: sign(c2) sqrt|c2| and sign(-c4) |c4|^(1/4)
for POI = REF.
"""
from math import factorial

import numpy as np


def set_partitions(items):
    """All partitions of a list into blocks (lists)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


_PARTITIONS = {k: [[tuple(b) for b in p] for p in set_partitions(list(range(k)))] for k in (1, 2, 3, 4)}


def distinct_sum(S, harmonics, roles):
    """Sum over distinct particles of prod_m w_m e^{i h_m phi_m}.

    S(h, n_poi, n_ref) -> (n_events,) complex: single-particle sum over the
    particles in every member's sample, weights to the members' powers.
    roles: 'P' or 'R' per member."""
    total = 0.0
    for part in _PARTITIONS[len(harmonics)]:
        coef = 1
        term = 1.0
        for b in part:
            coef *= (-1) ** (len(b) - 1) * factorial(len(b) - 1)
            nP = sum(1 for m in b if roles[m] == 'P')
            term = term * S(sum(harmonics[m] for m in b), nP, len(b) - nP)
        total = total + coef * term
    return total


_SYM_TYPES = {}


def _symmetric_types(m):
    """Set partitions of 2m members (m with +h, m with -h) aggregated by
    block type: {(sorted ((n_plus, n_minus) per block)): summed Moebius
    coefficient}.  A block of a single sample contributes S((n_plus -
    n_minus) h, 0, n_plus + n_minus), so partitions of one type give the same
    product (8 members: 4140 partitions, a few dozen types)."""
    if m not in _SYM_TYPES:
        types = {}
        for part in set_partitions(list(range(2 * m))):
            coef = 1
            key = []
            for b in part:
                coef *= (-1) ** (len(b) - 1) * factorial(len(b) - 1)
                a = sum(1 for x in b if x < m)
                key.append((a, len(b) - a))
            key = tuple(sorted(key))
            types[key] = types.get(key, 0) + coef
        _SYM_TYPES[m] = {k: c for k, c in types.items() if c != 0}
    return _SYM_TYPES[m]


def symmetric_distinct_sum(S, h, m):
    """Sum over 2m distinct particles of one (reference) sample of
    prod w_i exp(i h (phi_1 + .. + phi_m - phi_{m+1} - .. - phi_2m)): the
    numerator of <2m>_{h..h,-h..-h} (h = 0: the weighted 2m-tuple count).
    Exact for any particle weights; S(k, 0, p) = sum w^p e^{i k phi}."""
    total = 0.0
    for key, coef in _symmetric_types(m).items():
        term = coef
        for a, b in key:
            term = term * S((a - b) * h, 0, a + b)
        total = total + term
    return total


def reference_cumulant_arrays(S, n_harm, order):
    """Per-event numerators and denominators of <2>, <4>, ..., <order> of one
    reference sample for harmonics 1..n_harm-1: N2, D2, N4, D4, (N6, D6, N8,
    D8).  The single-particle sums at k h beyond the stored harmonics are 0:
    the terms they enter are self-correlations of k coinciding particles
    (k >= 2), suppressed by powers of 1/M_eff (cf. _note_missing_q2n)."""
    out = {}
    hs = range(1, n_harm)
    for m in range(1, order // 2 + 1):
        out[f'N{2 * m}'] = np.stack([symmetric_distinct_sum(S, h, m) for h in hs], axis=1)
        out[f'D{2 * m}'] = np.real(symmetric_distinct_sum(S, 0, m))
    return out


def reference_cumulants(arrays, w, order):
    """c_n{2}, ..., c_n{order} for harmonics 1..N-1 from reference_cumulant_arrays."""
    w = np.asarray(w, dtype=float)
    k = {2 * m: _avg(arrays[f'N{2 * m}'], arrays[f'D{2 * m}'], w) for m in range(1, order // 2 + 1)}
    c = {2: k[2]}
    if order >= 4:
        c[4] = k[4] - 2.0 * k[2] ** 2
    if order >= 6:
        c[6] = k[6] - 9.0 * k[4] * k[2] + 12.0 * k[2] ** 3
    if order >= 8:
        c[8] = k[8] - 16.0 * k[6] * k[2] - 18.0 * k[4] ** 2 + 144.0 * k[4] * k[2] ** 2 - 144.0 * k[2] ** 4
    return c


def vn_from_cumulant(c, order):
    """v_n{order} with the sign convention of the cores: sign(c) |c / norm|^(1/k)
    for the expected sign (v{2}: +c2; v{4}: -c4; v{6}: +c6/4; v{8}: -c8/33)."""
    norm = {2: 1.0, 4: -1.0, 6: 4.0, 8: -33.0}[order]
    x = c / norm
    return np.sign(x) * np.abs(x) ** (1.0 / order)


def correlator_arrays(S, n_harm, order, which=('poi', 'ref')):
    """Per-event numerators and denominators for harmonics 1..n_harm-1.

    'poi': N2p (n_events, n_harm-1) complex, D2p (n_events,) real, and with
    order 4 N4p, D4p; 'ref': N2, D2 (and N4, D4)."""
    out = {}
    hs = range(1, n_harm)
    specs = []
    if 'poi' in which:
        specs += [('2p', (1, -1), 'PR')] + ([('4p', (1, 1, -1, -1), 'PRRR')] if order >= 4 else [])
    if 'ref' in which:
        specs += [('2', (1, -1), 'RR')] + ([('4', (1, 1, -1, -1), 'RRRR')] if order >= 4 else [])
    for tag, sign, roles in specs:
        out['N' + tag] = np.stack([distinct_sum(S, tuple(h * x for x in sign), roles) for h in hs], axis=1)
        out['D' + tag] = np.real(distinct_sum(S, (0,) * len(sign), roles))
        if len(sign) == 4:
            _note_missing_q2n(S, [h for h in hs if 2 * h >= n_harm], n_harm)
    return out


_NOTED = set()


def _note_missing_q2n(S, harms, n_harm):
    """<4>_{n,n,-n,-n} needs Q_{2n} where two members coincide: only
    weight-squared self-correlation terms, of relative order 1/M_eff
    (M_eff = (sum w)^2 / sum w^2).  A container that does not store Q_{2n}
    drops them; say so once, with M_eff (oversampled 2D JETSCAPE containers:
    ~2e4, a < 1% effect on c3{4}; unit weights: M_eff = M)."""
    key = (tuple(harms), n_harm)
    if not harms or key in _NOTED:
        return
    _NOTED.add(key)
    w1, w2 = np.real(S(0, 0, 1)), np.real(S(0, 0, 2))
    ok = w2 > 0
    m_eff = float(np.median(w1[ok] ** 2 / w2[ok])) if ok.any() else float('nan')
    print(f"  note: v_n{{4}} for n = {', '.join(map(str, harms))}: Q_2n is not stored (harmonics < {n_harm}); "
          f"its self-correlation terms (relative order 1/M_eff, median M_eff = {m_eff:.3g}) are dropped")


def sc_arrays(S, m, n):
    """Per-event sums for the symmetric cumulant SC(m, n) = <<4>>_{m,n,-m,-n}
    - <<2>>_m <<2>>_n over distinct particles of one sample: N4, D4, N2m, N2n,
    D2 (Bilandzic et al., PRC 89, 064904)."""
    return {'N4': distinct_sum(S, (m, n, -m, -n), 'RRRR'), 'D4': np.real(distinct_sum(S, (0, 0, 0, 0), 'RRRR')),
            'N2m': distinct_sum(S, (m, -m), 'RR'), 'N2n': distinct_sum(S, (n, -n), 'RR'),
            'D2': np.real(distinct_sum(S, (0, 0), 'RR'))}


def two_particle_gap(S_AB, h):
    """Per-event <2> between two subevents (POI role = A, REF role = B):
    numerator and pair count over distinct particles."""
    return distinct_sum(S_AB, (h, -h), 'PR'), np.real(distinct_sum(S_AB, (0, 0), 'PR'))


def _avg(num, den, w):
    """<<k>> = sum_e w N_e / sum_e w D_e along the event axis (0)."""
    d = np.tensordot(w, den, axes=(0, 0))
    n = np.tensordot(w, np.real(num), axes=(0, 0))
    return np.where(np.abs(d) > 1e-300, n / np.where(np.abs(d) > 1e-300, d, 1.0), np.nan)


def vn_poi(arrays, w, order):
    """v'_n{order} for harmonics 1..N-1 from correlator_arrays output.

    Arrays may carry a trailing bin axis (N: (n_events, n_harm-1, n_bins),
    D: (n_events, n_bins)); the result is then (n_harm-1, n_bins)."""
    w = np.asarray(w, dtype=float)
    d2 = _avg(arrays['N2p'], arrays['D2p'], w)
    c2 = _avg(arrays['N2'], arrays['D2'], w)
    if arrays['N2p'].ndim == 3:            # binned POI, integrated reference
        c2 = c2[:, None]
    if order == 2:
        root = np.sqrt(np.abs(c2))
        # no reference pairs (c2 undefined): NaN, not 0
        return np.where(np.isnan(root), np.nan,
                        np.where(root > 0, d2 * np.sign(c2) / np.where(root > 0, root, 1.0), 0.0))
    four_p = _avg(arrays['N4p'], arrays['D4p'], w)
    four = _avg(arrays['N4'], arrays['D4'], w)
    if arrays['N4p'].ndim == 3:
        four = four[:, None]
    d4 = four_p - 2.0 * d2 * c2
    c4 = four - 2.0 * c2 ** 2
    # -d4 / v{4}^3 with the signed v{4} = sign(-c4) |c4|^(1/4) (until
    # November 2026 the sign of -c4 was dropped: c4 > 0 gave a negative v{4}
    # but an unflipped v'{4})
    den = np.sign(-c4) * np.abs(c4) ** 0.75
    return np.where(den != 0, -d4 / np.where(den != 0, den, 1.0), 0.0)


# ── backends ────────────────────────────────────────────────────────────────

class Container2D:
    """Single-particle sums on a boost-invariant QVectorData.

    poi / ref: dicts with 'idx' (species-axis indices), 'pt' (lo, hi) slice
    indices and 'jac' (n_species, n_pt) window Jacobian.  A particle's weight
    is jac / (2 rap_max nsamples); POI and REF share particles on the common
    species and pT bins (every particle is in every window, with its weight).
    """

    def __init__(self, qv, poi, ref):
        self.qv, self.poi, self.ref = qv, poi, ref
        self.ns = np.asarray(qv.nsamples, dtype=float)
        self.cache = {}

    def __call__(self, h, nP, nR):
        key = (abs(h), nP, nR)
        if key not in self.cache:
            self.cache[key] = self._sum(abs(h), nP, nR)
        v = self.cache[key]
        return np.conj(v) if h < 0 else v

    def _sum(self, h, nP, nR):
        qv = self.qv
        n_ev = qv.Qn.shape[0]
        if h >= qv.Qn.shape[2]:
            return np.zeros(n_ev, complex)       # harmonic not stored (cf. _core_vn4_single)
        sets = ([self.poi] if nP else []) + ([self.ref] if nR else [])
        idx = sorted(set(sets[0]['idx']).intersection(*[s['idx'] for s in sets[1:]]))
        lo = max(s['pt'][0] for s in sets)
        hi = min(s['pt'][1] for s in sets)
        if not idx or hi <= lo:
            return np.zeros(n_ev, complex)
        wgt = np.ones((len(idx), hi - lo))
        for s, k in ((self.poi, nP), (self.ref, nR)):
            if k:
                wgt = wgt * (s['jac'][idx, lo:hi] / (2.0 * qv.rap_max)) ** k
        raw = qv.Qn[:, idx, h, lo:hi].astype(np.complex128)
        return np.einsum('esp,sp->e', raw, wgt) / self.ns ** (nP + nR)


class Container3D:
    """Single-particle sums on unit-weight QGridData (one per sample).

    poi_grid / ref_grid: species-selected grids; windows (eta_lo, eta_hi,
    pt_lo, pt_hi); overlap: whether POI particles are reference particles
    (POI species within the reference species).  Unit weights: every power
    of the weight is 1."""

    def __init__(self, poi_grid, poi_win, ref_grid, ref_win, overlap):
        if getattr(ref_grid, 'rapidity_binned', False):
            # the POI of a rapidity-binned grid has fractional weights
            # (poi_acceptance); a reference needs unit weights
            raise ValueError("a rapidity-binned grid cannot be the reference sample")
        self.pg, self.pw, self.rg, self.rw, self.overlap = poi_grid, poi_win, ref_grid, ref_win, overlap
        self.cache = {}

    def __call__(self, h, nP, nR):
        key = (abs(h), bool(nP), bool(nR))
        if key not in self.cache:
            self.cache[key] = self._sum(*key)
        v = self.cache[key]
        return np.conj(v) if h < 0 else v

    def _sum(self, h, P, R):
        grid = self.pg if P else self.rg
        n_ev = grid.n_events
        if h >= grid.Qn.shape[-1]:
            return np.zeros(n_ev, complex)
        if P and R:
            if not self.overlap:
                return np.zeros(n_ev, complex)
            # POI particles inside the reference window (QGridData.shared:
            # common cells on eta grids, the expected count on y grids)
            return grid.shared(self.pw, self.rw)[:, h]
        if P:
            return grid.poi_acceptance(*self.pw)[:, h]
        return grid.acceptance(*self.rw)[:, h]
