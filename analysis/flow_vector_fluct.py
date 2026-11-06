"""pT-dependent flow-vector fluctuations (ALICE, PRC 109 (2024) 065202, arXiv:2403.15213).

The flow vector V_n(pT) of particles at pT need not be the pT-integrated
V_n times a factor: its angle and magnitude may fluctuate with pT.  Over
distinct particles (analysis.flow_pt.distinct), with the POI in a narrow pT
bin and reference particles over the record's flow_reference pT:

  v_ratio   v_n{2}/v_n[2] = <<cos n(phi1^POI - phi2)>> / sqrt(<<cos n(phi1
            - phi2)>> <<cos n(phi1^POI - phi2^POI)>>)              (Eqs. 5, 7, 8)
  r         r_n = V(a, t) / sqrt(V(a, a) V(t, t)), V(x, y) = <<cos n(phi1^x
            - phi2^y)>>, a the POI bin, t the trigger pT range     (Eq. 9)
  A_f       <<cos n(phi1^POI + phi2^POI - phi3 - phi4)>> /
            <<cos n(phi1^POI + phi2 - phi3^POI - phi4)>>           (Eq. 10)
  M_f       [<<cos n(phi1^POI + phi2 - phi3^POI - phi4)>> /
             (<<cos n(phi1^POI - phi3^POI)>> <<cos n(phi2 - phi4)>>)]
            / [<<4>> / <<2>>^2]                                    (Eqs. 13, 14)
  angle_lower_limit      sqrt((A_f + 1) / 2)                       (Eq. 11)
  magnitude_upper_limit  v_ratio / sqrt((A_f + 1) / 2)             (Eq. 12)

Two-particle correlators take particle 1 in one of the two subevents
(`subevents` A, B: ALICE's |Delta eta| > 0.8) and particle 2 in the other;
four-particle correlators (`four_particle_subevents` A, B: no gap) the
positive harmonics in one and the negative in the other.  Both orientations
are pooled; events are weighted by their tuple counts.  One value per POI
pT bin; signed roots.
"""
import numpy as np

from analysis.flow_pt import distinct

QUANTITIES = ('v_ratio', 'r', 'A_f', 'M_f', 'angle_lower_limit', 'magnitude_upper_limit')
_NEEDS = {'v_ratio': ('two',), 'r': ('trigger',), 'A_f': ('four',), 'M_f': ('four',),
          'angle_lower_limit': ('four',), 'magnitude_upper_limit': ('two', 'four')}


def needs(quantity):
    """The correlator families a quantity uses: 'two', 'trigger', 'four'."""
    return _NEEDS[quantity]


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def event_sums(S, n, quantity, n_bins):
    """Per-event numerators and tuple counts.  Windows of S: 'A', 'B' (the
    two-particle subevents, reference pT), 'A#b', 'B#b' (POI bin b), 'A#t',
    'B#t' (trigger pT); 'C', 'D' (the four-particle subevents, reference pT),
    'C#b', 'D#b'."""
    need = needs(quantity)
    out = {}

    def add(key, members):
        num = distinct(S, [(w, h, 0) for w, h in members])
        den = distinct(S, [(w, 0, 0) for w, _ in members])
        out[key] = out.get(key, 0.0) + np.real(num)
        out['W' + key] = out.get('W' + key, 0.0) + np.real(den)

    if 'two' in need or 'trigger' in need:
        for X, Y in (('A', 'B'), ('B', 'A')):
            if 'two' in need:
                add('ref2', [(X, n), (Y, -n)])
            if 'trigger' in need:
                add('tt2', [(f'{X}#t', n), (f'{Y}#t', -n)])
            for b in range(n_bins):
                add(f'pp2_{b}', [(f'{X}#{b}', n), (f'{Y}#{b}', -n)])
                if 'two' in need:
                    add(f'pr2_{b}', [(f'{X}#{b}', n), (Y, -n)])
                if 'trigger' in need:
                    add(f'pt2_{b}', [(f'{X}#{b}', n), (f'{Y}#t', -n)])
    if 'four' in need:
        for U, V in (('C', 'D'), ('D', 'C')):
            add('ref2cd', [(U, n), (V, -n)])
            add('ref4', [(U, n), (U, n), (V, -n), (V, -n)])
            for b in range(n_bins):
                add(f'num4_{b}', [(f'{U}#{b}', n), (f'{U}#{b}', n), (V, -n), (V, -n)])
                add(f'den4_{b}', [(f'{U}#{b}', n), (U, n), (f'{V}#{b}', -n), (V, -n)])
                add(f'pp2cd_{b}', [(f'{U}#{b}', n), (f'{V}#{b}', -n)])
    return out


def _avg(num, den, w):
    d = np.sum(w * den)
    return np.sum(w * num) / d if d > 0 else np.nan


def core(d, w):
    """One centrality class: the record's quantity per POI pT bin."""
    q = d['quantity']
    need = needs(q)

    def a(key):
        return _avg(d[key], d['W' + key], w)
    out = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for b in range(d['n_bins']):
            if q == 'r':
                out.append(a(f'pt2_{b}') / _sroot(a(f'pp2_{b}') * a('tt2')))
                continue
            if 'two' in need:
                v_ratio = a(f'pr2_{b}') / _sroot(a('ref2') * a(f'pp2_{b}'))
                if q == 'v_ratio':
                    out.append(v_ratio)
                    continue
            af = a(f'num4_{b}') / a(f'den4_{b}')
            if q == 'A_f':
                out.append(af)
            elif q == 'M_f':
                c2 = a('ref2cd')
                out.append((a(f'den4_{b}') / (a(f'pp2cd_{b}') * c2)) / (a('ref4') / c2 ** 2))
            elif q == 'angle_lower_limit':
                out.append(_sroot((af + 1.0) / 2.0))
            else:
                out.append(v_ratio / _sroot((af + 1.0) / 2.0))
    return np.array(out, dtype=float)
