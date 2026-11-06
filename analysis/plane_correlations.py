"""Symmetry-plane correlations in the scalar-product (SP) method (ATLAS PRC 90 024905).

A correlator <cos(sum_i c_i n_i Phi_{n_i})> (sum c_i n_i = 0) is measured from
flow vectors of subevents, each plane i from its own subevent, weighted by
the flow magnitudes (SP method, Eq. 10):

  numerator   <Re prod_i q_{n_i}^{c_i}>      (q^{c} for c < 0: conj(q)^{|c|})
  resolution  Res{|c| n Psi_n} = sqrt(<Re q_X^{|c|} conj(q_Y)^{|c|}>)   (two
              symmetric subevents, Eq. 16) or, for a subevent X with two
              others Y, Z, sqrt(<q_X q_Y*><q_X q_Z*> / <q_Y q_Z*>) (Eq. 17);
  correlator  numerator / prod_i Res_i.

Two planes: subevents P and N symmetric about eta = 0, both assignments
averaged (Eq. 12), the 2SE resolution.  Three planes: subevents A and C
symmetric, B in between or forward; Type t puts plane t in B and the other
two in A and C (both assignments averaged), plane t's resolution from the
3SE formula, the others from 2SE(A, C); the three types averaged (ATLAS
combines them statistically).  Large multiplicity: q_n -> v_n e^{i n Psi_n},
e.g. <cos 4(Phi_2 - Phi_4)> -> <v_2^2 v_4 cos 4(Psi_2 - Psi_4)> /
sqrt(<v_2^4><v_4^2>).

Powers of flow vectors are sums over distinct particles (q^c: distinct
c-tuples of the subevent, per event over their number), computed exactly for
any number of members by the generating function prod_i (1 + sum_t s_t
z_{i,t}) (analysis.flow_pt.distinct_gf), so that coincidences vanish
between disjoint 3D windows and are removed on 2D containers, where every
particle has a share in every window.  Events are weighted equally (each q normalized per event).
"""
import numpy as np

from analysis.flow_pt import distinct_gf


def _norm_product(S, members):
    """Per event: Re( distinct sum of the members / their tuple count ), and
    the tuple count; members [(window, h, count)]."""
    num = distinct_gf(S, members)
    den = np.real(distinct_gf(S, [(w, 0, c) for w, _, c in members]))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(den > 0, np.real(num) / np.where(den > 0, den, 1.0), np.nan)
    return r


def _members(assign, planes):
    """assign: subevent per plane; planes: [(n, c)] -> [(window, sign(c) n, |c|)]."""
    return [(w, n if c > 0 else -n, abs(c)) for w, (n, c) in zip(assign, planes)]


def event_terms(S, planes, layout):
    """Per-event normalized products for the core; planes [(n, c)];
    layout 'two' (subevents P, N) or 'three' (A, B, C)."""
    out = {}
    if layout == 'two':
        assert len(planes) == 2
        out['num'] = [_norm_product(S, _members(('P', 'N'), planes)),
                      _norm_product(S, _members(('N', 'P'), planes))]
        out['res'] = [_norm_product(S, [('P', n, abs(c)), ('N', -n, abs(c))]) for n, c in planes]
        return out
    assert layout == 'three' and len(planes) == 3
    for t in range(3):
        others = [i for i in range(3) if i != t]
        nums = []
        for xa, xc in (('A', 'C'), ('C', 'A')):
            assign = [None] * 3
            assign[t] = 'B'
            assign[others[0]], assign[others[1]] = xa, xc
            nums.append(_norm_product(S, _members(assign, planes)))
        out[f'num{t}'] = nums
    for i, (n, c) in enumerate(planes):
        j = abs(c)
        out[f'AC{i}'] = _norm_product(S, [('A', n, j), ('C', -n, j)])
        out[f'BA{i}'] = _norm_product(S, [('B', n, j), ('A', -n, j)])
        out[f'BC{i}'] = _norm_product(S, [('B', n, j), ('C', -n, j)])
    return out


def _mean(x, w):
    ok = np.isfinite(x)
    ws = np.sum(w[ok])
    return np.sum(w[ok] * x[ok]) / ws if ws > 0 else np.nan


def _sroot(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def core(d, w):
    """One centrality class: the correlator (length-1 array)."""
    planes, layout = d['planes'], d['layout']
    with np.errstate(divide='ignore', invalid='ignore'):
        if layout == 'two':
            num = np.mean([_mean(x, w) for x in d['num']])
            res = np.prod([_sroot(_mean(x, w)) for x in d['res']])
            return np.array([num / res])
        vals = []
        for t in range(3):
            num = np.mean([_mean(x, w) for x in d[f'num{t}']])
            res = 1.0
            for i in range(3):
                ac = _mean(d[f'AC{i}'], w)
                if i == t:   # 3SE: plane i measured in B
                    res = res * _sroot(_mean(d[f'BA{i}'], w) * _mean(d[f'BC{i}'], w) / ac)
                else:        # 2SE between the symmetric A and C
                    res = res * _sroot(ac)
            vals.append(num / res)
        return np.array([np.mean(vals)])
