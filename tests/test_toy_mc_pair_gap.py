"""v_n{2} with a pair gap (eta_gap without subevents) on 3D containers.

The two-particle correlation functions of ATLAS and CMS (2 < |Delta eta| < 5)
and STAR's Delta eta > 0.05 and > 1 correlate pairs beyond a gap.  On 3D
containers the pairs of eta cells i, j are weighted by the fraction of their
particle pairs beyond the gap (QGridData.pair_gap_fractions, exact for
uniform cells), self-pairs removed.  Checks:
  - the fractions equal a Monte Carlo of uniform pairs;
  - gap -> 0+ equals the distinct-pair v_n{2} without a gap (integrated and
    vs pT) to machine precision;
  - short-range non-flow (collinear clones within |Delta eta| ~ 0.05) biases
    v2{2} without a gap; with |Delta eta| > 1 the input v2 is recovered, and
    the result equals a particle-level pair-gap estimator within errors;
  - a gap wider than the acceptance leaves no pairs: NaN.
Run: python3 tests/test_toy_mc_pair_gap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.qvector import QGridData  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, evaluate  # noqa: E402

V2 = 0.06
results = []
_CACHE = {}


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")


def fx():
    """Events with v2 = 0.06 and, in a copy, collinear clones of half of the particles."""
    if not _CACHE:
        cfg = ToyGenConfig(n_events=600, mult_mean=1500, mult_sigma=150, seed=41, rap_max=2.5, pt_min=0.2, pt_max=3.0,
                           flow={2: dict(kind='const', value=V2), 3: dict(kind='const', value=0.02)})
        ev, tr = generate_events(cfg)
        rng = np.random.default_rng(7)
        nf = []
        for e in ev:
            k = rng.random(len(e['pt'])) < 0.5
            c = {key: np.asarray(v)[k] for key, v in e.items()}
            c['eta'] = c['eta'] + rng.normal(0, 0.05, k.sum())
            c['y'] = c['y'] + (c['eta'] - np.asarray(e['eta'])[k])
            c['phi'] = (c['phi'] + rng.normal(0, 0.05, k.sum())) % (2 * np.pi)
            nf.append({key: np.concatenate([np.asarray(e[key]), c[key]]) for key in e})
        _CACHE['events'] = nf
        _CACHE['flow'] = make_qgrid_data(ev, tr, n_eta=40, n_pt=14, n_harm=4, eta_half=3.0)
        _CACHE['nonflow'] = make_qgrid_data(nf, tr, n_eta=40, n_pt=14, n_harm=4, eta_half=3.0)
    return _CACHE


def rec(gap=None, diff=False):
    r = {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': 2, 'harmonic': 2, 'value_type': 'v2',
         'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'charged'},
         'centrality': {'classes': ['0-100%'], 'eta': [{'min': -3.0, 'max': -2.4}, {'min': 2.4, 'max': 3.0}]},
         'kinematic_range': {'eta': {'min': -2.1, 'max': 2.1}, 'pT': {'min': 0.2, 'max': 3.0}},
         'dimensionality': 'integrated'}
    if diff:
        r.update(dimensionality='differential_pT', flow_reference={'pT': {'min': 0.2, 'max': 3.0}})
        r['kinematic_range']['pT'] = {'min': 0.2, 'max': 3.0, 'bins': [0.2, 0.6, 1.0, 1.6, 3.0]}
    if gap is not None:
        r['eta_gap'] = gap
    return r


def run(r, grid, jk=True):
    v, e = evaluate(r, grid=grid, cent_method='simple', with_jackknife=jk)
    return np.atleast_1d(v), np.atleast_1d(e) if jk else None


def test_fractions():
    g = QGridData(Qn=np.zeros((1, 8, 2, 2), complex), eta_edges=np.array([-1.0, -0.8, -0.5, -0.4, 0.0, 0.3, 0.4, 0.8, 1.0]),
                  pt_edges=np.array([0.0, 1.0, 2.0]), cent_mult=np.ones(1), nsamples=np.ones(1))
    rng = np.random.default_rng(0)
    e = g.eta_edges
    worst = 0.0
    for gap in (0.05, 0.3, 0.7):
        F = g.pair_gap_fractions(gap)
        for i, j in ((0, 0), (0, 1), (1, 3), (2, 5), (3, 3), (4, 6)):
            x = rng.uniform(e[i], e[i + 1], 200000)
            y = rng.uniform(e[j], e[j + 1], 200000)
            worst = max(worst, abs(F[i, j] - np.mean(np.abs(x - y) > gap)))
    check(f'pair fractions = Monte Carlo (unequal cells; worst {worst:.4f})', worst < 0.005)


def test_zero_gap_limit():
    g = fx()['flow']
    for diff in (False, True):
        v0, _ = run(rec(None, diff), g, jk=False)
        v1, _ = run(rec(1e-13, diff), g, jk=False)
        check(f"gap -> 0+ = no gap ({'vs pT' if diff else 'integrated'})", np.allclose(v0, v1, rtol=1e-10, atol=0),
              f'{np.round(v0, 5)} vs {np.round(v1, 5)}')


def particle_level(events, gap, n=2):
    num = den = 0.0
    for e in events:
        s = (np.abs(e['eta']) < 2.1) & (e['pt'] > 0.2) & (e['pt'] < 3.0)
        eta, phi = e['eta'][s], e['phi'][s]
        o = np.argsort(eta)
        eta, z = eta[o], np.exp(1j * n * phi[o])
        # pairs with |Delta eta| > gap: for each a, partners beyond eta_a + gap (both orders: x2)
        cz = np.concatenate([[0], np.cumsum(z[::-1])])[::-1]          # suffix sums
        j = np.searchsorted(eta, eta + gap, side='right')
        num += 2 * np.real(np.sum(z * np.conj(cz[j])))
        den += 2 * np.sum(len(eta) - j)
    return np.sqrt(num / den)


def test_nonflow():
    """Same events with and without the clones: the clones add pairs within
    |Delta eta| ~ 0.05 (non-flow) and otherwise particles with the same flow."""
    c = fx()
    v_flow, e_flow = run(rec(None), c['flow'])
    v_nogap, e_nogap = run(rec(None), c['nonflow'])
    v_gapf, e_gapf = run(rec(1.0), c['flow'])
    v_gap, e_gap = run(rec(1.0), c['nonflow'])
    check(f'non-flow biases v2{{2}} without a gap ({v_nogap[0]:.4f} with clones vs {v_flow[0]:.4f} +- {e_flow[0]:.4f})',
          v_nogap[0] - v_flow[0] > 3 * e_flow[0])
    check(f'|Delta eta| > 1 removes it ({v_gap[0]:.4f} with clones vs {v_gapf[0]:.4f} +- {e_gapf[0]:.4f} without)',
          abs(v_gap[0] - v_gapf[0]) < 1.5 * e_gapf[0])
    check(f'|Delta eta| > 1: input {V2} within errors ({v_gapf[0]:.4f} +- {e_gapf[0]:.4f})', abs(v_gapf[0] - V2) < 4 * e_gapf[0])
    pl = particle_level(c['events'], 1.0)
    check(f'= particle-level pair-gap estimator within errors ({v_gap[0]:.4f} vs {pl:.4f})', abs(v_gap[0] - pl) < 2 * e_gap[0])
    vd, ed = run(rec(1.0, diff=True), c['nonflow'])
    check(f'vs pT with |Delta eta| > 1: input recovered ({np.round(vd, 4)})', np.all(np.abs(vd - V2) < 4 * ed))


def test_no_pairs():
    v, _ = run(rec(5.0), fx()['flow'], jk=False)
    check('a gap wider than the acceptance: NaN', np.all(np.isnan(v)), f'{v}')


if __name__ == '__main__':
    for t in (test_fractions, test_zero_gap_limit, test_nonflow, test_no_pairs):
        print(t.__name__)
        t()
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
