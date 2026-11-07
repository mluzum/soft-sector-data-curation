"""v_n{2}(eta) with side-dependent reference regions, and r_{n|n}(eta), through the production path.

Toy events whose flow angle turns with eta, Psi_2(eta) = Psi_2 + delta_e
eta (ToyGenConfig.plane_eta_twist), on 3D grids (windows on cell edges):
  - r_{2|2}(eta) and the partner-normalized v_2{2}(eta) (reference on the
    other side, divided by the reference-partner correlation) equal their
    expectation from the generated flow vectors (the same correlators with
    each e^{i n phi} replaced by v_n e^{i n Psi_n(eta)}), and r_{2|2} falls
    below 1 with |eta|;
  - without the twist r_{2|2} = 1 and v_2{2}(eta) = sqrt(<v_2^2>), 2D and 3D;
  - a bin without a reference segment is refused.
Run: python3 tests/test_toy_mc_eta_regions.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import resolve  # noqa: E402
from analysis.toy_mc import ToyGenConfig, generate_events, make_qgrid_data, make_qvector_data, evaluate  # noqa: E402

V2 = 0.07
BINS = [[-2.4, -2.0], [-0.6, -0.4], [0.0, 0.2], [0.4, 0.6], [2.0, 2.4], [3.0, 3.4]]
ABS_BINS = [[0.0, 0.4], [0.4, 0.8], [0.8, 1.2]]
REF = (2.6, 3.0)
SEGMENTS = [{'applies_to_eta': {'min': 0.0, 'max': 5.0}, 'eta': [{'min': -1.0, 'max': -0.8}],
             'pT': {'min': 0.0, 'max': 100.0}, 'partner': [{'min': 0.8, 'max': 1.0}]},
            {'applies_to_eta': {'min': -5.0, 'max': 0.0}, 'eta': [{'min': 0.8, 'max': 1.0}],
             'pT': {'min': 0.0, 'max': 100.0}, 'partner': [{'min': -1.0, 'max': -0.8}]}]
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx(twist):
    if twist not in _CACHE:
        cfg = ToyGenConfig(n_events=2000, mult_mean=6000, mult_sigma=200, seed=2307, rap_max=4.0, pt_min=0.05,
                           pt_max=3.0, flow={2: dict(kind='const', value=V2)}, fluctuate={2: 0.02},
                           plane_eta_twist={2: twist} if twist else None)
        events, truth = generate_events(cfg)
        _CACHE[twist] = {'events': events, 'truth': truth,
                         '3d': make_qgrid_data(events, truth, n_eta=50, n_pt=30, n_harm=5, eta_half=5.0),
                         '2d': make_qvector_data(events, truth, n_pt=30, n_harm=5)}
    return _CACHE[twist]


def rec_vn():
    return {'observable_type': 'flow_vn', 'method': 'cumulant', 'cumulant_order': 2, 'harmonic': 2,
            'value_type': 'v2', 'hepdata_file': 'toy.yaml', 'inspire_id': '0', 'version': 1,
            'particle_id': {'species': 'charged'}, 'dimensionality': 'differential_eta',
            'kinematic_range': {'eta': {'min': -2.4, 'max': 3.4, 'bin_ranges': BINS}, 'pT': {'min': 0.0, 'max': 100.0}},
            'flow_reference': {'segments': SEGMENTS},
            'centrality': {'classes': ['0-100%'], 'eta': [{'min': -4.0, 'max': 4.0}]}}


def rec_rnn():
    return {'observable_type': 'flow_harmonic_correlations', 'correlation_type': 'longitudinal_decorrelation',
            'quantity': 'r_nn', 'harmonics': [2], 'value_type': 'r_2|2', 'hepdata_file': 'toy.yaml',
            'inspire_id': '0', 'version': 1, 'particle_id': {'species': 'charged'},
            'dimensionality': 'differential_eta',
            'kinematic_range': {'eta': {'min': 0.0, 'max': 1.2, 'bin_ranges': ABS_BINS},
                                'pT': {'min': 0.0, 'max': 100.0}},
            'decorrelation_reference': {'eta': {'min': REF[0], 'max': REF[1]}, 'pT': {'min': 0.0, 'max': 100.0}},
            'centrality': {'classes': ['0-100%'], 'eta': [{'min': -4.0, 'max': 4.0}]}}


def run(r, kind, twist, jk=True):
    c = _fx(twist)[kind]
    return evaluate(r, qv=c, cent_method='simple', with_jackknife=jk) if kind == '2d' else \
        evaluate(r, grid=c, cent_method='simple', with_jackknife=jk)


def _expected(fx, which, n=2):
    tr = fx['truth']
    acc = {}

    def add(k, num, cnt):
        acc[k] = acc.get(k, 0.0) + num
        acc['W' + k] = acc.get('W' + k, 0.0) + cnt
    for e, ev in enumerate(fx['events']):
        eta = ev['eta']
        v = tr['vn_ebe'][n][e]
        psi = tr['psi'][n][e] + (tr['plane_eta_twist'][n][e] * eta if n in tr['plane_eta_twist'] else 0.0)
        u = v * np.exp(1j * n * psi)

        def s(lo, hi):
            m = (eta >= lo) & (eta < hi)
            return np.sum(u[m]), m.sum()

        def two(a, b):    # disjoint windows
            return np.real(a[0] * np.conj(b[0])), a[1] * b[1]
        if which == 'rnn':
            yn, yp = s(-REF[1], -REF[0]), s(*REF)
            for b, (lo, hi) in enumerate(ABS_BINS):
                pp, pm = s(lo, hi), s(-hi, -lo)
                o = [two(pp, yn), two(pm, yp)]
                q = [two(pm, yn), two(pp, yp)]
                add(f'o{b}', o[0][0] + o[1][0], o[0][1] + o[1][1])
                add(f's{b}', q[0][0] + q[1][0], q[0][1] + q[1][1])
        else:
            pos, neg = s(0.8, 1.0), s(-1.0, -0.8)
            for b, (lo, hi) in enumerate(BINS):
                p = s(lo, hi)
                y, x = (neg, pos) if lo >= 0 else (pos, neg)
                add(f'n{b}', *two(p, y))
                add(f'x{b}', *two(x, y))

    def a(k):
        return acc[k] / acc['W' + k]
    if which == 'rnn':
        return np.array([a(f'o{b}') / a(f's{b}') for b in range(len(ABS_BINS))])
    return np.array([a(f'n{b}') / np.sqrt(a(f'x{b}')) for b in range(len(BINS))])


def test_dispatch():
    check('v2{2}(eta) with partner regions dispatches to vn2_eta_partner',
          resolve(rec_vn())['compute'] == 'vn2_eta_partner')
    check('r_nn dispatches to rnn_decorrelation', resolve(rec_rnn())['compute'] == 'rnn_decorrelation')
    r = rec_vn()
    r['flow_reference'] = {'segments': SEGMENTS[:1]}
    try:
        run(r, '3d', 0.0, jk=False)
        ok = False
    except ValueError:
        ok = True
    check('a bin without a reference segment is refused', ok)


def test_expectation_3d():
    fx = _fx(0.15)
    for name, r, which in (('r_2|2(eta)', rec_rnn(), 'rnn'), ('v2{2}(eta), partner-normalized', rec_vn(), 'vn')):
        v, e = run(r, '3d', 0.15)
        want = _expected(fx, which)
        check(f'3D {name} = expectation from the generated flow vectors', np.all(np.abs(v - want) < 3 * e + 0.003),
              f'{v.round(4)} +- {e.round(4)} vs {want.round(4)}')
    v, e = run(rec_rnn(), '3d', 0.15)
    check('3D r_2|2 falls with |eta|', v[-1] < v[0] and v[-1] < 1 - 3 * e[-1], f'{v.round(4)}')


def test_no_twist():
    for kind in ('3d', '2d'):
        v, e = run(rec_rnn(), kind, 0.0)
        check(f'{kind}: r_2|2 = 1 without decorrelation', np.all(np.abs(v - 1) < 3 * e + 0.01), f'{v.round(4)}')
        v, e = run(rec_vn(), kind, 0.0)
        rms = np.sqrt(np.mean(_fx(0.0)['truth']['vn_ebe'][2] ** 2))
        check(f'{kind}: v2{{2}}(eta) = sqrt(<v2^2>) = {rms:.4f} without decorrelation',
              np.all(np.abs(v - rms) < 3 * e + 0.01 * rms), f'{v.round(4)} +- {e.round(4)}')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    n_ok = 0
    for t in tests:
        print(t.__name__)
        try:
            t()
            n_ok += 1
        except AssertionError as e:
            print(f'  FAILED: {e}')
    print(f'\n{n_ok}/{len(tests)} passed')
    sys.exit(0 if n_ok == len(tests) else 1)
