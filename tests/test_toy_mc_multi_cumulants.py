"""Multi-particle cumulants with the standard and three-subevent methods
(analysis.multi_cumulants, key multi_cumulant; ATLAS JHEP 01 (2020) 051).

Toy: v2 and v3 of each event drawn from two-dimensional Gaussians around
(v0, 0) with width delta (Bessel-Gaussian p(v_n)), v4 = chi v2^2 aligned
with Psi_2 plus a fluctuating part, particles uniform in |eta| < 2.5 (no
non-flow).  Truth: nc_n{4} = -v0^4 / (v0^2 + delta^2)^2, nc_n{6} = v0^6 /
(v0^2 + delta^2)^3; sc and ac from the generated flow vectors.

Checks:
  - 3D, exact against closed-form Q-vector sums of the particles in the
    windows (the grid's subevents on cell edges): standard c_2{4}, sc_{2,3},
    ac_2{3}; three-subevent c_2{4}, sc_{2,3}, ac_2{3} (each the mean of the
    three interchanged measurements); standard v_2{6} equal to the
    vn6_single core and sc to sc_mn;
  - physics, 2D and 3D: nc_2{4}, nc_2{6}, v_2{4}/v_2{2}, v_2{6}/v_2{4},
    nsc_{2,3}, nac_2{3} of both methods recover the generated values;
  - narrow reference classes (multiplicity_baseline): exact against an
    explicit per-interval computation when the interval averages are linear
    in the multiplicity (random tuple counts and residuals); toy whose
    <v2^2> rises with the multiplicity: the baseline recovers the fixed-
    multiplicity nc_2{4}, the class average is shifted by the spread of
    <v2^2>;
  - routing (v_2{4} / v_2{2} is not refused as a ratio of coefficients),
    the toy harness end to end with the jackknife, and refusals.
Run: python3 tests/test_toy_mc_multi_cumulants.py
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data, particles_to_qvector_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, SPECIES_NAMES  # noqa: E402
from analysis import multi_cumulants as mc  # noqa: E402

NE, M_MEAN = 4000, 600
V0, DELTA, V30, D3 = 0.06, 0.035, 0.0, 0.03
SUBS = {'A': (-2.5, -2.5 / 3), 'B': (-2.5 / 3, 2.5 / 3), 'C': (2.5 / 3, 2.5)}
FULL = (-2.5, 2.5)
PT = (0.2, 5.0)
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _gen(seed=11, ne=NE, mult_trend=0.0):
    """Particle-level toy.  mult_trend: <v2^2> grows with the event's
    multiplicity, v0^2 -> v0^2 (1 + mult_trend (M - M_MEAN) / M_MEAN)."""
    rng = np.random.default_rng(seed)
    events, V = [], []
    for _ in range(ne):
        m = int(rng.integers(M_MEAN // 2, 3 * M_MEAN // 2)) if mult_trend else rng.poisson(M_MEAN)
        v0 = V0 * np.sqrt(max(1.0 + mult_trend * (m - M_MEAN) / M_MEAN, 0.05))
        V2 = v0 + rng.normal(0, DELTA / np.sqrt(2)) + 1j * rng.normal(0, DELTA / np.sqrt(2))
        V3 = V30 + rng.normal(0, D3 / np.sqrt(2)) + 1j * rng.normal(0, D3 / np.sqrt(2))
        V3 *= 1.0 + 4.0 * (abs(V2) - V0)          # |v3| correlated with |v2|: sc > 0
        V4 = 5.0 * V2 ** 2 + rng.normal(0, 0.008) + 1j * rng.normal(0, 0.008)
        psi = rng.uniform(0, 2 * np.pi)
        harm = {2: V2 * np.exp(2j * psi), 3: V3 * np.exp(3j * rng.uniform(0, 2 * np.pi)), 4: V4 * np.exp(4j * psi)}
        vmax = 1 + 2 * sum(abs(v) for v in harm.values())
        phi = []
        while len(phi) < m:
            c = rng.uniform(0, 2 * np.pi, 2 * m)
            f = 1 + 2 * sum(np.real(np.conj(v) * np.exp(1j * h * c)) for h, v in harm.items())
            phi.extend(c[rng.uniform(0, vmax, c.size) < f].tolist())
        phi = np.array(phi[:m])
        events.append({'phi': phi, 'eta': rng.uniform(-2.5, 2.5, m), 'pt': rng.exponential(0.5, m) + 0.2,
                       'species_idx': np.zeros(m, int), 'mass': np.full(m, SPECIES_MASSES[0])})
        V.append([harm[2], harm[3], harm[4]])
    return events, np.array(V)


def _fixture(key='base', **kw):
    if key in _CACHE:
        return _CACHE[key]
    events, V = _gen(**kw)
    eta_edges = np.round(np.linspace(-3, 3, 37), 10)          # cells of 1/6: subevents on edges
    pt_edges = np.round(np.linspace(0.0, 10.0, 101), 10)
    mult = np.array([len(e['phi']) for e in events], float)
    grid = particles_to_qgrid_data(events, eta_edges, pt_edges, n_harm=9, cent_mult=mult, masses=SPECIES_MASSES)
    qv = particles_to_qvector_data(events, pt_edges, rap_max=2.5, species_names=list(SPECIES_NAMES),
                                   masses=SPECIES_MASSES, n_harm=9)
    _CACHE[key] = dict(events=events, V=V, grid=grid, qv=qv, mult=mult)
    return _CACHE[key]


def _kr(w):
    return {'eta': {'min': w[0], 'max': w[1]}, 'pT': {'min': PT[0], 'max': PT[1]}}


def _subs(labels):
    return {lab: {'kinematic_range': _kr(SUBS[lab])} for lab in labels}


def record_vn(quantity, order=4, method='standard', ratio_order=None, n=2, baseline=False):
    r = dict(observable_type='flow_vn', inspire_id='0', version=1, hepdata_file='t.yaml', value_type='x',
             collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             centrality={'classes': ['0-100%'], 'eta': [{'min': 3.2, 'max': 4.9}]},
             particle_id={'species': 'charged'}, kinematic_range=_kr(FULL), harmonic=n, method='cumulant',
             cumulant_order=order, cumulant_method=method, cumulant_quantity=quantity, dimensionality='integrated')
    if ratio_order is not None:
        r['ratio_order'] = ratio_order
    if method == 'three_subevent':
        r.update(particles='AABC', subevents=_subs('ABC'))
    elif quantity == 'normalized' or ratio_order == 2:
        r['normalization_subevents'] = _subs('AC')
    if baseline:
        r['multiplicity_baseline'] = {'eta': [{'min': FULL[0], 'max': FULL[1]}], 'pT': {'min': PT[0], 'max': PT[1]}}
    return r


def record_corr(ctype, harmonics, normalized, method='standard'):
    r = dict(observable_type='flow_harmonic_correlations', inspire_id='0', version=1, hepdata_file='t.yaml',
             value_type='x', observable='x', correlation_type=ctype, harmonics=list(harmonics),
             collision_system={'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             centrality={'classes': ['0-100%'], 'eta': [{'min': 3.2, 'max': 4.9}]},
             particle_id={'species': 'charged'}, kinematic_range=_kr(FULL), cumulant_method=method,
             normalized=normalized)
    if method == 'three_subevent':
        r.update(particles='ABC' if ctype == 'asymmetric_cumulant' else 'AABC', subevents=_subs('ABC'))
    elif normalized:
        r['normalization_subevents'] = _subs('AC')
    return r


_PREP = {}


def production(rec, dim, w=None, fx=None):
    """The record's value on the fixture (prepared sums cached per record)."""
    import json
    fx = fx or _fixture()
    key = (json.dumps(rec, sort_keys=True), dim, id(fx))
    ent = get_pipeline_entry(resolve(rec)['compute'])
    if key not in _PREP:
        p = extract_measurement_params(rec)
        _PREP[key] = ent['prepare_data_2d'](fx['qv'], p) if dim == '2D' else ent['prepare_data_3d'](fx['grid'], p)
    data = _PREP[key]
    n = len(fx['events'])
    return float(ent['core_fn'](data, np.ones(n) if w is None else w)[0])


def jk_error(fn, n, n_blocks=10):
    vals = []
    for b in np.array_split(np.arange(n), n_blocks):
        w = np.ones(n)
        w[b] = 0
        vals.append(fn(w))
    vals = np.array(vals)
    return np.sqrt((n_blocks - 1) / n_blocks * np.sum((vals - vals.mean()) ** 2))


# ── closed forms from the particles ─────────────────────────────────────────

def _q(e, win, hs):
    s = (e['eta'] >= win[0]) & (e['eta'] < win[1]) & (e['pt'] >= PT[0]) & (e['pt'] < PT[1])
    phi = e['phi'][s]
    out = {h: np.exp(1j * h * phi).sum() for h in hs}
    out['M'] = float(len(phi))
    return out


def closed_forms(events, n=2, m=3):
    acc = {}

    def add(k, num, den):
        a = acc.setdefault(k, [0.0, 0.0])
        a[0] += float(np.real(num))
        a[1] += float(den)
    hs = sorted({n, m, n + m, abs(n - m), 2 * n})
    for e in events:
        q = _q(e, FULL, hs)
        M = q['M']
        Qn, Qm, Q2n, Qnm, Qdm = q[n], q[m], q[2 * n], q[n + m], q[abs(n - m)]
        add('2n', abs(Qn) ** 2 - M, M * (M - 1))
        add('2m', abs(Qm) ** 2 - M, M * (M - 1))
        add('4n', abs(Qn) ** 4 + abs(Q2n) ** 2 - 2 * np.real(Q2n * np.conj(Qn) ** 2) - 4 * (M - 2) * abs(Qn) ** 2
            + 2 * M * (M - 3), M * (M - 1) * (M - 2) * (M - 3))
        add('4nm', abs(Qm) ** 2 * abs(Qn) ** 2 - 2 * np.real(Qnm * np.conj(Qm) * np.conj(Qn))
            - 2 * np.real(Qm * np.conj(Qdm) * np.conj(Qn)) + abs(Qnm) ** 2 + abs(Qdm) ** 2
            - (M - 4) * (abs(Qm) ** 2 + abs(Qn) ** 2) + M * (M - 6), M * (M - 1) * (M - 2) * (M - 3))
        add('3', Qn ** 2 * np.conj(Q2n) - abs(Q2n) ** 2 - 2 * abs(Qn) ** 2 + 2 * M, M * (M - 1) * (M - 2))
        s = {lab: _q(e, w, hs) for lab, w in SUBS.items()}
        for X, Y, Z in mc.TRIPLES:
            x, y, z = s[X], s[Y], s[Z]
            add(f'4{X}', (x[n] ** 2 - x[2 * n]) * np.conj(y[n]) * np.conj(z[n]), x['M'] * (x['M'] - 1) * y['M'] * z['M'])
            add(f'sc{X}', (x[n] * x[m] - x[n + m]) * np.conj(y[n]) * np.conj(z[m]),
                x['M'] * (x['M'] - 1) * y['M'] * z['M'])
        for X, Y, Z in mc.AC_TRIPLES:
            x, y, z = s[X], s[Y], s[Z]
            add(f'ac{Z}', x[n] * y[n] * np.conj(z[2 * n]), x['M'] * y['M'] * z['M'])
        for XY in ('AB', 'AC', 'BC'):
            x, y = s[XY[0]], s[XY[1]]
            add(f'p{XY}n', x[n] * np.conj(y[n]), x['M'] * y['M'])
            add(f'p{XY}m', x[m] * np.conj(y[m]), x['M'] * y['M'])
    k = {key: a[0] / a[1] for key, a in acc.items()}

    def p(X, Y, h):
        return k['p' + ''.join(sorted(X + Y)) + h]
    return {
        'c4': k['4n'] - 2 * k['2n'] ** 2,
        'sc': k['4nm'] - k['2n'] * k['2m'],
        'ac': k['3'],
        'c4_3sub': np.mean([k[f'4{X}'] - 2 * p(X, Y, 'n') * p(X, Z, 'n') for X, Y, Z in mc.TRIPLES]),
        'sc_3sub': np.mean([k[f'sc{X}'] - p(X, Y, 'n') * p(X, Z, 'm') for X, Y, Z in mc.TRIPLES]),
        'ac_3sub': np.mean([k[f'ac{Z}'] for _X, _Y, Z in mc.AC_TRIPLES]),
    }


def test_exact_3d():
    print('test_exact_3d')
    fx = _fixture()
    cf = closed_forms(fx['events'][:600])
    w = np.zeros(len(fx['events']))
    w[:600] = 1.0
    for name, rec in (('c4', record_vn('cumulant')), ('c4_3sub', record_vn('cumulant', method='three_subevent')),
                      ('sc', record_corr('symmetric_cumulant', [2, 3], False)),
                      ('sc_3sub', record_corr('symmetric_cumulant', [2, 3], False, 'three_subevent')),
                      ('ac', record_corr('asymmetric_cumulant', [2], False)),
                      ('ac_3sub', record_corr('asymmetric_cumulant', [2], False, 'three_subevent'))):
        got = production(rec, '3D', w)
        check(f'3D {name} == closed form', abs(got - cf[name]) < 1e-9 * abs(cf[name]), f'{got:.6e} vs {cf[name]:.6e}')
    # standard v_2{6} and sc against the existing cores
    r6 = record_vn('coefficient', order=6)
    ref6 = dict(r6, cumulant_order=6, value_type='v2{6}')
    for k in ('cumulant_method', 'cumulant_quantity'):
        ref6.pop(k)
    p = extract_measurement_params(ref6)
    ent = get_pipeline_entry(resolve(ref6)['compute'])
    v6_old = float(np.atleast_1d(ent['core_fn'](ent['prepare_data_3d'](fx['grid'], p), np.ones(NE)))[1])
    v6_new = production(r6, '3D')
    check('3D v_2{6} == vn6_single', abs(v6_new - v6_old) < 1e-9 * abs(v6_old), f'{v6_new:.6e} vs {v6_old:.6e}')
    rsc = record_corr('symmetric_cumulant', [2, 3], False)
    old = dict(rsc)
    old.pop('cumulant_method')
    p = extract_measurement_params(old)
    ent = get_pipeline_entry(resolve(old)['compute'])
    sc_old = float(ent['core_fn'](ent['prepare_data_3d'](fx['grid'], p), np.ones(NE))[0])
    sc_new = production(rsc, '3D')
    check('3D sc_{2,3} == sc_mn', abs(sc_new - sc_old) < 1e-9 * abs(sc_old), f'{sc_new:.6e} vs {sc_old:.6e}')


def truths(V):
    v2, v3, v4 = np.abs(V[:, 0]), np.abs(V[:, 1]), np.abs(V[:, 2])
    m2, m4, m6 = np.mean(v2 ** 2), np.mean(v2 ** 4), np.mean(v2 ** 6)
    c4 = m4 - 2 * m2 ** 2
    c6 = m6 - 9 * m4 * m2 + 12 * m2 ** 3
    v4c = np.sign(-c4) * abs(c4) ** 0.25
    v6c = np.sign(c6) * abs(c6 / 4) ** (1 / 6)
    sc = np.mean(v2 ** 2 * v3 ** 2) - m2 * np.mean(v3 ** 2)
    # ac_2{3} = <v2^2 v4 cos 4(Psi2 - Psi4)> = Re <V2^2 V4*> (V_n with their phases)
    ac = np.mean(np.real(V[:, 0] ** 2 * np.conj(V[:, 2])))
    return {'nc4': c4 / m2 ** 2, 'nc6': c6 / (4 * m2 ** 3), 'r42': v4c / np.sqrt(m2), 'r64': v6c / v4c,
            'nsc': sc / (m2 * np.mean(v3 ** 2)), 'nac': ac / np.sqrt((2 * m2 ** 2 + c4) * np.mean(v4 ** 2))}


def test_physics():
    print('test_physics')
    fx = _fixture()
    t = truths(fx['V'])
    cases = [('nc_2{4}', record_vn('normalized'), t['nc4']),
             ('nc_2{4} 3-sub', record_vn('normalized', method='three_subevent'), t['nc4']),
             ('nc_2{6}', record_vn('normalized', order=6), t['nc6']),
             ('v_2{4}/v_2{2}', record_vn('ratio', ratio_order=2), t['r42']),
             ('v_2{6}/v_2{4}', record_vn('ratio', order=6, ratio_order=4), t['r64']),
             ('nsc_{2,3}', record_corr('symmetric_cumulant', [2, 3], True), t['nsc']),
             ('nsc_{2,3} 3-sub', record_corr('symmetric_cumulant', [2, 3], True, 'three_subevent'), t['nsc']),
             ('nac_2{3}', record_corr('asymmetric_cumulant', [2], True), t['nac']),
             ('nac_2{3} 3-sub', record_corr('asymmetric_cumulant', [2], True, 'three_subevent'), t['nac'])]
    for dim in ('3D', '2D'):
        for name, rec, truth in cases:
            if dim == '2D' and '3-sub' in name:
                continue      # 2D: every window holds every particle; covered on 3D
            v = production(rec, dim)
            e = jk_error(lambda w, rec=rec, dim=dim: production(rec, dim, w), NE)
            check(f'{dim} {name} = {truth:.4f}', abs(v - truth) < 3.5 * e + 0.02 * abs(truth), f'{v:.4f} +- {e:.4f}')


def test_narrow_bins_exact():
    """Synthetic sums: three multiplicity intervals whose (tuple-weighted)
    averages of <2> and <4> are exactly linear in M; the regression average
    equals the explicit per-interval cumulants combined by events."""
    print('test_narrow_bins_exact')
    rng = np.random.default_rng(3)
    Ms = np.repeat([100.0, 150.0, 200.0], [40, 25, 35])
    D2 = rng.uniform(0.5, 2.0, Ms.size)
    D4 = rng.uniform(0.5, 2.0, Ms.size)

    def interval_values(D, a, b):
        x = np.empty(Ms.size)
        for M in np.unique(Ms):
            s = Ms == M
            r = rng.normal(0, 0.3 * a, s.sum())
            r -= np.sum(D[s] * r) / np.sum(D[s])          # the interval's weighted mean is a + b M
            x[s] = a + b * M + r
        return x
    x2 = interval_values(D2, 0.004, 1e-5)
    x4 = interval_values(D4, 4e-5, 2e-7)
    d = {'N2': x2 * D2, 'D2': D2, 'N4': x4 * D4, 'D4': D4, 'mult': Ms,
         'settings': {'kind': 'single', 'method': 'standard', 'harmonics': [2], 'order': 4,
                      'quantity': 'cumulant', 'ratio_order': None, 'normalized': False}}
    w = np.ones(Ms.size)
    got = float(mc.core(d, w)[0])
    want, nev = 0.0, 0
    for M in np.unique(Ms):
        s = Ms == M
        k2 = np.sum(x2[s] * D2[s]) / np.sum(D2[s])
        k4 = np.sum(x4[s] * D4[s]) / np.sum(D4[s])
        want += s.sum() * (k4 - 2 * k2 ** 2)
        nev += s.sum()
    want /= nev
    check('narrow intervals: regression == explicit', abs(got - want) < 1e-12 * abs(want), f'{got:.10e} vs {want:.10e}')
    d.pop('mult')
    wide = float(mc.core(d, w)[0])
    check('without the baseline: the class average differs', abs(wide - want) > 1e-3 * abs(want),
          f'{wide:.4e} vs {want:.4e}')


def test_narrow_bins_toy():
    print('test_narrow_bins_toy')
    trend = 1.5
    fx = _fixture('trend', seed=5, ne=4000, mult_trend=trend)
    V, mult = fx['V'], np.array([len(e['phi']) for e in fx['events']], float)
    # truth at fixed multiplicity: c4(M) of the generating distribution, averaged over events
    v2sq_M = (V0 ** 2 * np.maximum(1.0 + trend * (mult - M_MEAN) / M_MEAN, 0.05))
    m2 = v2sq_M + DELTA ** 2
    m4 = v2sq_M ** 2 + 4 * v2sq_M * DELTA ** 2 + 2 * DELTA ** 4
    c4_fixed = np.mean(m4 - 2 * m2 ** 2)
    c2 = np.mean(m2)
    truth_fixed = c4_fixed / c2 ** 2
    # the class average weights events by their tuples (M^4, M^2; A|C pairs ~ M^2)
    w4, w2 = mult * (mult - 1) * (mult - 2) * (mult - 3), mult * (mult - 1)
    k4, k2 = np.sum(w4 * m4) / np.sum(w4), np.sum(w2 * m2) / np.sum(w2)
    truth_class = (k4 - 2 * k2 ** 2) / k2 ** 2
    rec_b = record_vn('normalized', baseline=True)
    rec_c = record_vn('normalized')
    for dim in ('3D', '2D'):
        vb = production(rec_b, dim, fx=fx)
        vc = production(rec_c, dim, fx=fx)
        e = jk_error(lambda w: production(rec_b, dim, w, fx=fx), 4000)
        check(f'{dim} nc_2{{4}} at fixed multiplicity = {truth_fixed:.4f}', abs(vb - truth_fixed) < 3.5 * e,
              f'{vb:.4f} +- {e:.4f}')
        ec = jk_error(lambda w: production(rec_c, dim, w, fx=fx), 4000)
        check(f'{dim} class average = {truth_class:.4f}', abs(vc - truth_class) < 3.5 * ec, f'{vc:.4f} +- {ec:.4f}')
        check(f'{dim} the two truths differ by > 4 sigma', abs(truth_class - truth_fixed) > 4 * max(e, ec))


def test_routing_and_harness():
    print('test_routing_and_harness')
    from analysis.toy_mc import evaluate
    rec = record_vn('ratio', ratio_order=2)
    rec['value_type'] = 'v_2{4} / v_2{2}'
    check('v_2{4} / v_2{2} routes to multi_cumulant', resolve(rec)['compute'] == 'multi_cumulant')
    fx = _fixture()
    r = copy.deepcopy(record_vn('normalized'))
    r['centrality'] = {'classes': ['0-50%', '50-100%'], 'eta': [{'min': -2.5, 'max': 2.5}]}
    vals, errs = evaluate(r, grid=fx['grid'], n_blocks=10)
    check('harness: two classes, finite values and errors', np.all(np.isfinite(vals)) and np.all(errs > 0),
          f'{vals} +- {errs}')


def test_refusals():
    print('test_refusals')
    bad = [('harmonic 1', dict(record_vn('cumulant'), harmonic=1)),
           ('three-subevent c{6}', record_vn('cumulant', order=6, method='three_subevent')),
           ('normalized without A, C', {k: v for k, v in record_vn('normalized').items()
                                        if k != 'normalization_subevents'}),
           ('three-subevent particles ABC for c{4}', dict(record_vn('cumulant', method='three_subevent'),
                                                          particles='ABC')),
           ('standard method with particles', dict(record_vn('cumulant'), particles='AABC',
                                                   subevents=_subs('ABC'))),
           ('ratio_order 4 of v{4}', record_vn('ratio', ratio_order=4)),
           ('unknown quantity', record_vn('skewness')),
           ('sc with a normalization gap', dict(record_corr('symmetric_cumulant', [2, 3], True),
                                                normalization_eta_gap=1.0)),
           ('differential', dict(record_vn('cumulant'), dimensionality='differential_pT'))]
    for name, rec in bad:
        try:
            resolve(rec)
            ok = False
        except ValueError:
            ok = True
        check(f'refused: {name}', ok)
    # a harmonic the container does not store in a leading term
    fx = _fixture()
    rec = record_corr('symmetric_cumulant', [4, 5], False)
    try:
        production(rec, '3D')
        ok = False
    except ValueError:
        ok = True
    check('refused: sc_{4,5} without Q_9', ok)


if __name__ == '__main__':
    tests = [test_exact_3d, test_physics, test_narrow_bins_exact, test_narrow_bins_toy, test_routing_and_harness,
             test_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
