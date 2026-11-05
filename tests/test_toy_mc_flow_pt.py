"""Toy-MC validation of rho(v_n^2, [pT]) and cov(v_n^2, [pT]) (analysis/flow_pt.py).

- cov_n against explicit sums over distinct triplets of the particles
  (standard, two- and three-subevent methods; 3D cells on the window edges:
  exact for disjoint subevents, within the cell-mean-pT approximation of
  the pT-weighted flow vector otherwise);
- rho_n against the correlation injected between each event's v_2^2 and
  its pT scale (2D and 3D, every method), and zero without it (no
  self-correlation bias where flow and [pT] share particles);
- events of nsamples particlizations (weights 1/nsamples).
Run: python3 tests/test_toy_mc_flow_pt.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from analysis.converters import particles_to_qgrid_data, y_to_eta
from analysis.toy_mc import (SPECIES_MASSES, ToyGenConfig, evaluate, generate_events, make_qvector_data,
                             record_rho)

ETA_EDGES = np.linspace(-1.6, 1.6, 33)        # 0.1 cells: the windows on edges
FULL, A, B, C = (-0.8, 0.8), (-0.8, -0.4), (-0.4, 0.4), (0.4, 0.8)
PT = (0.2, 3.0)
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _grid(events, cfg, nsamples=None):
    # 0.02 GeV cells, 0.2 and 3.0 on edges, above the rescaled spectra (the
    # converter puts larger pT in the last cell)
    return particles_to_qgrid_data(events, ETA_EDGES, np.linspace(0.0, 4.0, 201), n_harm=5,
                                   nsamples=nsamples, cent_mult=np.array([len(e['pt']) for e in events], float),
                                   with_sum_pt=True, with_sum_pt2=True, masses=SPECIES_MASSES)


def _fixture(r, n_events=1500, mult=1500, seed=11):
    key = (r, n_events, mult, seed)
    if key not in _CACHE:
        cfg = ToyGenConfig(n_events=n_events, mult_mean=mult, mult_sigma=mult // 10, seed=seed, rap_max=1.2,
                           flow={2: dict(kind='const', value=0.08)}, fluctuate={2: 0.035},
                           pt_rescale_sigma=0.04, vn_pt_corr=(2, r))
        ev, truth = generate_events(cfg)
        _CACHE[key] = dict(events=ev, truth=truth, cfg=cfg, qv=make_qvector_data(ev, truth, n_pt=60),
                           grid=_grid(ev, cfg))
    return _CACHE[key]


def _sel(e, win):
    return (e['eta'] > win[0]) & (e['eta'] < win[1]) & (e['pt'] > PT[0]) & (e['pt'] < PT[1])


def brute_cov(events, method, n=2):
    """cov_n from explicit distinct triplets: i in X, j in Y, k in Z."""
    T, W, PZ = [], [], []
    for e in events:
        phi, p = e['phi'], e['pt']
        if method == 'standard':
            s = _sel(e, FULL)
            ph, pp = phi[s], p[s]
            Cm = np.cos(n * (ph[:, None] - ph[None, :]))
            np.fill_diagonal(Cm, 0.0)
            R = Cm.sum(axis=1)
            N = len(pp)
            T.append((pp[:, None] * (Cm.sum() - 2 * R)[:, None]).sum() if N >= 3 else 0.0)
            T1 = (Cm.sum() - 2 * R).sum()
            W.append(N * (N - 1) * (N - 2))
            PZ.append((pp.sum(), N, T1))
            continue
        sa, sc = _sel(e, A), _sel(e, C)
        Cm = np.cos(n * (phi[sa][:, None] - phi[sc][None, :]))   # A x C (disjoint)
        S_ac = Cm.sum()
        if method == 'three_subevent':
            sb = _sel(e, B)
            T.append(S_ac * p[sb].sum())
            W.append(sa.sum() * sc.sum() * sb.sum())
            PZ.append((p[sb].sum(), sb.sum(), S_ac * sb.sum()))
        else:   # two_subevent: k in A or C, distinct from i and j
            pa, pc = p[sa], p[sc]
            tp = (pa * (S_ac - Cm.sum(axis=1))).sum() + (pc * (S_ac - Cm.sum(axis=0))).sum()
            t1 = (S_ac - Cm.sum(axis=1)).sum() + (S_ac - Cm.sum(axis=0)).sum()
            T.append(tp)
            W.append(len(pa) * len(pc) * (len(pa) - 1) + len(pa) * len(pc) * (len(pc) - 1))
            PZ.append((pa.sum() + pc.sum(), len(pa) + len(pc), t1))
    T, W = np.array(T, float), np.array(W, float)
    pz = np.array([x[0] / x[1] for x in PZ])
    T1 = np.array([x[2] for x in PZ], float)
    M = np.sum(W * pz) / W.sum()
    return (T.sum() - M * T1.sum()) / W.sum()


def test_cov_exact_3d():
    """cov_n on 3D cells against the particle-level distinct-triplet sums."""
    print('test_cov_exact_3d')
    fx = _fixture(0.5, n_events=300, mult=200, seed=12)
    for method, tol in (('three_subevent', 1e-9), ('two_subevent', 0.01), ('standard', 0.01)):
        got = evaluate(record_rho(value_type='cov_n', method=method), grid=fx['grid'], cent_method='simple',
                       with_jackknife=False)[0][0]
        ref = brute_cov(fx['events'], method)
        check(f'{method}: cov_2', abs(got - ref) <= tol * abs(ref) + 1e-12, f'{got:.6e} vs {ref:.6e}')


def test_rho_injected():
    """rho(v_2^2, [pT]) recovers the injected correlation, every method, 2D and 3D."""
    print('test_rho_injected')
    fx = _fixture(0.5)
    truth = np.corrcoef(fx['truth']['vn_ebe'][2] ** 2, fx['truth']['pt_rescale'])[0, 1]
    for method, vt, ck in (('three_subevent', None, 'B'), ('two_subevent', None, 'full'),
                           ('combined_subevent', None, 'full'),
                           ('standard', {'c4': 'standard', 'c2': 'standard'}, 'full'),
                           ('three_subevent', {'c4': 'standard', 'c2': 'two_subevent'}, 'full')):
        rec = record_rho(method=method, variance_terms=vt, ck_window=ck)
        for tag, cont in (('2D', dict(qv=fx['qv'])), ('3D', dict(grid=fx['grid']))):
            v, e = evaluate(rec, cent_method='simple', **cont)
            check(f'{tag} {method} (var {vt or "two_subevent"}, c_k {ck}): rho = {truth:.3f}',
                  abs(v[0] - truth) < 3 * e[0] + 0.02, f'{v[0]:.3f} +- {e[0]:.3f}')


def test_rho_uncorrelated():
    """No injected correlation: rho ~ 0 also where flow and [pT] share
    particles (standard method), i.e. no self-correlation bias."""
    print('test_rho_uncorrelated')
    fx = _fixture(0.0)
    truth = np.corrcoef(fx['truth']['vn_ebe'][2] ** 2, fx['truth']['pt_rescale'])[0, 1]
    for method, vt in (('standard', {'c4': 'standard', 'c2': 'standard'}), ('three_subevent', None)):
        rec = record_rho(method=method, variance_terms=vt, ck_window='full')
        for tag, cont in (('2D', dict(qv=fx['qv'])), ('3D', dict(grid=fx['grid']))):
            v, e = evaluate(rec, cent_method='simple', **cont)
            check(f'{tag} {method}: rho = {truth:.3f}', abs(v[0] - truth) < 3 * e[0] + 0.02,
                  f'{v[0]:.3f} +- {e[0]:.3f}')


def test_multiplicity_baseline():
    """A [pT] trend with the event multiplicity across the class (ATLAS's
    wide classes; ATLAS measures in narrow Sigma E_T / N_ch bins) inflates
    c_k and lowers rho; `multiplicity_baseline` (deviations at the event's
    multiplicity) recovers the injected rho."""
    print('test_multiplicity_baseline')
    fx = _fixture(0.5, seed=14)
    events = [dict(e) for e in fx['events']]
    mult = np.array([len(e['pt']) for e in events], float)
    for e, m in zip(events, mult):
        e['pt'] = e['pt'] * (1 + 0.5 * (m - mult.mean()) / mult.mean())
        e['eta'] = y_to_eta(e['y'], e['pt'], e['mass'])
    truth = np.corrcoef(fx['truth']['vn_ebe'][2] ** 2, fx['truth']['pt_rescale'])[0, 1]
    conts = (('2D', dict(qv=make_qvector_data(events, fx['truth'], n_pt=60, pt_max=4.0))),
             ('3D', dict(grid=_grid(events, fx['cfg']))))
    base = {'observable': 'N', 'eta': [{'min': -1.6, 'max': 1.6}], 'pT': {'min': 0.0, 'max': 4.0}}
    for tag, cont in conts:
        rec = record_rho(method='three_subevent', ck_window='full')
        plain, e0 = evaluate(rec, cent_method='simple', **cont)
        fixed, e1 = evaluate(dict(rec, multiplicity_baseline=base), cent_method='simple', **cont)
        check(f'{tag}: the trend lowers rho', plain[0] < truth - 0.1, f'{plain[0]:.3f} +- {e0[0]:.3f}')
        check(f'{tag}: fixed multiplicity recovers rho = {truth:.3f}', abs(fixed[0] - truth) < 3 * e1[0] + 0.02,
              f'{fixed[0]:.3f} +- {e1[0]:.3f}')


def test_oversampled():
    """Events of nsamples particlizations: cov_n of the super-events (all
    their particles) on 3D, and rho on 2D and 3D unchanged by the sampling."""
    print('test_oversampled')
    ns, n_super = 4, 400
    cfg = ToyGenConfig(n_events=ns * n_super, mult_mean=60, mult_sigma=6, seed=13, rap_max=1.2,
                       flow={2: dict(kind='const', value=0.08)})
    base, _ = generate_events(cfg)
    rng = np.random.default_rng(5)
    sup = []
    for i in range(n_super):
        parts = base[i * ns:(i + 1) * ns]
        ev = {k: np.concatenate([q[k] for q in parts]) for k in ('y', 'pt', 'species_idx', 'mass')}
        ev['phi'] = np.concatenate([q['phi'] for q in parts])
        ev['pt'] = ev['pt'] * rng.normal(1.0, 0.05)
        ev['eta'] = y_to_eta(ev['y'], ev['pt'], ev['mass'])
        sup.append(ev)
    grid = _grid(sup, cfg, nsamples=np.full(n_super, float(ns)))
    for method, tol in (('three_subevent', 1e-9), ('standard', 0.01)):
        got = evaluate(record_rho(value_type='cov_n', method=method), grid=grid, cent_method='simple',
                       with_jackknife=False)[0][0]
        ref = brute_cov(sup, method)
        check(f'nsamples {ns}, {method}: cov_2', abs(got - ref) <= tol * abs(ref) + 1e-12,
              f'{got:.6e} vs {ref:.6e}')


if __name__ == '__main__':
    tests = [test_cov_exact_3d, test_oversampled, test_rho_injected, test_rho_uncorrelated,
             test_multiplicity_baseline]
    n_pass = n_fail = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except Exception as exc:
            print(f'  FAIL {t.__name__}: {exc}')
            n_fail += 1
    print(f'{n_pass}/{n_pass + n_fail} passed')
    sys.exit(1 if n_fail else 0)
