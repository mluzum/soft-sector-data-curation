"""The iEBE-MUSIC standard output's species axis (2D): an exact charged
decomposition with an "other charged" row (PDG 0), K0 rows, and a highest
stored harmonic per row (species_nmax).

Synthetic containers through analysis.sim_package.build_container (the
adapter contract) and the toy harness:
  - 'charged' is every charged row (pi, K, p, Xi, Omega and PDG 0) on an
    axis with the PDG-0 row: particles only in the PDG-0 row give the
    charged yield of the same counts in the Xi- row at the same mass;
    particles only in Lambda give none; without the PDG-0 row 'charged'
    stays pi + K + p (Xi not counted);
  - K0_S is (K0 + K0bar)/2: half of K+ + K- with the same counts;
  - species_nmax: a Lambda or K0_S record's prepare gets Qn to n = 6, a
    charged one all harmonics; Lambda v2{2} computes;
  - build_container refuses a nmax above the stored harmonics, sums above
    a row's nmax, a species_charge that disagrees with the PDG ids, and an
    uncharged PDG-0 row.
Run: python3 tests/test_standard_species_axis.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.dispatcher import extract_measurement_params  # noqa: E402
from analysis.pipeline_dispatch import container_for_prepare_2d  # noqa: E402
from analysis.sim_package import SimPackageError, build_container  # noqa: E402

PDG = [211, -211, 321, -321, 2212, -2212, 3122, -3122, 3312, -3312, 3334, -3334, 311, -311, 0]
CHARGE = [1, -1, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 0, 0, 1]
NMAX = [10, 10, 10, 10, 10, 10, 6, 6, 10, 10, 10, 10, 6, 6, 10]
MASS = [0.138, 0.138, 0.494, 0.494, 0.938, 0.938, 1.116, 1.116, 1.315, 1.315, 1.672, 1.672, 0.494, 0.494, 1.1894]
EDGES = np.round(np.linspace(0.0, 3.0, 31), 10)


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _arrays(rows, n_ev=400, seed=1, pdg=PDG, mass=MASS, charge=CHARGE, nmax=NMAX):
    """Particles only in `rows` (row -> mean count per pT bin), flow v2 = 0.08,
    a multiplicity spread for centrality."""
    rng = np.random.default_rng(seed)
    S, P = len(pdg), len(EDGES) - 1
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    scale = rng.uniform(0.2, 1.0, n_ev)
    q = np.zeros((n_ev, S, 11, P), complex)
    psi = rng.uniform(0, 2 * np.pi, n_ev)
    for r, mean in rows.items():
        c = rng.poisson(mean * scale[:, None] * np.exp(-centres / 0.5)[None, :] * 10).astype(float)
        q[:, r, 0, :] = c
        for n in range(1, nmax[r] + 1):
            v = 0.08 if n == 2 else 0.02
            q[:, r, n, :] = c * v * np.exp(1j * n * psi)[:, None] + rng.normal(0, 1, c.shape) * np.sqrt(c) * 0.7
    return {'Qn': q, 'pt_bin_edges': EDGES, 'masses': np.array(mass), 'species_names': [str(p) for p in pdg],
            'pdg_ids': list(pdg), 'species_charge': list(charge), 'species_nmax': list(nmax),
            'nsamples': np.full(n_ev, 4.0), 'rap_max': 5.0,
            'sum_pt': q[:, :, 0, :].real * centres}


def _yield(qv, species):
    from analysis.toy_mc import evaluate, record_yield
    v, _ = evaluate(record_yield(species=species, cent_classes=('0-20%', '20-60%')), qv=qv, with_jackknife=False)
    return v


def test_charged_and_k0():
    print('test_charged_and_k0')
    sigma = PDG.index(0)
    qv0 = build_container(_arrays({sigma: 5.0}), '2d')
    # the same counts in the Xi- row with the PDG-0 row's mass: a charged row
    m2 = list(MASS)
    m2[8] = MASS[sigma]
    qx = build_container(_arrays({8: 5.0}, mass=m2), '2d')
    y0, yx = _yield(qv0, 'charged'), _yield(qx, 'charged')
    check('charged includes the PDG-0 row (= the same counts in another charged row)',
          np.all(y0 > 0) and np.allclose(y0, yx, rtol=1e-12), f'{y0} vs {yx}')
    yl = _yield(build_container(_arrays({6: 5.0}), '2d'), 'charged')
    check('Lambda is not charged', np.allclose(yl, 0.0), f'{yl}')
    # without the PDG-0 row: the legacy pi + K + p (Xi- not counted)
    legacy = [p for p in PDG if p != 0]
    keep = [PDG.index(p) for p in legacy]
    a = _arrays({8: 5.0}, pdg=legacy, mass=[MASS[i] for i in keep], charge=[CHARGE[i] for i in keep],
                nmax=[NMAX[i] for i in keep])
    yleg = _yield(build_container(a, '2d'), 'charged')
    check('without the PDG-0 row, charged stays pi + K + p', np.allclose(yleg, 0.0), f'{yleg}')
    a = _arrays({0: 5.0, 1: 5.0, 2: 3.0, 3: 3.0}, seed=3)
    a['Qn'][:, 12], a['Qn'][:, 13] = a['Qn'][:, 2], a['Qn'][:, 3]     # K0, K0bar = K+, K- (same mass)
    a['sum_pt'][:, 12], a['sum_pt'][:, 13] = a['sum_pt'][:, 2], a['sum_pt'][:, 3]
    a['Qn'][:, 12, 7:], a['Qn'][:, 13, 7:] = 0, 0                       # K0 rows end at n = 6
    k = build_container(a, '2d')
    yk0, ykp = _yield(k, 'K0_S'), _yield(k, 'K+ + K-')
    check('K0_S = (K0 + K0bar)/2: half of K+ + K- with the same counts', np.allclose(yk0, ykp / 2, rtol=1e-12),
          f'{yk0} vs {ykp / 2}')


def test_nmax_and_refusals():
    print('test_nmax_and_refusals')
    from analysis.toy_mc import evaluate, record_flow_vn2
    qv = build_container(_arrays({0: 5.0, 1: 5.0, 6: 3.0, 7: 3.0, 12: 2.0, 13: 2.0}), '2d')
    for species, n in (('charged', 11), ('Lambda', 7), ('K0_S', 7), ('pi+ + pi-', 11), (None, 11)):
        rec = record_flow_vn2()
        if species:
            rec['particle_id'] = {'species': species}
        got = container_for_prepare_2d(qv, extract_measurement_params(rec)).Qn.shape[2]
        check(f'{species}: the prepare sees harmonics 0..{n - 1}', got == n, f'{got}')
    rec = record_flow_vn2(cent_classes=('0-30%', '30-70%'))
    rec['particle_id'] = {'species': 'Lambda'}
    v, _ = evaluate(rec, qv=qv, with_jackknife=False)
    check('Lambda v2{2} computes on the truncated container', np.all(np.isfinite(v)) and np.all(v > 0.03), f'{v}')
    bad = [('species_nmax above the stored harmonics', dict(species_nmax=[11] * 15)),
           ('a charge that disagrees with the PDG ids', dict(species_charge=[1] * 15)),
           ('an uncharged PDG-0 row', dict(species_charge=CHARGE[:-1] + [0]))]
    for name, change in bad:
        a = _arrays({0: 5.0})
        a.update(change)
        try:
            build_container(a, '2d')
            ok = False
        except SimPackageError:
            ok = True
        check(f'refused: {name}', ok)
    a = _arrays({6: 5.0})
    a['Qn'][:, 6, 8, :] = 1.0                       # Lambda Q8: above its nmax
    try:
        build_container(a, '2d')
        ok = False
    except SimPackageError:
        ok = True
    check('refused: sums above a row\'s nmax', ok)


if __name__ == '__main__':
    tests = [test_charged_and_k0, test_nmax_and_refusals]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
        except AssertionError as exc:
            print(f'FAILED {t.__name__}: {exc}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
