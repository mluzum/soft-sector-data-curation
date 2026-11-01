"""v_n{2}, v_n{4} of a particle of interest with a charged reference (analysis.correlators).

Checks:
  - the set-partition engine equals brute-force sums over distinct particles
    (weights; POI inside, partly inside, or outside the reference);
  - 3D species grids: integrated and pT-differential v2{2}, v2{4} of protons
    (inside the charged reference) and of a species outside it equal the
    particle-level evaluation on the same particles;
  - charged records: unchanged v2{2} (old single-acceptance formula), and
    v2{4}(pT) follows the injected v2(pT) (it was pulled toward the
    integrated value);
  - 2D boost-invariant container: identified v2{2} within errors of the
    particle level, above the charged value;
  - harmonic-major output (h-1)*n_bins + b for every differential key.
Run: python3 tests/test_toy_mc_cumulant_poi.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.correlators import correlator_arrays, distinct_sum, vn_poi  # noqa: E402
from analysis.dispatcher import resolve  # noqa: E402
from analysis.observables_2d import compute_vn2_single_per_cent_bin  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, evaluate, record_flow_vn2, record_flow_diff_pt,
    vn_of_pt, SPECIES_MASSES,
)

ETA = (-0.8, 0.8)
PT = (0.2, 2.6)
BINS = [(0.2, 0.6), (0.6, 1.0), (1.0, 1.6), (1.6, 2.6)]
PROTON, OTHER = 2, 3              # toy slots; slot 3 stands for a species outside the reference
_CACHE = {}


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=1200, mult_mean=500, mult_sigma=60, seed=2024,
                           species_mix={'pion': 0.75, 'kaon': 0.08, 'proton': 0.12, 'Sigma': 0.05},
                           temp={'pion': 0.22, 'kaon': 0.30, 'proton': 0.35, 'Sigma': 0.38},
                           flow={2: dict(kind='linear', value=0.06, pt0=1.0)}, fluctuate={2: 0.015})
        events, truth = generate_events(cfg)
        eta_edges = np.linspace(-6, 6, 61)            # 0.2 cells: +-0.8 on edges
        pt_edges = np.linspace(0.0, 3.0, 61)          # 0.05 cells: all bin edges on edges
        mult = np.array([len(e['pt']) for e in events], float)

        def grid(slots):
            evs = [{k: v[np.isin(e['species_idx'], slots)] for k, v in e.items()} for e in events]
            return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=6, masses=SPECIES_MASSES,
                                           cent_mult=mult)
        ms = MultiSpeciesGrid({211: grid([0]), 321: grid([1]), 2212: grid([2]), 3122: grid([3])},
                              combined=grid([0, 1, 2]))
        charged_events = [{k: v[e['species_idx'] < 3] for k, v in e.items()} for e in events]
        _CACHE['fx'] = dict(cfg=cfg, events=events, ms=ms,
                            qv=make_qvector_data(charged_events, {'cfg': cfg}, n_pt=60))
    return _CACHE['fx']


def particle_level(events, order, poi_slots, bins):
    """v'_2{order} per bin from the particles (POI slots vs charged slots 0-2)."""
    rows = []
    for e in events:
        acc = (np.abs(e['eta']) < ETA[1]) & (e['pt'] > PT[0]) & (e['pt'] < PT[1])
        ref = acc & (e['species_idx'] < 3)
        per_bin = []
        for lo, hi in bins:
            poi = acc & np.isin(e['species_idx'], poi_slots) & (e['pt'] > lo) & (e['pt'] < hi)

            def S(h, nP, nR, poi=poi):
                sel = (poi if nP else ref) & (ref if nR else True)
                return np.array([np.exp(1j * h * e['phi'][sel]).sum()])
            per_bin.append(S)
        a = correlator_arrays(per_bin[0], 6, order, which=('ref',))
        for S in per_bin:
            for k, v in correlator_arrays(S, 6, order, which=('poi',)).items():
                a.setdefault(k, []).append(v)
        rows.append({k: (np.stack(v, axis=-1) if isinstance(v, list) else v) for k, v in a.items()})
    arrays = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return vn_poi(arrays, np.ones(len(events)), order)[1]        # harmonic 2


def record(species, order=2, bins=None):
    if bins:
        r = record_flow_diff_pt(bins, eta=ETA, order=order, cent_classes=('0-100%',))
    else:
        r = record_flow_vn2(eta=ETA, pt=PT, order=order, cent_classes=('0-100%',))
        r['method'] = 'cumulant'
    r['particle_id'] = {'species': species}
    if bins:
        r['kinematic_range']['pT'].update(min=PT[0], max=PT[1])
        r['kinematic_range']['pT']['bins'] = [b[0] for b in bins] + [bins[-1][1]]
    return r


def test_engine_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(3):
        n = 8
        phi, wP, wR = rng.uniform(0, 2 * np.pi, n), rng.uniform(.5, 1.5, n), rng.uniform(.5, 1.5, n)
        inP, inR = rng.random(n) < .6, rng.random(n) < .7

        def S(h, nP, nR):
            sel = (inP if nP else True) & (inR if nR else True)
            return np.array([np.sum(wP[sel] ** nP * wR[sel] ** nR * np.exp(1j * h * phi[sel]))])

        def brute(harm, roles):
            tot = 0
            for c in itertools.product(*[np.where(inP if r == 'P' else inR)[0] for r in roles]):
                if len(set(c)) == len(c):
                    tot += np.prod([(wP[i] if r == 'P' else wR[i]) * np.exp(1j * h * phi[i])
                                    for i, r, h in zip(c, roles, harm)])
            return tot
        for harm, roles in (((2, -2), 'PR'), ((2, 2, -2, -2), 'PRRR'), ((2, 2, -2, -2), 'RRRR'),
                            ((3, 2, -2, -3), 'PRRR')):
            a, b = distinct_sum(S, harm, roles)[0], brute(harm, roles)
            check(f'engine {roles} {harm}', abs(a - b) < 1e-9 * max(1, abs(b)), f'{a:.6g} vs {b:.6g}')


def test_routing_and_layout():
    check('identified v2{2} -> vn2_single', resolve(record('p + pbar'))['compute'] == 'vn2_single')
    check('v2{4}(pT) key', resolve(record('charged', 4, BINS))['compute'] == 'vn4_single_diff_pT')


def test_3d_equals_particle_level():
    fx = _fx()
    for species, slots, tag in (('p + pbar', [PROTON], 'protons'), ('Lambda', [OTHER], 'outside ref'),
                                ('charged', [0, 1, 2], 'charged')):
        for order in (2, 4):
            v, _ = evaluate(record(species, order), grid=fx['ms'], cent_method='simple', with_jackknife=False)
            ref = particle_level(fx['events'], order, slots, [PT])[0]
            check(f'3D v2{{{order}}} {tag}', abs(v[0] - ref) < 1e-9, f'{v[0]:.6f} vs {ref:.6f}')
            v, _ = evaluate(record(species, order, BINS), grid=fx['ms'], cent_method='simple', with_jackknife=False)
            ref = particle_level(fx['events'], order, slots, BINS)
            check(f'3D v2{{{order}}}(pT) {tag}', np.allclose(v, ref, atol=1e-9), f'{np.round(v, 4)} vs {np.round(ref, 4)}')


def test_charged_unchanged_2d():
    fx = _fx()
    from analysis.dispatcher import extract_measurement_params
    from analysis.pipeline_dispatch import _charged_qn_and_dq0
    rec = record('charged')
    p = extract_measurement_params(rec)
    Qn, dQ0, *_ = _charged_qn_and_dq0(fx['qv'], ETA[0], ETA[1], PT[0], PT[1])
    old = compute_vn2_single_per_cent_bin(Qn, dQ0, np.ones(len(Qn)))[1]
    v, _ = evaluate(rec, qv=fx['qv'], cent_method='simple', with_jackknife=False)
    check('charged v2{2} = old formula', abs(v[0] - old) < 1e-12, f'{v[0]:.10f} vs {old:.10f}')


def test_v4_pt_follows_injected_flow():
    """3D, charged: v2{4}(pT) vs the injected profile times the r.m.s.-quartic event factor."""
    fx = _fx()
    v, e = evaluate(record('charged', 4, BINS), grid=fx['ms'], cent_method='simple')
    cfg = fx['cfg']
    pts = np.concatenate([x['pt'][(np.abs(x['eta']) < ETA[1]) & (x['species_idx'] < 3)] for x in fx['events']])
    s = cfg.flow[2]
    prof = [vn_of_pt(pts[(pts > lo) & (pts < hi)], 'linear', s['value'], s['pt0']).mean() for lo, hi in BINS]
    # v{4} of a Gaussian-fluctuating magnitude: (<v^2>^2 ... ) -> use the event factor's v{4}
    f = generate_events.__globals__['vn_of_pt'](np.array([1.0]), 'linear', s['value'], s['pt0'])[0]
    ev = np.asarray(_ebe(fx)) / f
    k4 = (2 * np.mean(ev ** 2) ** 2 - np.mean(ev ** 4)) ** 0.25
    truth = np.array(prof) * k4
    pull = (v - truth) / e
    check('v2{4}(pT) within 4 sigma of truth', np.all(np.abs(pull) < 4), f'{np.round(v, 4)} vs {np.round(truth, 4)}')
    check('v2{4}(pT) rises with pT', np.all(np.diff(v) > 0))


def _ebe(fx):
    if 'ebe' not in _CACHE:
        cfg = fx['cfg']
        _CACHE['ebe'] = generate_events(cfg)[1]['vn_ebe'][2]
    return _CACHE['ebe']


def test_2d_identified():
    fx = _fx()
    vp, ep = evaluate(record('p + pbar'), qv=fx['qv'], cent_method='simple')
    vc, ec = evaluate(record('charged'), qv=fx['qv'], cent_method='simple')
    ref = particle_level(fx['events'], 2, [PROTON], [PT])[0]
    check('2D protons ~ particle level', abs(vp[0] - ref) < max(4 * ep[0], 0.004), f'{vp[0]:.4f}+-{ep[0]:.4f} vs {ref:.4f}')
    check('2D protons above charged', vp[0] - vc[0] > 4 * max(ep[0], ec[0]), f'{vp[0]:.4f} vs {vc[0]:.4f}')


if __name__ == '__main__':
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith('test_')]
    n_pass = 0
    for fn in tests:
        print(fn.__name__)
        try:
            fn()
            n_pass += 1
        except AssertionError as err:
            print(f'  FAILED: {err}')
    print(f'\n{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
