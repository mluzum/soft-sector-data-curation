"""Toy-MC validation of rapidity-odd v1 of an identified species in rapidity
bins (v1_odd_sp_diff_eta with differential_y), as STAR's BES v1(y) of
kaons, Lambda, phi (INSPIRE 1618747) and pions and protons (1277069).

Truth per particle: v1 = c(pT) f_species(eta), with c(pT) rising linearly
in pT and the species shapes of opposite sign at midrapidity (pions
negative, kaons positive).  The reference is charged hadrons at 3.3 <
|eta| < 5 (STAR's BBC), odd; the sign is fixed by pions being negative at
0 < y < 1 (STAR's pion slope, negative at every energy and centrality).
The POI are K+ in y bins, pT > 0.2 GeV/c and |p| < 1.6 GeV/c (STAR's PID
limit), so the momentum cut removes the high-pT kaons of the outer bins,
whose v1 is largest.  Compared with the particle-level average of each
bin.  Run: python3 tests/test_toy_mc_v1_odd_pid.py
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import SPECIES_MASSES, ToyGenConfig, evaluate, generate_events, vn_of_pt  # noqa: E402

EDGES = np.round(np.linspace(-1.0, 1.0, 11), 10)
BINS = list(zip(EDGES[:-1], EDGES[1:]))
PT, P_MAX = (0.2, 3.0), 1.6
FLOW1 = dict(kind='linear', value=1.0, pt0=1.0)
SHAPES = {'pion': lambda eta: -0.015 * eta + 0.002 * eta ** 3,
          'kaon': lambda eta: 0.05 * np.tanh(eta),
          'proton': lambda eta: 0.01 * eta + 0.002 * eta ** 3}

RECORD = dict(
    observable_type='flow_vn', value_type='v1', harmonic=1, method='event_plane',
    dimensionality='differential_y', hepdata_file='toy.yaml', inspire_id='0', version=1,
    particle_id=dict(species='K+'),
    kinematic_range=dict(y=dict(min=-1.0, max=1.0, bins=[float(x) for x in EDGES]),
                         pT=dict(min=PT[0], max=PT[1]), p=dict(max=P_MAX)),
    centrality=dict(classes=['0-100%'], eta=[dict(min=-0.5, max=0.5)]),
    flow_reference=dict(
        odd=True, detector='BBC', eta=[dict(min=-5.0, max=-3.3), dict(min=3.3, max=5.0)],
        pT=dict(min=0.0, max=100.0),
        sign_convention=dict(species='pi+ + pi-', y=dict(min=0.0, max=1.0), pT=dict(min=0.2, max=1.6),
                             sign='negative')))

results = []
_CACHE = {}


def check(name, ok, detail=''):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + ('' if ok else f'  {detail}'))


def fixture():
    if 'fx' in _CACHE:
        return _CACHE['fx']
    cfg = ToyGenConfig(n_events=4000, mult_mean=2500, mult_sigma=100, rap_max=5.5, seed=1618,
                       species_mix={'pion': 0.6, 'kaon': 0.25, 'proton': 0.15}, pt_min=0.05, pt_max=3.0,
                       flow={1: FLOW1, 2: dict(kind='const', value=0.05)}, eta_shape={1: SHAPES})
    events, _ = generate_events(cfg)
    eta_edges = np.round(np.linspace(-6, 6, 121), 10)
    pt_edges = np.round(np.linspace(0.0, 3.0, 31), 10)
    mult = np.array([len(e['pt']) for e in events], float)

    def grid(slot=None):
        evs = events if slot is None else [{k: v[e['species_idx'] == slot] for k, v in e.items()} for e in events]
        return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=4, masses=SPECIES_MASSES, cent_mult=mult)
    ms = MultiSpeciesGrid({211: grid(0), 321: grid(1), 2212: grid(2)}, combined=grid())
    _CACHE['fx'] = (events, ms)
    return _CACHE['fx']


def truth(events, slot, p_max=P_MAX):
    shape = SHAPES[['pion', 'kaon', 'proton'][slot]]
    out = []
    for lo, hi in BINS:
        v = []
        for ev in events:
            p = ev['pt'] * np.cosh(ev['eta'])
            sel = ((ev['species_idx'] == slot) & (ev['y'] >= lo) & (ev['y'] < hi) & (ev['pt'] >= PT[0])
                   & (ev['pt'] < PT[1]) & (p < p_max))
            v.append(vn_of_pt(ev['pt'][sel], FLOW1['kind'], FLOW1['value'], FLOW1['pt0'])
                     * shape(ev['eta'][sel]))
        out.append(np.concatenate(v).mean())
    return np.array(out)


def main():
    from analysis.dispatcher import resolve
    events, ms = fixture()
    t_cut, t_all = truth(events, 1), truth(events, 1, np.inf)
    vals, errs = evaluate(RECORD, grid=ms, cent_method='simple', n_blocks=20)
    for (lo, hi), v, e, t, ta in zip(BINS, vals, errs, t_cut, t_all):
        print(f"    y {lo:+.1f}..{hi:+.1f}: v1 = {v:+.4f} +- {e:.4f}   truth {t:+.4f}  (no p cut {ta:+.4f})")
    pull = np.abs(vals - t_cut) / np.maximum(errs, 1e-6)
    check('every K+ bin agrees with the input v1 within 4 sigma', np.all(pull < 4), f'pulls {pull.round(1)}')
    w2 = t_cut ** 2 / errs ** 2
    scale, s_err = np.sum(vals * t_cut / errs ** 2) / w2.sum(), 1 / np.sqrt(w2.sum())
    print(f"    fitted scale result/input = {scale:.3f} +- {s_err:.3f}")
    check('overall scale consistent with 1 and known to 0.05', abs(scale - 1) < 3 * s_err and s_err < 0.05,
          f'{scale:.3f} +- {s_err:.3f}')
    check('K+ positive at y > 0 although the anchor (pions) is negative there', np.all(vals[5:] > 0))

    # the POI counts per y bin, with and without the momentum cut (no flow
    # noise): the y -> eta mapping per pT cell and the cut fractions
    from analysis.dispatcher import extract_measurement_params
    from analysis.pipeline_dispatch import get_pipeline_entry
    entry = get_pipeline_entry('v1_odd_sp_diff_eta')
    for label, pmax, eta_max in (('with the |p| < 1.6 cut', P_MAX, np.inf), ('without the cut', np.inf, np.inf),
                                 ('with the cut and |eta| < 1', P_MAX, 1.0)):
        rec = copy.deepcopy(RECORD)
        if pmax == np.inf:
            del rec['kinematic_range']['p']
        if eta_max < np.inf:
            rec['kinematic_range']['eta'] = dict(min=-eta_max, max=eta_max)
        n_pipe = entry['prepare_data_3d'](ms, extract_measurement_params(rec))['n_poi'].sum(axis=0)
        n_true = np.array([sum(np.count_nonzero(
            (ev['species_idx'] == 1) & (ev['y'] >= lo) & (ev['y'] < hi) & (ev['pt'] >= PT[0]) & (ev['pt'] < PT[1])
            & (ev['pt'] * np.cosh(ev['eta']) < pmax) & (np.abs(ev['eta']) < eta_max)) for ev in events)
            for lo, hi in BINS])
        if pmax < np.inf and eta_max == np.inf:
            n_cut = n_true
        elif pmax == np.inf:
            n_all = n_true
        dev = n_pipe / n_true - 1
        print(f"    K+ per bin {label}: pipeline/particles - 1 = {dev.round(4)}  (kept {n_true.min()}-{n_true.max()})")
        check(f'K+ counts per y bin {label} within 1.5%', np.all(np.abs(dev) < 0.015))
    n_true = n_all
    removed = 1 - n_cut / n_true
    check('the cut removes more than twice the central share from the outer bins (pT cosh eta)',
          removed[[0, 9]].min() > 2 * removed[[4, 5]].max() > 0.05, f'{removed.round(3)}')

    # the anchor species: anchoring on the kaons themselves (positive) gives
    # the same curve; anchoring on pions as positive flips it
    rec = copy.deepcopy(RECORD)
    rec['flow_reference']['sign_convention'] = dict(y=dict(min=0.0, max=1.0), sign='positive')
    v_self, _ = evaluate(rec, grid=ms, cent_method='simple', n_blocks=20, with_jackknife=False)
    check('a K+ anchor (positive) gives the same values', np.allclose(v_self, vals, rtol=1e-10, atol=1e-12))
    rec = copy.deepcopy(RECORD)
    rec['flow_reference']['sign_convention']['sign'] = 'positive'
    v_flip, _ = evaluate(rec, grid=ms, cent_method='simple', n_blocks=20, with_jackknife=False)
    check('a positive pion anchor flips the whole curve', np.allclose(v_flip, -vals, rtol=1e-10, atol=1e-12))

    # pions, whose particles are in the charged reference, and protons
    for sp, slot in (('pi+ + pi-', 0), ('p', 2)):
        rec = copy.deepcopy(RECORD)
        rec['particle_id'] = dict(species=sp)
        v, e = evaluate(rec, grid=ms, cent_method='simple', n_blocks=20)
        t = truth(events, slot)
        pull = np.abs(v - t) / np.maximum(e, 1e-6)
        check(f'{sp}: every bin within 4 sigma of the input', np.all(pull < 4),
              f'{v.round(4)} vs {t.round(4)}; pulls {pull.round(1)}')

    # refusals
    def refused(rec):
        try:
            resolve(rec)
        except ValueError:
            return True
        return False
    rec = copy.deepcopy(RECORD)
    rec['particle_id'] = dict(species='charged')
    check('charged hadrons in y bins are refused (no single mass)', refused(rec))
    rec = copy.deepcopy(RECORD)
    rec['flow_reference']['sign_convention'] = dict(species='charged', y=dict(min=0.0, max=1.0), sign='negative')
    check('a charged sign window in rapidity is refused', refused(rec))
    rec = dict(observable_type='flow_vn', value_type='v2', harmonic=2, method='cumulant', cumulant_order=2,
               particle_id=dict(species='charged'), dimensionality='integrated',
               kinematic_range=dict(eta=dict(min=-1, max=1), pT=dict(min=0.2, max=2.0), p=dict(max=1.6)),
               centrality=dict(classes=['0-10%']))
    check('a momentum cut on another observable is refused', refused(rec))


if __name__ == '__main__':
    main()
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
