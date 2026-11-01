"""Two-subevent v_n{2} of identified particles with a charged reference (vn2_gap_poi).

Toy events with a rising v2(pT) give protons a larger pT-integrated v2 than
charged hadrons.  The record (STAR PRL 129, 252301 style): protons with
0.2 < pT < 2 GeV/c correlated across a gap with charged hadrons, divided by
the charged v2{2}ab.  Checks:
  - an identified two-subevent record resolves to vn2_gap_poi, a charged
    one still to vn2_gap;
  - 3D (species grids, windows on cell edges): equal to the particle-level
    estimator on the same particles;
  - 2D (boost-invariant container): within errors of the particle-level
    value, and well above the charged v2{2}ab;
  - the truth: <v2(pT)> over the POI spectrum times the r.m.s. event factor;
  - overlapping subevents are refused on a 3D grid.
Run: python3 tests/test_toy_mc_vn2_gap_poi.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.converters import particles_to_qgrid_data  # noqa: E402
from analysis.dispatcher import extract_measurement_params, resolve  # noqa: E402
from analysis.pipeline_dispatch import get_pipeline_entry  # noqa: E402
from analysis.qvector import MultiSpeciesGrid  # noqa: E402
from analysis.toy_mc import (  # noqa: E402
    ToyGenConfig, generate_events, make_qvector_data, evaluate, vn_of_pt, SPECIES_MASSES,
)

NE = 2000
POI_PT = (0.2, 2.0)
REF_PT = (0.2, 3.0)
A, B = (0.4, 1.0), (-1.0, -0.4)          # on the 3D grid's 0.2-wide eta cells
PROTON = 2                               # toy species slot (positives only)
_CACHE = {}


def _fx():
    if 'fx' not in _CACHE:
        cfg = ToyGenConfig(n_events=NE, mult_mean=600, mult_sigma=80, seed=4242,
                           flow={2: dict(kind='linear', value=0.06, pt0=1.0)},
                           fluctuate={2: 0.02})
        events, truth = generate_events(cfg)
        eta_edges = np.linspace(-6, 6, 61)
        pt_edges = np.linspace(0.05, 3.0, 60)

        def grid(slot=None):
            evs = events if slot is None else [
                {k: v[ev['species_idx'] == slot] for k, v in ev.items()} for ev in events]
            return particles_to_qgrid_data(evs, eta_edges, pt_edges, n_harm=6, masses=SPECIES_MASSES,
                                           cent_mult=np.array([len(ev['pt']) for ev in events], float))

        ms = MultiSpeciesGrid({211: grid(0), 321: grid(1), 2212: grid(2)}, combined=grid())
        _CACHE['fx'] = dict(cfg=cfg, events=events, truth=truth, ms=ms,
                            qv=make_qvector_data(events, {'cfg': cfg}, n_pt=80))
    return _CACHE['fx']


def record(species='p + pbar', sub_a=A, sub_b=B, ref_species=None):
    sub = {}
    for lab, eta in (('A', sub_a), ('B', sub_b)):
        sub[lab] = {'kinematic_range': {'eta': {'min': eta[0], 'max': eta[1]},
                                        'pT': {'min': REF_PT[0], 'max': REF_PT[1]}}}
        if ref_species:
            sub[lab]['particle_id'] = {'species': ref_species}
    return dict(observable_type='flow_vn', method='cumulant', cumulant_order=2, particles='AB',
                dimensionality='integrated', harmonic=2, value_type='v2{2}', hepdata_file='toy.yaml',
                inspire_id='0', version=1, particle_id={'species': species},
                kinematic_range={'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': POI_PT[0], 'max': POI_PT[1]}},
                subevents=sub,
                centrality={'classes': ['0-100%'], 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]})


def particle_level(events, h=2, poi_slot=PROTON, poi_pt=POI_PT):
    """Brute force: POI in A x reference in B, plus B x A, over the reference v_n{2}ab."""
    num_d = den_d = num_c = den_c = 0.0
    for ev in events:
        def q(eta, pt, slot=None):
            sel = (ev['eta'] > eta[0]) & (ev['eta'] < eta[1]) & (ev['pt'] > pt[0]) & (ev['pt'] < pt[1])
            if slot is not None:
                sel &= ev['species_idx'] == slot
            return np.exp(1j * h * ev['phi'][sel]).sum(), sel.sum()
        pA, mA = q(A, poi_pt, poi_slot)
        pB, mB = q(B, poi_pt, poi_slot)
        QA, MA = q(A, REF_PT)
        QB, MB = q(B, REF_PT)
        num_d += (pA * np.conj(QB)).real + (pB * np.conj(QA)).real
        den_d += mA * MB + mB * MA
        num_c += (QA * np.conj(QB)).real
        den_c += MA * MB
    return (num_d / den_d) / np.sqrt(num_c / den_c)


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        raise AssertionError(f'{name}: {detail}')


def test_routing():
    check('identified -> vn2_gap_poi', resolve(record())['compute'] == 'vn2_gap_poi')
    check('charged -> vn2_gap', resolve(record(species='charged'))['compute'] == 'vn2_gap')
    p = extract_measurement_params(record(ref_species='charged'))
    check('subevent species in params', p['subevent_ranges']['A']['particle_id'] == {'species': 'charged'})
    check('POI range in params', p['poi_range']['pT'] == POI_PT, str(p['poi_range']))


def test_3d_equals_particle_level():
    fx = _fx()
    ref = particle_level(fx['events'])
    # full-sample value
    v, _ = evaluate(record(), grid=fx['ms'], cent_method='simple', with_jackknife=False)
    check('3D == particle level', abs(v[0] - ref) < 1e-9 * max(abs(ref), 1), f'{v[0]:.9f} vs {ref:.9f}')
    vc, _ = evaluate(record(ref_species='charged'), grid=fx['ms'], cent_method='simple', with_jackknife=False)
    check('explicit charged reference is the default', abs(vc[0] - v[0]) < 1e-12)


def test_2d_within_errors():
    fx = _fx()
    ref = particle_level(fx['events'])
    v, e = evaluate(record(), qv=fx['qv'], cent_method='simple')
    check('2D ~ particle level', abs(v[0] - ref) < max(4 * e[0], 0.004), f'{v[0]:.4f}+-{e[0]:.4f} vs {ref:.4f}')
    vch, ech = evaluate(record(species='charged'), qv=fx['qv'], cent_method='simple')
    check('protons above charged', v[0] - vch[0] > 5 * max(e[0], ech[0]),
          f'p {v[0]:.4f} vs charged {vch[0]:.4f}')


def test_truth():
    """d/sqrt(c) = <v2(pT)>_POI * sqrt(<s^2>), s = event factor (v_e / v(pT0))."""
    fx = _fx()
    pts = np.concatenate([ev['pt'][(ev['species_idx'] == PROTON) & (ev['pt'] > POI_PT[0]) & (ev['pt'] < POI_PT[1])
                                   & (np.abs(ev['eta']) > 0.4) & (np.abs(ev['eta']) < 1.0)] for ev in fx['events']])
    flow = fx['cfg'].flow[2]
    prof = vn_of_pt(pts, 'linear', flow['value'], flow['pt0']).mean()
    s = fx['truth']['vn_ebe'][2] / vn_of_pt(np.array([flow['pt0']]), 'linear', flow['value'], flow['pt0'])[0]
    truth = prof * np.sqrt(np.mean(s ** 2))
    v, e = evaluate(record(), grid=fx['ms'], cent_method='simple')
    check('3D ~ truth', abs(v[0] - truth) < max(4 * e[0], 0.004), f'{v[0]:.4f}+-{e[0]:.4f} vs {truth:.4f}')


def test_pt_differential_3d():
    """Two-subevent v2{2}(pT): POI in each pT bin vs the integrated reference."""
    fx = _fx()
    edges = [0.2, 0.6, 1.0, 1.4, 2.0]
    for species, slot in (('p + pbar', PROTON), ('charged', None)):
        rec = record(species=species)
        rec['dimensionality'] = 'differential_pT'
        rec['kinematic_range']['pT'] = {'min': edges[0], 'max': edges[-1], 'bins': edges}
        check(f'{species} v2(pT) key', resolve(rec)['compute'] == 'vn2_gap_poi')
        v, _ = evaluate(rec, grid=fx['ms'], cent_method='simple', with_jackknife=False)
        ref = [particle_level(fx['events'], poi_slot=slot, poi_pt=(lo, hi)) for lo, hi in zip(edges, edges[1:])]
        check(f'3D {species} v2(pT) == particle level', np.allclose(v, ref, atol=1e-9), f'{np.round(v, 4)} vs {np.round(ref, 4)}')


def test_overlapping_subevents_refused_3d():
    fx = _fx()
    rec = record(sub_a=(-0.6, 1.0), sub_b=(-1.0, 0.0))
    params = extract_measurement_params(rec)
    entry = get_pipeline_entry('vn2_gap_poi')
    try:
        entry['prepare_data_3d'](fx['ms'], params)
    except ValueError as err:
        check('overlap refused', 'disjoint' in str(err), str(err))
    else:
        check('overlap refused', False, 'no error')


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
