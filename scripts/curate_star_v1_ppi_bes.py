#!/usr/bin/env python3
"""Curate STAR directed flow of protons, antiprotons and pions in Au+Au at 7.7-200 GeV, INSPIRE 1277069.

L. Adamczyk et al. (STAR), "Beam-energy dependence of the directed flow of
protons, antiprotons, and pions in Au+Au collisions", PRL 112 (2014)
162301, arXiv:1401.3043.

From the paper:
- rapidity-odd v1(y) relative to the first-order event plane of the two
  Beam-Beam Counters (inner tiles 3.3 < |eta| < 5.0) at 7.7-39 GeV, and
  of the ZDC-SMD (spectator neutrons) at 62.4 and 200 GeV, where the BBC
  resolution is inadequate;
- centrality from the charged multiplicity in |eta| < 0.5 (RefMult);
- TPC tracks (|eta| < 1.0), pT > 0.2 GeV/c; protons and antiprotons at
  0.4 < pT < 2.0 GeV/c, pions at pT > 0.2 GeV/c and p < 1.6 GeV/c
  (dE/dx and TOF); the 10-40% (anti)proton values weighted for the
  pT-dependent efficiency;
- sign: "The established convention assigns a positive sign to v1 for
  nucleons detected near beam rapidity" on the positive-rapidity side.
  A model without spectators has no such nucleons, so the records anchor
  the sign on pions: v1 < 0 at 0 < y < 1 (pT 0.2-1.6 GeV/c), which every
  pion table here obeys at every energy and centrality (>= 5 sigma; the
  paper: pion slopes negative at all energies);
- errors: statistical (systematic uncertainties are given for the slopes
  only).

Records (public, 7.7-39 GeV, BBC plane, computed with a charged odd
reference in the BBC acceptance and the scalar product, the event-plane
result at low resolution): v1(y) of protons in 0-10, 10-40, 40-80%
(Table 1; its 10-40% columns are those of Table 3), pi- in 0-10 and
40-80% (Table 2), antiprotons (Table 3) and pi+, pi- (Table 4) in 10-40%.
Table 2's 10-40% pi- differs from Table 4's (Fig. 2, the slopes' input)
by up to 11% of the largest |v1| (4 statistical sigma; the paper does
not say why): Table 4 is curated.  y bins of 0.2 around the tabulated
points (0.1, 0.3, ...).  Staged (--staged OUT): 62.4 and 200 GeV (ZDC-SMD spectator
plane) and the dv1/dy slopes (Tables 5, 6: fits to v1(y), not computed).

Run from the repository root:
  python scripts/curate_star_v1_ppi_bes.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1277069/HEPData-ins1277069-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'L. Adamczyk et al. (STAR), PRL 112 (2014) 162301, arXiv:1401.3043'
OUT = 'data/flow_vn/star_flow_vn_v1_ppi_AuAu_7.7_39gev_ins1277069.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'charged multiplicity in |eta| < 0.5 (RefMult).'}
BBC = [{'min': -5.0, 'max': -3.3}, {'min': 3.3, 'max': 5.0}]
PION_SIGN = {'species': 'pi+ + pi-', 'y': {'min': 0.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 1.6},
             'sign': 'negative'}
SIGN_NOTE = ('Sign: STAR takes v1 positive for nucleons near beam rapidity on the positive side; a model without '
             'spectators has none, so the sign is anchored on pions, v1 < 0 at 0 < y < 1 (STAR 1277069: every '
             'energy and centrality, >= 5 sigma).')
BES = (7.7, 11.5, 19.6, 27.0, 39.0)
# species -> (reaction product, kinematic_range without y, cut note)
TPC = {'eta': {'min': -1.0, 'max': 1.0}}
KIN = {'p': ('P', {**TPC, 'pT': {'min': 0.4, 'max': 2.0}}, 'TPC tracks |eta| < 1, 0.4 < pT < 2.0 GeV/c'),
       'pbar': ('PBAR', {**TPC, 'pT': {'min': 0.4, 'max': 2.0}}, 'TPC tracks |eta| < 1, 0.4 < pT < 2.0 GeV/c'),
       'pi+': ('PI+', {**TPC, 'pT': {'min': 0.2, 'max': 1.6}, 'p': {'max': 1.6}},
               'TPC tracks |eta| < 1, pT > 0.2 GeV/c, p < 1.6 GeV/c'),
       'pi-': ('PI-', {**TPC, 'pT': {'min': 0.2, 'max': 1.6}, 'p': {'max': 1.6}},
               'TPC tracks |eta| < 1, pT > 0.2 GeV/c, p < 1.6 GeV/c')}
NAMES = {'protons': 'p', 'proton': 'p', 'anti-proton': 'pbar', '$\\pi^{+}$': 'pi+', '$\\pi^{-}$': 'pi-'}


def y_bins(centres):
    """Contiguous bins from y = 0 outward, each tabulated point its bin's
    centre (STAR's points: 0.1, 0.3, ... -> edges 0, 0.2, 0.4, ...)."""
    pos = sorted(c for c in centres if c > 0)
    assert sorted(-c for c in centres if c < 0) == pos and len(pos) * 2 == len(centres), centres
    edges = [0.0]
    for c in pos:
        edges.append(round(2 * c - edges[-1], 6))
        assert edges[-1] > c, (centres, edges)
    full = sorted({-e for e in edges} | set(edges))
    return [round(e, 6) for e in full]


def bbc_reference(energy):
    if energy <= 39.0:
        return {'odd': True, 'detector': 'BBC', 'eta': BBC, 'pT': {'min': 0.0, 'max': 100.0},
                'sign_convention': dict(PION_SIGN),
                'notes': 'First-order event plane of the two BBCs (inner tiles 3.3 < |eta| < 5.0), east and west '
                         'with opposite signs; computed as the scalar product with a charged odd reference there.'}
    return {'odd': True, 'detector': 'ZDC-SMD', 'plane': 'spectator', 'sign_convention': dict(PION_SIGN),
            'notes': 'First-order spectator-neutron plane of the ZDC-SMDs (|eta| > 6.3).'}


def record(inspire, version, cite, hf, i, label, energy, cls, species, kin, prod, cut, edges, note,
           errors='statistical'):
    kr = {k: dict(v) for k, v in kin.items()}
    kr['y'] = {'min': edges[0], 'max': edges[-1], 'bins': edges}
    d = {'observable_type': 'flow_vn', 'inspire_id': inspire, 'version': version, 'hepdata_file': hf,
         'dv_index': i, 'hepdata_table_label': label,
         'collision_system': {'energy_gev': energy, 'reaction': f'AU AU --> {prod} X'},
         'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': species},
         'kinematic_range': kr, 'harmonic': 1, 'method': 'event_plane', 'value_type': 'v1',
         'dimensionality': 'differential_y', 'flow_reference': bbc_reference(energy),
         'notes': {'value': f'{cite}. Rapidity-odd v1(y) of {species}, Au+Au {energy:g} GeV {cls}; {cut}. '
                            f'{note}y bins contiguous from y = 0, centred on the tabulated points. '
                            f'{SIGN_NOTE} Errors: {errors}.'}}
    if energy > 39.0:
        d['staging_reason'] = ('v1 relative to the ZDC-SMD spectator plane (62.4 and 200 GeV): models have no '
                               'spectators')
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for tab in (1, 2, 3, 4):
        hf = f'Table{tab}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        centres = [float(v['value']) for v in t['independent_variables'][0]['values']]
        edges = y_bins(centres)
        for i, dv in enumerate(t['dependent_variables']):
            q = {x['name']: x['value'] for x in dv['qualifiers']}
            cls = q['CENTRALITY'].replace('$\\%$', '%')
            energy = float(q['SQRT(sNN)'])
            m = re.fullmatch(r'\$v_\{1\}\$ for (.+)', dv['header']['name'])
            species = NAMES[m.group(1)]
            assert [e['label'] for e in dv['values'][0]['errors']] == ['stat.'], (hf, i)
            if tab == 1 and cls == '10-40%':             # the same values as Table 3's protons
                t3 = yaml.safe_load(open(os.path.join(RAW, 'Table3.yaml')))
                twin = [d for d in t3['dependent_variables'] if d['header']['name'].endswith('proton')
                        and float(d['qualifiers'][1]['value']) == energy][0]
                assert [v['value'] for v in twin['values']] == [v['value'] for v in dv['values']], energy
            if (tab == 3 and species == 'p' and energy <= 39.0) or (tab == 2 and cls == '10-40%'):
                continue                                  # Table 1's protons; Table 4's pi- (Fig. 2)
            prod, kin, cut = KIN[species]
            fig = {1: 'Fig. 1', 2: 'Fig. 1', 3: 'Fig. 2', 4: 'Fig. 2'}[tab]
            label = f'{fig}: v1(y) of {species}, Au+Au {energy:g} GeV, {cls} ({hf} column {i})'
            note = ('Efficiency-weighted within the acceptance. ' if species in ('p', 'pbar') and cls == '10-40%'
                    else '')
            d = record('1277069', VERSION, CITE, hf, i, label, energy, cls, species, kin, prod, cut, edges, note)
            vals = [float(v['value']) for v in dv['values']]
            if species.startswith('pi'):       # the anchor holds in the data
                assert sum(v for c, v in zip(centres, vals) if c > 0) < 0, (hf, i)
            (staged if 'staging_reason' in d else public).append(d)
    for tab in (5, 6):
        hf = f'Table{tab}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            staged.append({'observable_type': 'flow_vn', 'inspire_id': '1277069', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f'Fig. 3: {name} vs beam energy, 10-40% ({hf} column {i})',
                           'collision_system': {'reaction': 'AU AU --> X'},
                           'centrality': {'classes': ['10-40%'], **REFMULT}, 'harmonic': 1,
                           'value_type': 'dv1/dy', 'dimensionality': 'integrated',
                           'notes': {'value': f'{CITE}. {name}: linear term of a cubic fit to v1(y) (|y| < 0.5 at '
                                              '7.7-39 GeV, linear over |y| < 1 at 62.4 and 200 GeV), one value '
                                              'per beam energy. Errors: stat. and syst.'},
                           'staging_reason': 'slope of a fit to v1(y) vs beam energy: not computed (the v1(y) '
                                             'records are curated)'})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
