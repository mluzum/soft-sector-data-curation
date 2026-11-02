#!/usr/bin/env python3
"""Curate ALICE v2 relative to the spectator plane, Pb-Pb 2.76 and Xe-Xe 5.44 TeV, INSPIRE 2070420.

S. Acharya et al. (ALICE), "Elliptic flow of charged particles at
midrapidity relative to the spectator plane in Pb-Pb and Xe-Xe collisions",
PLB 846 (2023) 137453, arXiv:2204.10240.

From the paper (Sec. 3):
- charged particles |eta| < 0.8, pT > 0.2 GeV/c; centrality from the V0
  (V0M), 5-70%;
- v2{Psi_SP}: relative to the neutron spectator plane (ZDC A and C,
  |eta| > 8.8);
- v2{2, |Delta eta| > 1} (pair gap: minimum pseudorapidity separation 1)
  and v2{4}: multi-particle cumulants; pT-differential v2{4} from the
  differential cumulants;
- errors: stat and sys (the ZDC scale uncertainty separately).

Records (public): v2{2, |Delta eta| > 1} and v2{4} vs centrality in Pb-Pb
2.76 TeV (5% classes) and Xe-Xe 5.44 TeV (Figs. 2a, b), v2{4}(pT) in Pb-Pb
(Fig. 3b, reference pT > 0.2).  The pair gap is stated as two subevents
-0.8 < eta < -0.5 and 0.5 < eta < 0.8 (as for ALICE 900651; equal for a
boost-invariant model, close otherwise).  Staged (--staged OUT):
v2{Psi_SP} vs centrality and pT (flow_reference plane spectator: models have
no spectators).  Not curated: the ratios to v2{4} (Figs. 2c, d, 3c), the
slopes of their linear fits (Fig. 3d) and v2/epsilon2 (Fig. 4: model
eccentricities).  The 5 old records filed Xe-Xe at 5.02 TeV and computed
v2{Psi_SP} and ratios as v2{2}.

Run from the repository root:
    python scripts/curate_alice_v2_spectator_plane.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2070420/HEPData-ins2070420-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PLB 846 (2023) 137453, arXiv:2204.10240'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
PT = {'min': 0.2, 'max': 100.0}
ETA = {'min': -0.8, 'max': 0.8}
ZDC = {'detector': 'ZDC', 'plane': 'spectator', 'eta': [{'min': -100.0, 'max': -8.77}, {'min': 8.78, 'max': 100.0}],
       'notes': 'First-order spectator plane of the neutron ZDCs (A and C sides, |eta| > 8.8).'}
SYSTEMS = {'figure_2.1.yaml': ('PB PB', 2760.0), 'figure_2.2.yaml': ('XE XE', 5440.0),
           'figure_3.1.yaml': ('PB PB', 2760.0), 'figure_3.2.yaml': ('PB PB', 2760.0)}
OLD = 'data/flow_vn/alice_flow_vn_charged_particles_XeXe_5020gev.yaml'
SUB = {'particles': 'AB', 'eta_gap': 1.0,
       'subevents': {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.5}, 'pT': dict(PT)}},
                     'B': {'kinematic_range': {'eta': {'min': 0.5, 'max': 0.8}, 'pT': dict(PT)}}}}


def kind(h):
    if 'Psi' in h:
        return 'SP'
    if 'Delta' in h:
        return 'gap'
    return '4'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, (beams, energy) in SYSTEMS.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        differential = iv['header']['name'] == 'PT'
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            k = kind(h)
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            if differential:
                cls = q['CENTRALITY'] + '%'
                pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
                classes, dim = [cls], 'differential_pT'
                kr = {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}}
                where = f'vs pT, {cls}'
            else:
                classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
                dim, kr, where = 'integrated', {'eta': dict(ETA), 'pT': dict(PT)}, 'vs centrality, pT > 0.2 GeV/c'
            name = {'SP': 'v2{Psi_SP}', 'gap': 'v2{2, |Delta eta| > 1}', '4': 'v2{4}'}[k]
            d = {'observable_type': 'flow_vn', 'inspire_id': '2070420', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{name} {where}, {beams.title()} {energy:g} GeV ({hf})',
                 'collision_system': {'energy_gev': energy, 'reaction': f'{beams} --> CHARGED X'},
                 'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': kr, 'harmonic': 2, 'value_type': h.strip('$'), 'dimensionality': dim}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if k == 'SP':
                d.update({'method': 'scalar_product', 'flow_reference': dict(ZDC),
                          'staging_reason': 'v2 relative to the neutron spectator plane (ZDC): models have no '
                                            'spectators'})
            elif k == 'gap':
                d.update({'method': 'cumulant', 'cumulant_order': 2, **SUB})
            else:
                d.update({'method': 'cumulant', 'cumulant_order': 4})
                if differential:
                    d['flow_reference'] = {'pT': dict(PT)}
            d['notes'] = {'value': f'{CITE}. {name} of charged particles {where}, |eta| < 0.8, {beams.title()} '
                                   f'{energy:g} GeV'
                                   + ('; the pair gap stated as subevents -0.8 < eta < -0.5, 0.5 < eta < 0.8'
                                      if k == 'gap' else '')
                                   + ('; reference particles pT > 0.2 GeV/c' if k == '4' and differential else '')
                                   + '. Errors: stat and sys (ZDC scale uncertainty not included).'}
            (staged if k == 'SP' else public).append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '2070420' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/alice_flow_vn_v2_PbPb_XeXe_ins2070420.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['collision_system']['energy_gev'], d['dimensionality'], d['cumulant_order']) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
