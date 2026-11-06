#!/usr/bin/env python3
"""Curate ALICE phi production in p-Pb at 8.16 TeV, INSPIRE 1946970.

S. Acharya et al. (ALICE), "K*(892)0 and phi(1020) production in p-Pb
collisions at sqrt(sNN) = 8.16 TeV", PRC 106 (2022) 034907,
arXiv:2110.10042 (the 5.02 TeV analysis: scripts/
curate_alice_phi_kstar_ppb_5020.py).

From the paper:
- the Pb beam toward +z, the proton toward -z: Delta y = -0.465 (y_lab =
  y_cms - 0.465);
- seven V0A multiplicity classes (amplitude, 2.8 < eta_lab < 5.1, the
  Pb-going side): 0-5, 5-10, 10-20, 20-40, 40-60, 60-80, 80-100%; NSD
  spectra besides;
- phi -> K+ K- and (K*0 + K*0bar)/2 at -0.5 < y_cms < 0, d2N/(dpT dy) per
  event, phi up to 16 GeV/c;
- errors: stat and sys.

Records (public): phi spectra per class (Tables 10-16, dN_dpT_dy), in the
nn_cms frame (the V0A window at 3.265 < eta_cm < 5.565).  Staged
(--staged OUT): K*0 (a resonance the pipeline does not provide: Tables
1-8), the NSD phi spectrum (Table 9: no class), R_pPb (Tables 17-22).

Run from the repository root:  python scripts/curate_alice_phi_ppb_8160.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1946970/HEPData-ins1946970-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PRC 106 (2022) 034907, arXiv:2110.10042'
OUT = 'data/particle_spectra/alice_particle_spectra_phi_pPb_8160gev_ins1946970.yaml'
SHIFT = -0.465
V0A_CMS = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
           'eta': [{'min': round(2.8 - SHIFT, 3), 'max': round(5.1 - SHIFT, 3)}],
           'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side), here in the nn_cms frame.'}
CS = {'energy_gev': 8160.0, 'frame': 'nn_cms', 'rapidity_shift': SHIFT, 'forward_beam': 'Pb'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        n = int(re.fullmatch(r'Table(\d+)\.yaml', hf).group(1))
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if n > 16:
            staged.append({'observable_type': 'nuclear_modification', 'inspire_id': '1946970', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': 0, 'hepdata_table_label': f'{desc[:100]} ({hf})',
                           'value_type': 'R_pPb', 'notes': {'value': f'{CITE}. {desc}'},
                           'staging_reason': 'R_pPb (needs the pp reference and <T_pPb>): not curated here'})
            continue
        dv = t['dependent_variables'][0]
        q = {k['name']: k['value'] for k in dv['qualifiers']}
        assert q['SQRT(S)/NUCLEON'] == '8160.0' and q['YRAP(C=CM)'] == '-0.5-0', q
        kstar = n <= 8
        assert ('K^{*0}' in desc) == kstar and (('phi' in desc) != kstar), desc
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}, hf
        cls = None if q['class'] == 'NSD' else \
            '-'.join(f'{float(x):g}' for x in q['class'].split('-')) + '%'
        species, prod = ('(K*0 + K*0bar)/2', 'K*0') if kstar else ('phi', 'PHI')
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1946970', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'd2N/(dpT dy) of {species}, {cls or "NSD"} ({hf})',
             'collision_system': dict(CS, reaction=f'P PB --> {prod} X'),
             'centrality': {'classes': [cls] if cls else ['NSD'], **V0A_CMS},
             'particle_id': {'species': 'K*0' if kstar else 'phi'},
             'kinematic_range': {'y': {'min': -0.5, 'max': 0.0},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy',
             'notes': {'value': f'{CITE}. 1/N_ev d2N/(dpT dy) of {species}, {cls or "NSD"} (V0A), -0.5 < y_cms < 0, '
                                'p-Pb 8.16 TeV. Errors: stat and sys.'}}
        if kstar:
            d['staging_reason'] = 'K*0: a resonance the pipeline does not provide'
            staged.append(d)
        elif cls is None:
            d['staging_reason'] = 'NSD spectrum: no event class'
            staged.append(d)
        else:
            public.append(d)
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
