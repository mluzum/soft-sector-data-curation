#!/usr/bin/env python3
"""Curate ALICE phi -> mu mu spectra at forward rapidity in Pb-Pb at 2.76 TeV, INSPIRE 1669805.

S. Acharya et al. (ALICE), "phi meson production at forward rapidity in
Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", EPJC 78 (2018) 559,
arXiv:1804.08906.

From the paper:
- phi -> mu+ mu- in the muon spectrometer, 2.5 < y < 4, 2 < pT < 5 GeV/c;
- centrality from the V0 amplitude (V0A 2.8 < eta < 5.1, V0C -3.7 < eta <
  -1.7; Glauber fit), 0-90%;
- Table 3 (HEPData Tables 2, 3): the phi yield d2N/(dy dpT) in 0-40% and
  40-90%; errors stat, bin-to-bin uncorrelated and correlated sys.

Records (public): the two Pb-Pb spectra.  Staged (--staged OUT): the pp
spectrum (no centrality), dN/dy per <N_part> (Glauber) and R_AA.

Run from the repository root:  python scripts/curate_alice_phi_forward_pbpb.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1669805/HEPData-ins1669805-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), EPJC 78 (2018) 559, arXiv:1804.08906'
OUT = 'data/particle_spectra/alice_particle_spectra_phi_forward_PbPb_2760gev_ins1669805.yaml'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}],
       'notes': 'V0 amplitude (V0A and V0C), Glauber fit.'}
CLASSES = {'Table_2.yaml': '0-40%', 'Table_3.yaml': '40-90%'}
ERRS = ('stat', 'sys,bin-to-bin uncorrelated', 'sys,bin-to-bin correlated')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if hf not in CLASSES:
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'nuclear_modification' if hf == 'Table_5.yaml' else
                               'particle_spectra' if hf == 'Table_1.yaml' else 'particle_yields',
                               'inspire_id': '1669805', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc.strip()}'},
                               'staging_reason': ('pp: no centrality classes' if hf == 'Table_1.yaml' else
                                                  'dN/dy per <N_part> (Glauber)' if hf == 'Table_4.yaml' else
                                                  'R_AA: needs the pp reference')})
            continue
        cls = CLASSES[hf]
        assert cls.replace('%', '\\%') in desc, desc
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        dv = t['dependent_variables'][0]
        q = {k['name']: k['value'] for k in dv['qualifiers']}
        assert q['ABS(YRAP)'] == '2.5-4.0' and q['RE'] == 'PB PB --> PHI < MU+ MU- >', q
        assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {ERRS}
        public.append({
            'observable_type': 'particle_spectra', 'inspire_id': '1669805', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'phi d2N/(dy dpT), 2.5 < y < 4, Pb-Pb 2.76 TeV {cls} ({hf})',
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> PHI X'},
            'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': 'phi'},
            'kinematic_range': {'y': {'min': 2.5, 'max': 4.0},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
            'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy',
            'notes': {'value': f'{CITE}, Table 3. The phi yield d2N/(dy dpT) (from phi -> mu+ mu-; the phi yield, '
                               f'not times the branching ratio), 2.5 < y < 4, Pb-Pb 2.76 TeV {cls}. Errors: stat, '
                               'bin-to-bin uncorrelated and correlated sys.'}})
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
