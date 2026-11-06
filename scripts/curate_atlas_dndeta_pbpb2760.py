#!/usr/bin/env python3
"""Curate ATLAS charged-particle dN/deta in Pb+Pb at 2.76 TeV, INSPIRE 925723.

ATLAS Collaboration, "Measurement of the centrality dependence of the
charged particle pseudorapidity distribution in lead-lead collisions at
sqrt(sNN) = 2.76 TeV with the ATLAS detector", PLB 710 (2012) 363,
arXiv:1108.6027.

From the paper:
- charged particles from pixel tracklets and pixel tracks, |eta| < 2,
  corrected down to pT ~ 0 (reconstruction from pT ~ 30 MeV);
- centrality from the FCal Sigma E_T, 3.2 < |eta| < 4.9: 10% classes
  over 0-80% for dN/deta(eta); 2% classes over 0-20% and 5% over 20-80%
  for the |eta| < 0.5 multiplicity;
- errors: stat and sys (dN/deta(eta)); one total error (|eta| < 0.5).

Records (public): dN/deta(eta) in eight classes (Tables 1, 2; 0.1 bins)
and dN/deta at |eta| < 0.5 vs 22 classes (Table 3).  Staged (--staged
OUT): <N_part> and dN/deta / (<N_part>/2) (Glauber-dependent).

Run from the repository root:  python scripts/curate_atlas_dndeta_pbpb2760.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins925723/HEPData-ins925723-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'ATLAS Collaboration, PLB 710 (2012) 363, arXiv:1108.6027'
OUT = 'data/particle_yields/atlas_particle_yields_dNch_deta_PbPb_2760gev_ins925723.yaml'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}],
        'notes': 'FCal Sigma E_T, 3.2 < |eta| < 4.9.'}
SYSTEM = {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'}


def head(hf, i, label, classes):
    return {'observable_type': 'particle_yields', 'inspire_id': '925723', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})', 'collision_system': dict(SYSTEM),
            'centrality': {'classes': classes, **FCAL}, 'particle_id': {'species': 'charged'}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf in ('Table1.yaml', 'Table2.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        assert bins[0][0] == -2.0 and bins[-1][1] == 2.0 and len(bins) == 40, bins
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            assert q['RE'] == 'PB PB --> CHARGED X' and float(q['sqrts/NUCLEON']) == 2760.0, q
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}
            d = head(hf, i, f'dNch/deta(eta), {cls}', [cls])
            d['kinematic_range'] = {'eta': {'min': -2.0, 'max': 2.0, 'bin_ranges': bins},
                                    'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='dNch_deta', value_type='dNch_deta', dimensionality='differential_eta')
            d['notes'] = {'value': f'{CITE} (HEPData {hf}). dNch/deta of charged particles vs eta (0.1 bins), Pb+Pb '
                                   f'2.76 TeV {cls}, corrected to all pT. Errors: stat and sys.'}
            public.append(d)
    hf = 'Table3.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    classes = [f"{v['low']:g}-{v['high']:g}%" for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name'].strip()
        q = {k['name']: k['value'] for k in dv['qualifiers']}
        assert q['ETARAP'] == '-0.5-0.5', q
        if name != 'D(N)/DETARAP':
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '925723', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{name} ({hf} column {i})',
                           'value_type': name, 'notes': {'value': f'{CITE}.'},
                           'staging_reason': '<N_part> or dN/deta per participant pair: Glauber-dependent'})
            continue
        d = head(hf, i, 'dNch/deta at |eta| < 0.5 vs centrality', classes)
        d['kinematic_range'] = {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
        d.update(observable='dNch_deta', value_type='dNch_deta')
        d['notes'] = {'value': f'{CITE}, Table 2 (HEPData {hf}). dNch/deta of charged particles at |eta| < 0.5 vs '
                               'centrality (2% classes over 0-20%, 5% over 20-80%), Pb+Pb 2.76 TeV, all pT. '
                               'Errors: one component (unlabelled, total).'}
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
