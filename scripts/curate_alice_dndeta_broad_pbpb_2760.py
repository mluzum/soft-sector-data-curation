#!/usr/bin/env python3
"""Curate ALICE dNch/deta over a broad eta range in peripheral Pb-Pb at 2.76 TeV, INSPIRE 1394676.

J. Adam et al. (ALICE), "Centrality evolution of the charged-particle
pseudorapidity density over a broad pseudorapidity range in Pb-Pb
collisions at sqrt(sNN) = 2.76 TeV", PLB 754 (2016) 373, arXiv:1509.07299.

From the paper and tables:
- dNch/deta in -3.5 < eta < 5.0 (0.25 bins) from the SPD and FMD,
  combined and symmetrised; centrality from the V0 amplitude (V0M),
  classes 30-40 ... 80-90%;
- errors: systematic (the statistical ones negligible); a centrality-
  determination uncertainty per class (1.2-4.9%) quoted separately.

Records (public): dNch/deta(eta) per class (Table 1, dimensionality
differential_eta; computed on 3D containers).  Not curated: the widths
(Table 2), dNch/deta per participant pair vs N_part (Table 3) and the
extrapolated total multiplicity (Table 4), which need Glauber N_part or a
fit.  The 2 old records gave one dN/dy value per class over |eta| < 0.8
and an unknown reaction.

Run from the repository root:  python scripts/curate_alice_dndeta_broad_pbpb_2760.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1394676/HEPData-ins1394676-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), PLB 754 (2016) 373, arXiv:1509.07299'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
OLD = 'data/particle_yields/alice_particle_yields_centrality_evolution_of_the_charged-particle_2760gev.yaml'


def main():
    public = []
    t = yaml.safe_load(open(os.path.join(RAW, 'Table1.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_yields', 'inspire_id': '1394676', 'version': VERSION,
             'hepdata_file': 'Table1.yaml', 'dv_index': i, 'hepdata_table_label': f'dNch/deta(eta), {cls} (Table1.yaml)',
             'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
             'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
             'notes': {'value': f'{CITE}, Table 1. dNch/deta of charged particles vs eta, {cls}, combined SPD and '
                                f"FMD, symmetrised. Errors: systematic (the centrality uncertainty {q.get('Centrality', '')} "
                                'quoted separately, not included).'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1394676' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/particle_yields/alice_particle_yields_dNch_deta_eta_PbPb_2760gev_ins1394676.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')


if __name__ == '__main__':
    main()
