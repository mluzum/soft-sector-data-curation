#!/usr/bin/env python3
"""Curate BRAHMS dNch/deta in Au+Au at 200 GeV, INSPIRE 567754.

I. G. Bearden et al. (BRAHMS), "Pseudorapidity distributions of charged
particles from Au+Au collisions at the maximum RHIC energy", PRL 88,
202301 (2002), nucl-ex/0112001.

From the paper:
- dNch/deta from the Si-strip array (SiMA, -3.0 <= eta <= 3.0) and the
  Beam-Beam Counters (2.1 <= |eta| <= 4.7);
- centrality from the total multiplicity of the corresponding array: the
  Multiplicity Array (SiMA and tiles, -3.0 <= eta <= 3.0) for the SiMA
  points, the BBC arrays for the BBC points (the two agree within 2% where
  both were analysed);
- errors: statistical in HEPData (< 1%); systematic 8% (SiMA) and 10%
  (BBC) in the paper, mostly an overall scale.

Records (public): dNch/deta(eta), 0-5 ... 40-50%, SiMA (Table 1a, 20 bins
of 0.3) and BBC (Table 1b, 16 bins of 0.4, overlapping), differential_eta
(computed on 3D containers).  Not curated: the symmetrized combination of
the same points (Table 3), the ratio to 130 GeV (Table 4: another paper's
data) and dN/deta per participant pair (Table 5, N_part from HIJING).  The
old record had the SiMA 0-5% distribution with pT up to 5e9 GeV/c and the
SiMA as the centrality estimator for every point.

Run from the repository root:  python scripts/curate_brahms_dndeta_200.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins567754/HEPData-ins567754-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'I. G. Bearden et al. (BRAHMS), PRL 88, 202301 (2002), nucl-ex/0112001'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -3.0, 'max': 3.0}],
      'notes': 'Total multiplicity of the Multiplicity Array (Si strips and tiles).'}
BBC = {'estimator': {'name': 'BBC', 'observable': 'Nch', 'detector': 'BBC'},
       'eta': [{'min': -4.7, 'max': -2.1}, {'min': 2.1, 'max': 4.7}],
       'notes': 'Total multiplicity of the Beam-Beam Counter arrays.'}
TABLES = {'tab1a.yaml': ('SiMA', MA, '8%'), 'tab1b.yaml': ('BBC', BBC, '10%')}
OLD = 'data/particle_yields/brahms_particle_yields_dNdetaprm_eta_AuAu_200gev_ins567754.yaml'
OUT = 'data/particle_yields/brahms_particle_yields_dNch_deta_AuAu_200gev_ins567754.yaml'
BLANK = ('', '-', '--', 'none')


def main():
    public = []
    for hf, (det, cent, sys_err) in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            cls = q['CENTRALITY'] + '%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'particle_yields', 'inspire_id': '567754', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'dNch/deta(eta), {det}, {cls} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': [cls], **cent}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': min(p[0] for p in pairs), 'max': max(p[1] for p in pairs),
                                             'bin_ranges': pairs},
                                     'pT': {'min': 0.0, 'max': 100.0}},
                 'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'dimensionality': 'differential_eta',
                 'notes': {'value': f'{CITE}, Fig. 1. dNch/deta of charged particles vs eta, {cls}, from the {det}; '
                                    f'centrality from the {det if det == "BBC" else "Multiplicity Array"} total '
                                    f'multiplicity. Errors: statistical (systematic {sys_err} in the paper, mostly '
                                    'an overall scale).'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '567754' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')


if __name__ == '__main__':
    main()
