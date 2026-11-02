#!/usr/bin/env python3
"""Curate CMS v2{2k} in PbPb at 5.02 TeV, INSPIRE 2724506.

A. Tumasyan et al. (CMS), "Higher-order moments of the elliptic flow
distribution in PbPb collisions at sqrt(sNN) = 5.02 TeV", JHEP 02 (2024)
106, arXiv:2311.11370.

From the paper:
- charged tracks |eta| < 2.4, 0.5 < pT < 3.0 GeV/c, efficiency-weighted;
- Q-cumulants v2{2k}, k = 1 ... 5, single acceptance (no gap), events
  weighted by the number of 2m-particle combinations;
- centrality from the HF energy (both HF calorimeters), 5% classes 5-60%;
- errors: stat and sys.

Records (public): v2{2}, v2{4}, v2{6}, v2{8} vs centrality (figure_2).
Staged (--staged OUT): v2{10} (no ten-particle cumulant), the relative
differences (figure_3), h_1, h_2 and their ratios (figures 4-5), the
skewness, kurtosis and superskewness (figure_6).  The 8 old records filed
each of figures 2-6 as v2{2}.

Run from the repository root:
    python scripts/curate_cms_v2_moments_pbpb_5020.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2724506/HEPData-ins2724506-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Tumasyan et al. (CMS), JHEP 02 (2024) 106, arXiv:2311.11370'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
OLD = 'data/flow_vn/cms_flow_vn_higher-order_moments_of_the_elliptic_PbPb_5020gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    public, staged = [], []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            assert '0.5 < p_' in q.get('Transverse momentum range', '') and '2.4' in q.get('Pseudorapidity range', ''), (hf, q)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
            m = re.fullmatch(r'\$v_2\\\{(\d+)\\\}\$', h)
            d = {'observable_type': 'flow_vn', 'inspire_id': '2724506', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{h} ({hf})",
                 'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': classes, **HF}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': 0.5, 'max': 3.0}},
                 'harmonic': 2, 'method': 'cumulant', 'value_type': h.strip('$'), 'dimensionality': 'integrated'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            order = int(m.group(1)) if m else None
            if order is not None:
                d['cumulant_order'] = order
            d['notes'] = {'value': f'{CITE}. {h.strip("$")} vs centrality, |eta| < 2.4, 0.5 < pT < 3 GeV/c, Q-cumulants '
                                   'without a gap. Errors: stat and sys.'}
            if order in (2, 4, 6, 8):
                public.append(d)
            else:
                d['staging_reason'] = ('v2{10}: no ten-particle cumulant' if order == 10 else
                                       'a ratio or moment derived from the cumulants (no computation)')
                staged.append(d)
    if os.path.exists(OLD):
        assert all(str(y['inspire_id']) == '2724506' for y in yaml.safe_load_all(open(OLD)) if y), OLD
        os.remove(OLD)
    out = 'data/flow_vn/cms_flow_vn_v2_2k_PbPb_5020gev_ins2724506.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [d['cumulant_order'] for d in public])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
