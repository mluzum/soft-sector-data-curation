#!/usr/bin/env python3
"""Curate CMS v2, v3 in OO and NeNe at 5.36 TeV, INSPIRE 3062822.

CMS Collaboration, "Observation of long-range collective flow in OO and
NeNe collisions and implications for nuclear structure", arXiv:2510.02580.

From the paper:
- centrality from the HF E_T sum (Glauber fit from 80%);
- charged particles |eta| < 2.4, 0.3 < pT < 3 GeV (trigger and associated);
- v_n{2, |Delta eta| > 2} = sqrt(V_n Delta) from the two-particle
  correlation function averaged over |Delta eta| > 2 (a pair gap);
  "sub": with the dijet correlations of 70-80% subtracted;
- v2{4} from Q-cumulants in the same acceptance;
- errors: statistical and systematic (none for the "sub" values, drawn as
  lines).

Records (OO and NeNe; classes 0-1, 1-2, 2-5, 5-10, then 5% to 50%, from the
tabulated centres):
- v2{2, |Delta eta| > 2}, v3{2, |Delta eta| > 2}: cumulant order 2 in
  |eta| < 2.4, the pair gap recorded as eta_gap (not computed, as the
  ATLAS 2967110 2PC records);
- v2{4};
- the NeNe / OO ratios of the three (system ratios).
Staged (--staged OUT): the dijet-subtracted v2, v3 and their ratios (no
uncertainties: a chi^2 cannot weight them).

Run from the repository root:  python scripts/curate_cms_vn_oo_nene_5360.py [--staged OUT]
"""
import argparse
import copy
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins3062822/HEPData-ins3062822-v*-yaml')[0]
VERSION = int(re.search(r'-v(\d+)-yaml', RAW).group(1))
CITE = 'CMS Collaboration, arXiv:2510.02580'
OUT = 'data/flow_vn/cms_flow_vn_OO_NeNe_5360gev_ins3062822.yaml'
OUT_RATIOS = 'data/observable_ratios/cms_observable_ratios_vn_NeNe_over_OO_5360gev_ins3062822.yaml'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
SYS = {'data1.yaml': ('O+O', 'O O --> CHARGED X'), 'data2.yaml': ('Ne+Ne', 'NE NE --> CHARGED X')}
KR = {'eta': {'min': -2.4, 'max': 2.4}, 'pT': {'min': 0.3, 'max': 3.0}}


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def classes(t):
    out, prev = [], 0.0
    for v in t['independent_variables'][0]['values']:
        hi = 2 * float(v['value']) - prev
        out.append(f'{prev:g}-{hi:g}%')
        prev = hi
    return out


def kind(header):
    """(harmonic, order, subtracted) of a column header."""
    n = int(re.search(r'v_\{(\d)\}', header).group(1))
    order = 4 if r'\{4\}' in header else 2
    return n, order, 'subtracting' in header


def what(n, order, sub):
    return (f'v{n}{{4}}' if order == 4 else f'v{n}{{2, |Delta eta| > 2}}') + (', dijet-subtracted' if sub else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged, index = [], [], {}
    for hf, (label, reaction) in SYS.items():
        t = load(hf)
        cl = classes(t)
        for i, dv in enumerate(t['dependent_variables']):
            n, order, sub = kind(dv['header']['name'])
            w = what(n, order, sub)
            d = {'observable_type': 'flow_vn', 'inspire_id': '3062822', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{w} vs centrality, {label} ({hf})',
                 'collision_system': {'energy_gev': 5360.0, 'reaction': reaction},
                 'particle_id': {'species': 'charged'}, 'value_type': dv['header']['name'],
                 'centrality': {'classes': cl, **copy.deepcopy(HF)}, 'kinematic_range': copy.deepcopy(KR),
                 'harmonic': n, 'method': 'cumulant', 'cumulant_order': order, 'dimensionality': 'integrated'}
            if order == 2:
                d['eta_gap'] = 2.0
            d['notes'] = {'value': f'{CITE}, Fig. 1. {w} vs centrality, charged particles |eta| < 2.4, 0.3 < pT < 3 '
                                   f'GeV, {label} 5.36 TeV; HF E_T centrality.'
                                   + (' Two-particle correlation function averaged over |Delta eta| > 2 (pair gap, '
                                      'recorded as eta_gap, not computed).' if order == 2 else ' Q-cumulants.')
                                   + (' Dijet correlations of 70-80% subtracted.' if sub else '')
                                   + (' No uncertainties (a line in the figure).' if sub else ' Errors: stat and sys.')}
            if sub:
                d['staging_reason'] = 'dijet-subtracted values without uncertainties (a line in Fig. 1)'
                staged.append(d)
            else:
                index[(n, order, label)] = len(public)
                public.append(d)
    ratios = []
    for hf in ('data3.yaml', 'data4.yaml'):
        t = load(hf)
        for i, dv in enumerate(t['dependent_variables']):
            n, order, sub = kind(dv['header']['name'])
            w = what(n, order, sub)
            r = {'observable_type': 'observable_ratios', 'inspire_id': '3062822', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{w} NeNe / OO vs centrality ({hf})',
                 'collision_system': {'energy_gev': 5360.0, 'reaction': 'Ne+Ne and O+O'},
                 'centrality': {'classes': classes(t), **copy.deepcopy(HF)}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': copy.deepcopy(KR), 'observable': f'v{n}', 'value_type': dv['header']['name'],
                 'ratio_type': 'system_ratio', 'numerator': 'Ne+Ne', 'denominator': 'O+O',
                 'numerator_system': 'Ne+Ne', 'denominator_system': 'O+O',
                 'notes': {'value': f'{CITE}, Fig. 2. {w} in NeNe over OO vs centrality.'
                                    + (' No uncertainties.' if sub else ' Errors: stat and sys.')}}
            if sub:
                r['staging_reason'] = 'dijet-subtracted ratio without uncertainties (a line in Fig. 2)'
                staged.append(r)
            else:
                r['numerator_record'] = f'{OUT}#{index[(n, order, "Ne+Ne")]}'
                r['denominator_record'] = f'{OUT}#{index[(n, order, "O+O")]}'
                ratios.append(r)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    with open(OUT_RATIOS, 'w') as fh:
        yaml.safe_dump_all(ratios, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {OUT_RATIOS}: {len(ratios)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} staged records')


if __name__ == '__main__':
    main()
