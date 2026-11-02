#!/usr/bin/env python3
"""Curate PHENIX truncated <pT> and p/h in Au+Au at 130 GeV, INSPIRE 590820 (staged only).

K. Adcox et al. (PHENIX), "Centrality dependence of the high pT charged
hadron suppression in Au+Au collisions at sqrt(sNN) = 130 GeV", PLB 561
(2003) 82, nucl-ex/0207009 (PPG013).

From the paper:
- charged hadrons in the central arms with the fiducial cut |eta| < 0.18;
- centrality from the BBC and ZDC, six exclusive classes 0-5, 5-15,
  15-30, 30-60, 60-80, 80-92%;
- the truncated mean <pT> - pT,min above 0.5 and 2 GeV/c (MeV/c);
  p/h = (p + pbar)/h vs pT (minimum bias) and above 1.8 GeV/c vs
  centrality;
- errors: stat.

Every record is staged (--staged OUT): the truncated mean (Fig. 4: the
offset by pT,min is not computed) and p/h (Fig. 3: a particle ratio,
minimum bias for the pT dependence; HEPData labels the Fig. 3b column
'$p_T$' and Fig. 3a 'h/p').  Not curated: the Glauber table (Fig. 2), R_AA
(Figs. 5, 6: nuclear modification).  The old public record was at 200 GeV
and read the truncated mean as <pT> in |eta| < 0.8.

Run from the repository root:  python scripts/curate_phenix_truncated_pt_auau130.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins590820/HEPData-ins590820-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'K. Adcox et al. (PHENIX), PLB 561 (2003) 82, nucl-ex/0207009'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge with the ZDC energy.'}
ETA = {'min': -0.18, 'max': 0.18}
SYSTEM = {'energy_gev': 130.0, 'reaction': 'AU AU --> CHARGED X'}
OLD = 'data/mean_pT/phenix_mean_pT_centrality_dependence_of_the_high_AuAu_200gev.yaml'


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def classes_of(iv):
    return [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    t = load('ppg013fig4.yaml')
    cls = classes_of(t['independent_variables'][0])
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        ptmin = float(re.search(r'[\d.]+', q['$p^{min}_T$']).group())
        staged.append({'observable_type': 'mean_pT', 'inspire_id': '590820', 'version': VERSION,
                       'hepdata_file': 'ppg013fig4.yaml', 'dv_index': i,
                       'hepdata_table_label': f'truncated <pT> above {ptmin:g} GeV/c vs centrality (ppg013fig4.yaml)',
                       'collision_system': dict(SYSTEM), 'centrality': {'classes': cls, **BBC},
                       'particle_id': {'species': 'charged'},
                       'kinematic_range': {'eta': dict(ETA), 'pT': {'min': ptmin, 'max': 100.0}},
                       'observable': 'mean_pT', 'value_type': dv['header']['name'],
                       'notes': {'value': f'{CITE}, Fig. 4. Truncated mean <pT> - pT,min of charged hadrons above '
                                          f'{ptmin:g} GeV/c, in MeV/c. Errors: stat.'},
                       'staging_reason': 'truncated mean <pT> - pT,min: the offset is not computed'})
    t = load('ppg013fig3a.yaml')
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
    staged.append({'observable_type': 'particle_ratios', 'inspire_id': '590820', 'version': VERSION,
                   'hepdata_file': 'ppg013fig3a.yaml', 'dv_index': 0,
                   'hepdata_table_label': 'p/h vs pT, minimum bias (ppg013fig3a.yaml)', 'collision_system': dict(SYSTEM),
                   'centrality': {'classes': ['0-92%'], **BBC}, 'particle_id': {'species': 'p + pbar'},
                   'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1],
                                                                'bin_ranges': pairs}},
                   'observable': 'p/h', 'value_type': dv['header']['name'], 'numerator': 'p + pbar',
                   'denominator': 'charged', 'dimensionality': 'differential_pT',
                   'notes': {'value': f"{CITE}, Fig. 3a. (p + pbar)/h vs pT, minimum bias (HEPData header 'h/p'). "
                                      'Errors: stat.'},
                   'staging_reason': 'particle ratio, minimum bias: not computed'})
    t = load('ppg013fig3b.yaml')
    iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
    staged.append({'observable_type': 'particle_ratios', 'inspire_id': '590820', 'version': VERSION,
                   'hepdata_file': 'ppg013fig3b.yaml', 'dv_index': 0,
                   'hepdata_table_label': 'p/h above 1.8 GeV/c vs centrality (ppg013fig3b.yaml)',
                   'collision_system': dict(SYSTEM), 'centrality': {'classes': classes_of(iv), **BBC},
                   'particle_id': {'species': 'p + pbar'},
                   'kinematic_range': {'eta': dict(ETA), 'pT': {'min': 1.8, 'max': 100.0}},
                   'observable': 'p/h', 'value_type': dv['header']['name'], 'numerator': 'p + pbar',
                   'denominator': 'charged',
                   'notes': {'value': f"{CITE}, Fig. 3b. (p + pbar)/h above 1.8 GeV/c vs centrality (HEPData header "
                                      "'$p_T$'). Errors: stat."},
                   'staging_reason': 'particle ratio: not computed'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '590820' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    print(f'{len(staged)} staged records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
