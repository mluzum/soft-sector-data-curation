#!/usr/bin/env python3
"""Curate PHENIX charged-hadron spectra in Au+Au at 200 GeV, INSPIRE 625472.

S. S. Adler et al. (PHENIX), "High pT charged hadron suppression in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRC 69, 034910 (2004), nucl-ex/0308006
(PPG023).

From the paper:
- charged hadrons in the west central arm, fiducial |eta| < 0.18;
- centrality from the BBC charge vs ZDC energy; minimum bias 92.2% of 6.9 b;
  two sets of classes (0-5, 5-10, 10-15, 15-20, 20-30, ... and 0-10,
  10-20, ..., 60-92%);
- (h+ + h-)/2 invariant yields per unit eta; errors: stat, sys and an
  occupancy uncertainty.

Records (public): invariant yields of (h+ + h-)/2 for 0-92% and the 13
classes in HEPData (Fig. 8; one record per class), |eta| < 0.18.  Staged (--staged OUT): the truncated mean
<pT> - pT,min above 0.5, 2 and 4.5 GeV/c vs centrality (Fig. 10: the
offset by pT,min is not computed).  Not curated: R_CP, R_AA, h/pi0 and the
integrated yield ratios (Figs. 11-14: nuclear modification), the p+p
reference (Table I).  The old record read the truncated mean as <pT> in
|eta| < 0.8.

Run from the repository root:
    python scripts/curate_phenix_charged_spectra_auau200_ppg023.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins625472/HEPData-ins625472-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. S. Adler et al. (PHENIX), PRC 69, 034910 (2004), nucl-ex/0308006'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge with the ZDC energy.'}
ETA = {'min': -0.18, 'max': 0.18}
OLD = 'data/mean_pT/phenix_mean_pT_high_p_t_charged_hadron_pp_200gev.yaml'
OUT = 'data/particle_spectra/phenix_particle_spectra_charged_AuAu_200gev_ins625472.yaml'
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for k in 'abcde':
        hf = f'ppg023fig8{k}.yaml'
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            lo, hi = re.findall(r'\d+', q['Centrality'])[:2]
            cls = f'{lo}-{hi}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '625472', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'(h+ + h-)/2 invariant yield, {cls} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': 'charged', 'charge_average': True},
                 'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}, Fig. 8. 1/(2 pi pT) d2N/(dpT deta) of (h+ + h-)/2, {cls}, |eta| < 0.18. '
                                    'Errors: stat, sys and occupancy.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    t = load('ppg023fig10.yaml')
    iv = t['independent_variables'][0]
    classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        ptmin = float(re.search(r'[\d.]+', q['$p_T^{min}$']).group())
        staged.append({'observable_type': 'mean_pT', 'inspire_id': '625472', 'version': VERSION,
                       'hepdata_file': 'ppg023fig10.yaml', 'dv_index': i,
                       'hepdata_table_label': f'truncated <pT> above {ptmin:g} GeV/c vs centrality (ppg023fig10.yaml)',
                       'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
                       'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': 'charged'},
                       'kinematic_range': {'eta': dict(ETA), 'pT': {'min': ptmin, 'max': 8.0}},
                       'observable': 'mean_pT', 'value_type': 'mean_pT',
                       'notes': {'value': f'{CITE}, Fig. 10. Truncated mean <pT> - pT,min of charged hadrons with '
                                          f'{ptmin:g} < pT < 8 GeV/c (the data stop near 6.5 GeV/c in peripheral '
                                          'classes). Errors: stat.'},
                       'staging_reason': 'truncated mean <pT> - pT,min: the offset is not computed'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '625472' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records;', [d['centrality']['classes'][0] for d in public])
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
