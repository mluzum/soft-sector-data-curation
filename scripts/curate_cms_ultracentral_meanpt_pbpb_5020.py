#!/usr/bin/env python3
"""Curate CMS <pT> vs multiplicity in ultra-central Pb-Pb at 5.02 TeV, INSPIRE 2747107 (staged only).

A. Hayrapetyan et al. (CMS), "Extracting the speed of sound in quark-gluon
plasma with ultrarelativistic lead-lead collisions at the LHC", Rept. Prog.
Phys. 87 (2024) 077801, arXiv:2401.06896.

From the tables: charged particles at |eta| < 0.5, extrapolated to all
pT; ultra-central events in bins of N_ch / N_ch^0 (N_ch^0: the 0-5% value);
<pT> / <pT>^0 vs N_ch / N_ch^0 (errors stat and sys) and the event
fraction vs N_ch / N_ch^0; c_s^2 at T_eff from the slope.

Every record is staged (--staged OUT): <pT>/<pT>^0 vs N_ch/N_ch^0 (figure
3: event classes in normalized multiplicity and a ratio to the 0-5% value,
not computed) and the event-fraction distribution (figure 2: a multiplicity
distribution, no uncertainties).  Not curated: c_s^2 (figure 4, derived).
The old public record read figure 3 as <pT> in 0-5%, |eta| < 2.4.

Run from the repository root:  python scripts/curate_cms_ultracentral_meanpt_pbpb_5020.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2747107/HEPData-ins2747107-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Hayrapetyan et al. (CMS), Rept. Prog. Phys. 87 (2024) 077801, arXiv:2401.06896'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
OLD = 'data/mean_pT/cms_mean_pT_extracting_the_speed_of_sound_PbPb_5.02gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    for hf, otype, what, reason in (
            ('figure_3.yaml', 'mean_pT', '<pT>/<pT>^0 vs N_ch/N_ch^0',
             'event classes in normalized multiplicity N_ch/N_ch^0 and a ratio to the 0-5% value: not computed'),
            ('figure_2.yaml', 'multiplicity_distribution', 'event fraction vs N_ch/N_ch^0',
             'multiplicity distribution in normalized N_ch (no uncertainties): not computed')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        x = [[min(float(v['low']), float(v['high'])), max(float(v['low']), float(v['high']))] for v in iv['values']]
        staged.append({'observable_type': otype, 'inspire_id': '2747107', 'version': VERSION, 'hepdata_file': hf,
                       'dv_index': 0, 'hepdata_table_label': f'{what} ({hf})',
                       'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                       'centrality': {'classes': ['0-5%'], **HF}, 'particle_id': {'species': 'charged'},
                       'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
                       'observable': what, 'value_type': dv['header']['name'], 'nch_bins': x,
                       'notes': {'value': f'{CITE}. {what} of charged particles at |eta| < 0.5, all pT (extrapolated), '
                                          'in ultra-central events (bins of N_ch normalized to the 0-5% value).'},
                       'staging_reason': reason})
        if otype == 'multiplicity_distribution':
            staged[-1]['efficiency_corrected'] = True        # N_ch corrected and extrapolated to all pT
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '2747107' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    print(f'{len(staged)} staged records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
