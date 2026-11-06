#!/usr/bin/env python3
"""Curate STAR strange hadron spectra in Cu+Cu and Au+Au at 200 GeV, INSPIRE 918779.

G. Agakishiev et al. (STAR), "Strangeness enhancement in Cu+Cu and Au+Au
collisions at sqrt(sNN) = 200 GeV", PRL 108 (2012) 072301,
arXiv:1107.2955.

From the paper:
- K0_S, Lambda, Lambdabar, Xi-, Xibar+, Omega + Omegabar at |y| < 0.5
  from their weak decays; Fig. 1: invariant yields 1/(2 pi pT) d2N/(dpT
  dy) per class, Lambda and Lambdabar NOT corrected for the weak-decay
  feed-down (~15% from Xi, independent of pT), errors stat and sys
  combined;
- centrality from the charged multiplicity in |eta| < 0.5; Cu+Cu 0-10,
  10-20, 20-30, 30-40, 40-60%; Au+Au 0-5, 10-20, 20-40, 40-60, 60-80%;
- Fig. 2: spectra over <N_part> (Cu+Cu 0-10%, Au+Au 20-40%, Lambda
  feed-down subtracted; the Au+Au multi-strange data from PRL 98 062301);
  Fig. 3: yields per <N_part> relative to pp (enhancement factors);
  Fig. 4: Cu+Cu/Au+Au yields at <N_part> = 99.

Records (public): the Fig. 1 spectra, 45 (Cu+Cu K0_S, Lambda, Lambdabar,
Xi-, Xibar+, Omega + Omegabar in 5 classes; Au+Au K0_S, Lambda,
Lambdabar in 5 classes).  Staged (--staged OUT): Figs. 2-4 (spectra over
an approximate <N_part>, ratios to pp, the cross-system ratio).

Run from the repository root:  python scripts/curate_star_strange_cucu_auau200.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins918779/HEPData-ins918779-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Agakishiev et al. (STAR), PRL 108 (2012) 072301, arXiv:1107.2955'
OUT = 'data/particle_spectra/star_particle_spectra_strange_CuCu_AuAu_200gev_ins918779.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'charged multiplicity in |eta| < 0.5.'}
SPECIES = {'$K^0_S$': ('K0_S', 'K0S'), '$\\Lambda$': ('Lambda', 'LAMBDA'), '$\\bar{\\Lambda}$': ('Lambdabar', 'LAMBDABAR'),
           '$\\Xi$': ('Xi-', 'XI-'), '$\\bar{\\Xi}$': ('Xibar+', 'XIBAR+'),
           '$\\Omega + \\bar{\\Omega}$': ('Omega- + Omegabar+', 'OMEGA- OMEGABAR+')}
SYSTEM = {'Cu+Cu': 'CU CU', 'Au+Au': 'AU AU'}
FEED = {'Lambda': ' Not corrected for the Xi feed-down (~15%, independent of pT).',
        'Lambdabar': ' Not corrected for the Xibar feed-down (~15%, independent of pT).'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if not hf.startswith('fig1.'):
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'particle_yields', 'inspire_id': '918779', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{desc[:100]} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc}'},
                               'staging_reason': ('spectra over an approximate <N_part> (Fig. 1 holds the spectra)'
                                                  if hf.startswith('fig2') else
                                                  'yields per <N_part> relative to pp (enhancement factor)'
                                                  if hf.startswith('fig3') else
                                                  'Cu+Cu/Au+Au yield ratio at equal <N_part> (cross-system)')})
            continue
        assert 'invariant mass spectra' in desc and '|y| < 0.5' in desc, hf
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            q = list({k['name']: k['value'] for k in dv['qualifiers']}.values())[0]
            m = re.fullmatch(r'(Cu\+Cu|Au\+Au) (.+?)\s+\$?(\d+)-(\d+)\\?%\$?', q.strip())
            assert m, q
            system, sp, lo, hi = m.groups()
            species, prod = SPECIES[sp.strip()]
            cls = f'{lo}-{hi}%'
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            assert rows == list(range(len(bins))), (hf, i)
            assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('Stat+Sys error',)}, hf
            d = {'observable_type': 'particle_spectra', 'inspire_id': '918779', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'Fig. 1: {species} invariant yield, {system} 200 GeV, {cls} '
                                                      f'({hf} column {i})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': f'{SYSTEM[system]} --> {prod} X'},
                 'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': species},
                 'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                 'hepdata_corrections': {'pT': "HEPData's qualifier named p_T holds the system, species and class; "
                                               "the pT bins are the table's"},
                 'notes': {'value': f'{CITE}, Fig. 1. 1/(2 pi pT) d2N/(dpT dy) of {species}, |y| < 0.5, {system} 200 '
                                    f'GeV {cls}.{FEED.get(species, "")} Errors: stat and sys combined.'}}
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
