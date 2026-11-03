#!/usr/bin/env python3
"""Curate CMS K0S, Lambda and h+- in OO and pp at 5.36 TeV, INSPIRE 3192577 (staged only).

CMS Collaboration, "Observation of enhanced baryon production using strange
hadrons in oxygen-oxygen collisions", arXiv:2608.18256.

From the paper: minimum-bias OO and pp collisions; K0S (pT > 0.5 GeV) and
Lambda (pT > 0.7 GeV, feed-down from Xi corrected) at |y| < 1, h+- at
|eta| < 1 (pT < 3 GeV here); invariant cross sections (1/(2 pi pT))
d2sigma/(dpT dy), the nuclear modification factor R_OO (with the pp
reference) and the Lambda/K0S cross-section ratio, in OO and pp; errors:
statistical and systematic (a normalization uncertainty correlated over pT).

Every record is staged (--staged OUT): minimum-bias cross sections and pp
references (the pipeline computes per-event yields in centrality classes),
R_OO (a ratio to pp) and Lambda/K0S of minimum-bias collisions; Lambda is
also absent from current JETSCAPE O+O output.

Run from the repository root:  python scripts/curate_cms_strange_oo_5360.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins3192577/HEPData-ins3192577-v*-yaml')[0]
VERSION = int(re.search(r'-v(\d+)-yaml', RAW).group(1))
CITE = 'CMS Collaboration, arXiv:2608.18256'
SPECIES = {'K_{0}^{S}': 'K0_S', 'K^{0}_{S}': 'K0_S', '\\Lambda': 'Lambda', 'h^\\pm': 'charged', 'h^{\\pm}': 'charged'}
REASON = {'spectrum': 'minimum-bias invariant cross section (no centrality classes; per-event yields only)',
          'R_OO': 'nuclear modification factor of minimum-bias OO over the pp reference',
          'ratio': 'Lambda/K0S cross-section ratio of minimum-bias collisions'}


def species_of(desc):
    for k, v in SPECIES.items():
        if k in desc:
            return v
    raise ValueError(desc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', required=True, help='write the staged records here (a staging collection)')
    out = ap.parse_args().staged
    staged = []
    for sub in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))):
        if not sub or not sub.get('data_file'):
            continue
        hf, desc = sub['data_file'], sub.get('description', '')
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]['values']
        bins = [[float(v['low']), float(v['high'])] for v in iv]
        dv = t['dependent_variables'][0]
        pp = 'pp collisions' in desc
        kind = 'R_OO' if 'R_{OO}' in desc else ('ratio' if 'Lambda/K' in desc else 'spectrum')
        sp = 'Lambda/K0_S' if kind == 'ratio' else species_of(desc)
        d = {'observable_type': {'spectrum': 'particle_spectra', 'R_OO': 'nuclear_modification',
                                 'ratio': 'particle_ratios'}[kind],
             'inspire_id': '3192577', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'{desc.strip()} ({hf})',
             'collision_system': {'energy_gev': 5360.0,
                                  'reaction': f"{'P P' if pp else 'O O'} --> {sp.upper()} X"},
             'particle_id': {'species': sp},
             'kinematic_range': {('eta' if sp == 'charged' else 'y'): {'min': -1.0, 'max': 1.0},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'observable': kind, 'value_type': dv['header']['name'],
             'notes': {'value': f'{CITE}. {desc.strip()} Errors: stat and sys.'},
             'staging_reason': REASON[kind] + ('; pp' if pp else '')}
        staged.append(d)
    with open(out, 'w') as fh:
        yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(staged)} staged records')


if __name__ == '__main__':
    main()
