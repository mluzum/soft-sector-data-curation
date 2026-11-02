#!/usr/bin/env python3
"""Curate PHENIX pair anisotropies of the central-collision excess in d+Au at 200 GeV, INSPIRE 1222874 (staged only).

A. Adare et al. (PHENIX), "Quadrupole anisotropy in dihadron azimuthal
correlations in central d+Au collisions at sqrt(sNN) = 200 GeV", PRL 111,
212301 (2013), arXiv:1303.1794 (PPG152).

From the paper:
- charged-hadron pairs in the central arms (|eta| < 0.35), 0.48 <
  |Delta eta| < 0.7; centrality from the BBC on the Au-going side: central
  0-5% and peripheral 50-88%;
- the central-minus-peripheral excess Y_c - Y_p, its Fourier coefficients
  c2, c3 (trigger 0.5 < pT < 0.75 GeV/c); v2(pT) from c2 by factorization,
  "the excess v2 beyond any which is present in peripheral d+Au";
- v2/epsilon2 with three Glauber variants;
- errors: stat and sys.

Every record is staged (--staged OUT): c2 and c3 vs associated pT and the
excess v2(pT), 0-5% (Tables 1-3; the peripheral subtraction is not
computed; pT central values, edges at the midpoints).  Not curated:
v2/epsilon2 (Tables 4, 5: Glauber normalized).  The old public record read
the excess v2 as v2{2}.

Run from the repository root:  python scripts/curate_phenix_dau_excess_v2_200.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1222874/HEPData-ins1222874-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRL 111, 212301 (2013), arXiv:1303.1794'
BBCS = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'}, 'eta': [{'min': -3.9, 'max': -3.0}],
        'notes': 'BBC charge on the Au-going side.'}
OLD = 'data/flow_vn/phenix_flow_vn_quadrupole_anisotropy_AuAu_200gev.yaml'
REASON = ('the central-minus-peripheral (50-88%) excess: the peripheral subtraction is not computed; pT points without '
          'bin edges')


def edges(x):
    mids = [round((a + b) / 2, 4) for a, b in zip(x, x[1:])]
    e = [round(x[0] - (mids[0] - x[0]), 4)] + mids + [round(x[-1] + (x[-1] - mids[-1]), 4)]
    return [[a, b] for a, b in zip(e, e[1:])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    staged = []
    for hf, n, what in (('Table1.yaml', 2, 'c2 of the excess vs associated pT'),
                        ('Table2.yaml', 3, 'c3 of the excess vs associated pT'),
                        ('Table3.yaml', 2, 'excess v2(pT)')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        x = [float(v['value']) for v in t['independent_variables'][0]['values']]
        bins = edges(x)
        dv = t['dependent_variables'][0]
        pair = hf != 'Table3.yaml'
        d = {'observable_type': 'flow_vn', 'inspire_id': '1222874', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'{what}, 0-5% d+Au ({hf})',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'D AU --> CHARGED X', 'frame': 'lab',
                                  'rapidity_shift': 0.0, 'forward_beam': 'd'},
             'centrality': {'classes': ['0-5%'], **BBCS}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
             'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 0.48,
             'value_type': dv['header']['name'], 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}. {what}, 0-5% d+Au minus 50-88% (scaled), pairs at |eta| < 0.35 with 0.48 < '
                                '|Delta eta| < 0.7' + ('; trigger 0.5 < pT < 0.75 GeV/c' if pair else
                                                       '; v2 from c2 by factorization')
                                + '; pT central values, edges at the midpoints. Errors: stat and sys.'},
             'staging_reason': REASON + ('; c_n is a pair coefficient' if pair else '')}
        if pair:
            d['flow_reference'] = {'pT': {'min': 0.5, 'max': 0.75}, 'notes': 'The trigger-particle pT window.'}
        staged.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '1222874' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    print(f'{len(staged)} staged records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
