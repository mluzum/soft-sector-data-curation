#!/usr/bin/env python3
"""Curate PHENIX v2 and v3 of charged hadrons in 0-5% 3He+Au at 200 GeV, INSPIRE 1384274.

A. Adare et al. (PHENIX), "Measurements of elliptic and triangular flow in
high-multiplicity 3He+Au collisions at sqrt(sNN) = 200 GeV", PRL 115,
142301 (2015), arXiv:1507.06273 (PPG181).

From the paper:
- inclusive charged hadrons in the central arms, |eta| < 0.35;
- centrality from the charge in the BBC on the Au-going (south) side,
  -3.9 < eta < -3.0; the 3He goes north (+eta);
- final v_n from the n-th order event plane of the FVTX-S (Au-going
  endcap, -3.0 < eta < -1.0), resolution from three event planes;
- errors: stat and sys.

Records (public): v2(pT) and v3(pT), 0-5% (Fig. 4), the FVTX-S event plane
computed with the scalar-product estimator; lab frame (equal beam energies
per nucleon), the 3He toward +eta.  The old record had v2 only, both BBCs
as the estimator, no reference and no frame.

Run from the repository root:  python scripts/curate_phenix_v2_v3_he3au_200.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1384274/HEPData-ins1384274-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRL 115, 142301 (2015), arXiv:1507.06273'
BBCS = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'}, 'eta': [{'min': -3.9, 'max': -3.0}],
        'notes': 'BBC charge on the Au-going side.'}
OLD = 'data/flow_vn/phenix_flow_vn_measurements_of_elliptic_and_triangular_AuAu_200gev.yaml'
OUT = 'data/flow_vn/phenix_flow_vn_v2_v3_charged_He3Au_200gev_ins1384274.yaml'
BLANK = ('', '-', '--', 'none')


def fvtxs(n):
    return {'detector': 'FVTX-S', 'eta': [{'min': -3.0, 'max': -1.0}], 'pT': {'min': 0.0, 'max': 100.0},
            'notes': f'Order-{n} event plane of the FVTX on the Au-going side (hit clusters: all pT).'}


def main():
    t = yaml.safe_load(open(os.path.join(RAW, 'ppg181fig4.yaml')))
    iv = t['independent_variables'][0]
    public = []
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        n = int(h.strip('$').split('_')[1])
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'flow_vn', 'inspire_id': '1384274', 'version': VERSION, 'hepdata_file': 'ppg181fig4.yaml',
             'dv_index': i, 'hepdata_table_label': f'v{n}(pT), 0-5% 3He+Au (ppg181fig4.yaml)',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'HE3 AU --> CHARGED X', 'frame': 'lab',
                                  'rapidity_shift': 0.0, 'forward_beam': '3He'},
             'centrality': {'classes': ['0-5%'], **BBCS}, 'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'harmonic': n, 'method': 'event_plane', 'flow_reference': fvtxs(n), 'value_type': h,
             'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}, Fig. 4. v{n}(pT) of charged hadrons, 0-5% 3He+Au (BBC-S), |eta| < 0.35, '
                                f'relative to the order-{n} FVTX-S event plane (-3 < eta < -1), computed with the '
                                'scalar-product estimator; lab frame, 3He toward +eta. Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        public.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '1384274' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records')


if __name__ == '__main__':
    main()
