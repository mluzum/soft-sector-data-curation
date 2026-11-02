#!/usr/bin/env python3
"""Curate PHENIX pi0 v2 in Au+Au at 200 GeV, INSPIRE 858845.

A. Adare et al. (PHENIX), "Azimuthal anisotropy of pi0 production in
Au+Au collisions at sqrt(sNN) = 200 GeV: path-length dependence of jet
quenching and the role of initial geometry", PRL 105, 142301 (2010),
arXiv:1006.3740 (PPG110).

From the paper and tables:
- pi0 -> gamma gamma in the EMCal (|eta| < 0.35); centrality from the BBC;
- v2 relative to the event plane of the MPC (3.1 < |eta| < 3.7) and the
  inner RXN ring (1.5 < |eta| < 2.8) combined;
- errors: stat (Fig. 1a), stat and sys (Fig. 2a).

Records (public): pi0 v2(pT), 1 < pT < 18 GeV/c, in 0-10 ... 50-60% (Fig.
1a) and v2 vs centrality for 6 < pT < 9 and pT > 9 GeV/c (Fig. 2a),
MPC+RXN_in event plane computed with the scalar-product estimator; pi0 by
isospin, (pi+ + pi-)/2, unless the container has a pi0 axis.  Not
curated: R_AA (nuclear modification: out of scope), the MPC/RXN ratios
(Fig. 1b) and the model curves (Fig. 2b).  The 3 old records read the
N_part column as v2.

Run from the repository root:  python scripts/curate_phenix_pi0_v2_ppg110.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins858845/HEPData-ins858845-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRL 105, 142301 (2010), arXiv:1006.3740'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
REF = {'detector': 'MPC + RXN_in', 'eta': [{'min': -3.7, 'max': -3.1}, {'min': -2.8, 'max': -1.5},
                                           {'min': 1.5, 'max': 2.8}, {'min': 3.1, 'max': 3.7}],
       'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'Event plane of the MPC (3.1 < |eta| < 3.7) and the inner RXN ring (1.5 < |eta| < 2.8) combined.'}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_azimuthal_anisotropy_of_neutral_pion_AuAu_200gev.yaml'


def base(hf, i, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '858845', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf})', 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> PI0 X'},
            'particle_id': {'species': 'pi0'}, 'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(REF),
            'value_type': 'v_2'}


def main():
    public = []
    t = yaml.safe_load(open(os.path.join(RAW, 'ppg110fig1a.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        cls = str(dv['qualifiers'][0]['value'])
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = base('ppg110fig1a.yaml', i, f'pi0 v2(pT), {cls}')
        d.update({'centrality': {'classes': [cls], **BBC}, 'dimensionality': 'differential_pT',
                  'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                  'notes': {'value': f'{CITE}, Fig. 1a. v2(pT) of pi0, {cls}, |eta| < 0.35, MPC+RXN_in event plane, '
                                     'computed with the scalar-product estimator. Errors: stat.'}})
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        public.append(d)
    for hf in ('ppg110fig2a1.yaml', 'ppg110fig2a2.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            if 'N_{part}' in dv['header']['name']:
                continue
            q = str(dv['qualifiers'][0]['value'])
            nums = [float(x) for x in re.findall(r'[\d.]+', q)]
            pt = {'min': nums[0], 'max': nums[1]} if len(nums) == 2 else {'min': nums[0], 'max': 100.0}
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
            what = f"{pt['min']:g} < pT < {pt['max']:g} GeV/c" if len(nums) == 2 else f"pT > {pt['min']:g} GeV/c"
            d = base(hf, i, f'pi0 v2 vs centrality, {what}')
            d.update({'centrality': {'classes': classes, **BBC}, 'dimensionality': 'integrated',
                      'kinematic_range': {'eta': dict(ETA), 'pT': pt},
                      'notes': {'value': f'{CITE}, Fig. 2a. v2 of pi0 vs centrality, {what}, |eta| < 0.35, MPC+RXN_in '
                                         'event plane, computed with the scalar-product estimator. Errors: stat and sys.'}})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '858845' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_pi0_v2_AuAu_200gev_ins858845.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')


if __name__ == '__main__':
    main()
