#!/usr/bin/env python3
"""Curate ATLAS pT-integrated v2 vs eta and vs pT threshold in Pb+Pb at 2.76 TeV, INSPIRE 1296260.

G. Aad et al. (ATLAS), "Measurement of the centrality and pseudorapidity
dependence of the integrated elliptic flow in lead-lead collisions at
sqrt(sNN) = 2.76 TeV with the ATLAS detector", EPJC 74 (2014) 2982,
arXiv:1405.3936.

From the paper (Secs. 3-5):
- centrality from the FCal Sum E_T (3.2 < |eta| < 4.9), 10% classes;
- event plane from the FCal half opposite the tracks (3.2 < |eta| < 4.8);
- three track methods: tracklets (TKT, field-off, pT > 0.07 GeV), pixel
  tracks (PXT, pT > 0.1 GeV) and inner-detector tracks (IDT, pT > 0.5 GeV);
- errors as tabulated.

Records (public): v2(eta) integrated over pT > 0.07 (TKT), > 0.1 (PXT) and
> 0.5 GeV (PXT) for 0-10 ... 70-80% (Tables 24-31), and v2 at |eta| < 1
integrated over pT > pT,0 (Tables 16-23: one record per pT,0, TKT at
0.07, PXT at 0.1-0.5 GeV); method event_plane with the mirrored FCal
reference (for |eta| < 1 both halves, as for a particle spanning eta = 0).
Not curated: the IDT pT > 0.5 series (repeats PXT pT > 0.5 with another
tracking method), v2 vs |eta| - y_beam (Table 32), and the efficiency and
fake-rate tables (Tables 1-15, 33-38: three of them were public as v2).

Run from the repository root:  python scripts/curate_atlas_integrated_v2_pbpb_2760.py
"""
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1296260/HEPData-ins1296260-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), EPJC 74 (2014) 2982, arXiv:1405.3936'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
FCAL_REF = {'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.8}], 'pT': {'min': 0.0, 'max': 100.0},
            'notes': 'FCal half opposite the tracks (E_T-weighted in the paper; unit weights here). Reference pT: '
                     'all particles.'}
OLD = 'data/flow_vn/atlas_flow_vn_centrality_and_pseudorapidity_dependence_of_PbPb_2760gev.yaml'


def base(hf, i, cls, method_label):
    return {'observable_type': 'flow_vn', 'inspire_id': '1296260', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [cls], **FCAL}, 'particle_id': {'species': 'charged'},
            'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(FCAL_REF)}


def main():
    docs = []
    for n in range(24, 32):                                   # v2(eta), pT-integrated
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if h.startswith('Elliptic Flow (IDT'):
                continue
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            ptmin = float(re.search(r'p_\{T\}>([\d.]+)', h).group(1))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = base(hf, i, cls, h)
            d.update({'hepdata_table_label': f'{h}, {cls} ({hf})',
                      'kinematic_range': {'eta': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                          'pT': {'min': ptmin, 'max': 100.0}},
                      'value_type': h, 'dimensionality': 'differential_eta',
                      'notes': {'value': f'{CITE}. v2(eta) of charged particles with pT > {ptmin:g} GeV ({h.split(",")[0][15:]}), '
                                         f'{cls}, FCal event plane from the opposite side; computed with the '
                                         'scalar-product estimator. Errors as tabulated.'}})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            docs.append(d)
    for n in range(16, 24):                                   # v2(pT > pT0), |eta| < 1
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if h.startswith('Elliptic Flow (IDT'):
                continue
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            assert q['ETARAP'] == '-1.0-1.0', (hf, q)
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            for j, v in enumerate(dv['values']):
                if str(v['value']).strip() in ('', '-'):
                    continue
                pt0 = float(iv['values'][j]['value'])
                d = base(hf, i, cls, h)
                d.update({'hepdata_table_label': f'{h}, pT > {pt0:g} GeV, {cls} ({hf}, row {j})',
                          'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': pt0, 'max': 100.0}},
                          'value_type': h, 'dimensionality': 'integrated', 'table_rows': [j],
                          'notes': {'value': f'{CITE}. v2 of charged particles with pT > {pt0:g} GeV, |eta| < 1, {cls} '
                                             f'({h[15:-1]}), FCal event plane from the opposite side (both halves '
                                             'here); computed with the scalar-product estimator.'}})
                docs.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '1296260' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/atlas_flow_vn_integrated_v2_PbPb_2760gev_ins1296260.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['dimensionality'], d['value_type'][:25]) for d in docs)))


if __name__ == '__main__':
    main()
