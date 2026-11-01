#!/usr/bin/env python3
"""Curate ATLAS v_n (n = 2-6) in Pb+Pb at 2.76 TeV, INSPIRE 1093733.

G. Aad et al. (ATLAS), "Measurement of the azimuthal anisotropy for charged
particle production in sqrt(sNN) = 2.76 TeV lead-lead collisions with the
ATLAS detector", PRC 86, 014907 (2012), arXiv:1203.3087.

From the paper:
- tracks |eta| < 2.5; centrality from the FCal Sigma E_T, 3.2 < |eta| < 4.9;
- event plane (EP), "full FCal" method: tracks correlated with the event
  plane of the whole FCal (3.2 < |eta| < 4.9), resolution from the two
  halves -- v_n(pT) (Tables 46-53) and v_n vs centrality;
- "FCal_P(N)" method: tracks at eta >= 0 (< 0) with the event plane of the
  opposite FCal half -- v_n(|eta|) (Tables 6-45);
- two-particle correlations (2PC) with 2 < |Delta eta| < 5: v_n(pT_b) =
  v_nn(pT_a, pT_b) / v_n(pT_a) for reference ranges pT_a (Tables 91-114);
- errors: statistical and systematic.

Records: one per table column.  EP records: event_plane with the FCal
reference (both halves for the full-FCal v_n(pT); the opposite half,
mirror, for v_n(|eta|), computed at eta > 0), computed with the scalar-
product estimator.  2PC records: cumulant order 2, POI pT_b points
(0.1 GeV windows), reference pT_a (flow_reference.pT), pair gap recorded
as eta_gap.  Not curated: v_n vs centrality (classes 0-1% and 0-5% in one
table overlap), the resolution and chi tables, correlation functions,
v_nn(pT_a, pT_b), v_n and v_nn vs n, v_1,1 and the fitted v_1.

Run from the repository root:  python scripts/curate_atlas_vn_pbpb_2760.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1093733/HEPData-ins1093733-v1-yaml'
CITE = 'G. Aad et al. (ATLAS), PRC 86, 014907 (2012), arXiv:1203.3087'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
WINDOW = 0.1


def rng(text):
    lo, hi = (float(x) for x in str(text).split('-'))
    return lo, hi


def bins(iv):
    pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
    ax = {'min': pairs[0][0], 'max': pairs[-1][1]}
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])):
        ax['bins'] = [pairs[0][0]] + [p[1] for p in pairs]
    else:
        ax['bin_ranges'] = pairs
    return ax


def rows(dv):
    return [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]


def base(hf, i, n, cls, label):
    lo, hi = rng(cls)
    return {'observable_type': 'flow_vn', 'inspire_id': '1093733', 'version': 1, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': label,
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [f'{lo:g}-{hi:g}%'], **FCAL},
            'particle_id': {'species': 'charged'}, 'harmonic': n, 'value_type': f'v_{n}'}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        desc = ' '.join(x['description'].split())
        hf = x['data_file']
        if desc.startswith(('V_n vs PT for centrality', 'The Fourier coefiiciant V_n vs eta', 'v_n extracted from 2PC')):
            t = yaml.safe_load(open(os.path.join(RAW, hf)))
        else:
            continue
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: qq.get('value') for qq in dv.get('qualifiers', [])}
            r = rows(dv)
            if not r:
                continue
            assert r == list(range(r[0], r[-1] + 1)), (hf, i)
            if desc.startswith('V_n vs PT'):
                n = int(float(q['N']))
                d = base(hf, i, n, q['CENTRALITY'], f'{desc} (n = {n})')
                d.update({'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5}, 'pT': bins({'values': [iv['values'][j] for j in r]})},
                          'method': 'event_plane', 'dimensionality': 'differential_pT',
                          'flow_reference': {'detector': 'FCal', 'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}],
                                             'notes': 'Event plane of the full FCal, 3.2 < |eta| < 4.9.'},
                          'notes': {'value': f'{CITE}. v_{n}(pT), |eta| < 2.5, full-FCal event plane; computed with the '
                                             'scalar-product estimator. Errors: statistical and systematic.'}})
            elif desc.startswith('The Fourier'):
                n = int(float(q['N']))
                plo, phi_ = rng(q['PT'])
                d = base(hf, i, n, q['CENTRALITY'], f'{desc} (n = {n})')
                d.update({'kinematic_range': {'eta': bins({'values': [iv['values'][j] for j in r]}), 'pT': {'min': plo, 'max': phi_}},
                          'method': 'event_plane', 'dimensionality': 'differential_eta',
                          'flow_reference': {'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.9}],
                                             'notes': 'FCal_P(N): event plane of the FCal half opposite the track.'},
                          'notes': {'value': f'{CITE}. v_{n}(|eta|) for {plo:g} < pT < {phi_:g} GeV, FCal_P(N) event '
                                             'plane (opposite half); computed at eta > 0 with the scalar-product '
                                             'estimator. Errors as tabulated.'}})
            else:
                n = int(dv['header']['name'].split('_')[1])
                alo, ahi = rng(q['PT(C=A)'])
                pts = [float(iv['values'][j]['value']) for j in r]
                d = base(hf, i, n, q['CENTRALITY'], f'{desc} (v_{n}, reference {alo:g} < pT_a < {ahi:g} GeV)')
                d.update({'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5},
                                              'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                                     'points': pts, 'point_window': WINDOW}},
                          'method': 'cumulant', 'cumulant_order': 2, 'dimensionality': 'differential_pT', 'eta_gap': 2.0,
                          'flow_reference': {'pT': {'min': alo, 'max': ahi},
                                             'notes': 'Two-particle correlations, 2 < |Delta eta| < 5 (no pair gap in '
                                                      'the computation).'},
                          'notes': {'value': f'{CITE}. v_{n}(pT_b) = v_nn(pT_a, pT_b)/v_n(pT_a), reference {alo:g} < '
                                             f'pT_a < {ahi:g} GeV; pT_b points ({WINDOW} GeV windows). Errors as '
                                             'tabulated.'}})
            docs.append(d)
    out = 'data/flow_vn/atlas_flow_vn_azimuthal_anisotropy_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;', dict(collections.Counter((d['method'], d['dimensionality']) for d in docs)))


if __name__ == '__main__':
    main()
