#!/usr/bin/env python3
"""Curate PHENIX v2 and v4 of charged hadrons in Au+Au at 200 GeV, INSPIRE 850211.

A. Adare et al. (PHENIX), "Elliptic and hexadecapole flow of charged hadrons
in Au+Au collisions at sqrt(sNN) = 200 GeV", PRL 105, 062301 (2010),
arXiv:1003.5586 (PPG098).

From the paper:
- charged hadrons in the central arms, |eta| <= 0.35; centrality from the
  BBC (3.1 < |eta| < 3.9);
- v_2k = <cos(2k (phi - Phi_2))> / Res(Psi_2k), k = 1, 2: v2 and v4 both
  relative to the SECOND-order event plane Phi_2;
- event planes from the BBC (3.1 < |eta| < 3.9), MPC (3.1 < |eta| < 3.7)
  and RXN inner (1.5 < |eta| < 2.8), outer (1.0 < |eta| < 1.5) and combined
  rings; the results use the combined RXN (1.0 < |eta| < 2.8);
- errors: statistical (Fig. 3; systematic in the paper: about 5-10%),
  statistical and systematic (Fig. 4).

Records (public): v2(pT) in 11 classes 0-5 ... 50-60% (Fig. 3a), v2 vs
centrality for five pT windows (Fig. 4a), both with the RXN event plane,
computed with the scalar-product estimator (pT bins from HEPData).  Staged
(--staged OUT): v4 (Figs. 3b, 4a) and v4/v2^2 (Fig. 4b): v4{Psi_2}, a
mixed-harmonic correlation with no computation.  Not curated: the
resolutions (Fig. 1), the detector comparison (Fig. 2: repeats Fig. 4 for
the RXN; in HEPData the BBC 0.8-1.2 GeV/c v2 and v4 columns are swapped
and the MPC 0.5-0.8 GeV/c v4 column repeats v2), the Glauber table.  The 5
old records all read the N_part column.

Run from the repository root:
    python scripts/curate_phenix_v2_v4_charged_ppg098.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins850211/HEPData-ins850211-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRL 105, 062301 (2010), arXiv:1003.5586'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}]}
RXN = {'detector': 'RXN', 'eta': [{'min': -2.8, 'max': -1.0}, {'min': 1.0, 'max': 2.8}], 'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'Second-order event plane of the RXN inner and outer rings combined, 1.0 < |eta| < 2.8 (scintillator '
                'weighting is not modelled).'}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_elliptic_and_hexadecapole_flow_of_AuAu_200gev.yaml'
STAGING = 'v4{Psi_2} (v4 relative to the second-order event plane): mixed-harmonic, no computation'


def base(hf, i, h, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '850211', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': f'{label} ({hf})', 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
            'particle_id': {'species': 'charged'}, 'value_type': h, 'method': 'event_plane',
            'flow_reference': dict(RXN)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    # Fig. 3: v2(pT), v4(pT) per class
    for hf, n in (('ppg098fig3a.yaml', 2), ('ppg098fig3b.yaml', 4)):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if not h.startswith('$v_'):
                continue                                         # the <pT> column
            cls = str(dv['qualifiers'][0]['value']) + '%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = base(hf, i, h, f'v{n}(pT), {cls}')
            d.update({'centrality': {'classes': [cls], **BBC}, 'harmonic': n, 'dimensionality': 'differential_pT',
                      'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1],
                                                                   'bin_ranges': pairs}},
                      'notes': {'value': f'{CITE}, Fig. 3. v{n}(pT) of charged hadrons, {cls}, |eta| < 0.35, '
                                         'relative to the RXN second-order event plane (1.0 < |eta| < 2.8)'
                                         + ('; computed with the scalar-product estimator' if n == 2 else '')
                                         + '. Errors: statistical (systematic in the paper).'}})
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if n == 4:
                d['staging_reason'] = STAGING
                staged.append(d)
            else:
                public.append(d)
    # Fig. 4a: v2, v4 vs centrality for pT windows; Fig. 4b: v4/v2^2
    for hf in ('ppg098fig4a.yaml', 'ppg098fig4b.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            if 'N_{part}' in h:
                continue
            m = re.search(r'([\d.]+) < \$p_T\$ < ([\d.]+)', str(dv['qualifiers'][0]['value']))
            pt = {'min': float(m.group(1)), 'max': float(m.group(2))}
            ratio = '/' in h
            n = 4 if h.startswith('$v_4') else 2
            what = 'v4/v2^2' if ratio else f'v{n}'
            d = base(hf, i, h, f"{what} vs centrality, {pt['min']:g} < pT < {pt['max']:g} GeV/c")
            d.update({'centrality': {'classes': classes, **BBC}, 'harmonic': n, 'dimensionality': 'integrated',
                      'kinematic_range': {'eta': dict(ETA), 'pT': pt},
                      'notes': {'value': f"{CITE}, Fig. 4. {what} of charged hadrons vs centrality, {pt['min']:g} < pT < "
                                         f"{pt['max']:g} GeV/c, |eta| < 0.35, RXN second-order event plane"
                                         + ('; computed with the scalar-product estimator' if n == 2 and not ratio else '')
                                         + '. Errors: statistical and systematic.'}})
            if ratio or n == 4:
                d['staging_reason'] = STAGING + (' (and a ratio)' if ratio else '')
                staged.append(d)
            else:
                public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '850211' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_v2_charged_AuAu_200gev_ins850211.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
