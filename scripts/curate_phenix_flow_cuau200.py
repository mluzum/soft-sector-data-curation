#!/usr/bin/env python3
"""Curate PHENIX v2, v3 in Cu+Au at 200 GeV, INSPIRE 1394897.

A. Adare et al. (PHENIX), "Measurements of directed, elliptic, and
triangular flow in Cu+Au collisions at sqrt(sNN) = 200 GeV", PRC 94,
054910 (2016), arXiv:1509.07784.

From the paper (Sec. II):
- charged hadrons and pi, K, p (charges combined) in the central arms,
  |eta| < 0.35; centrality from the total BBC charge (3.0 < |eta| < 3.9,
  both sides);
- Cu (projectile) flies toward +eta (North), Au toward -eta (South);
  equal beam energies per nucleon: lab = nucleon-nucleon frame;
- v2, v3: event plane of the combined South and North BBCs (3.0 < |eta| <
  3.9; every particle); v1: relative to the spectator plane of the South
  SMD (Au spectator neutrons);
- errors: statistical and systematic.

Records (public): charged v2(pT) (Fig. 5, six classes; Fig. 16, 0-5%), v3(pT)
(Fig. 6; Fig. 17), identified v2(pT) (Fig. 7) and v3(pT) (Fig. 8), with
collision_system frame nn_cms and forward_beam Cu.  Staged (--staged OUT):
v1 (Figs. 4, 9: spectator plane).  Not curated: the eccentricity- and
N_part-scaled v2, v3 (Figs. 11-15: Glauber; mostly Au+Au, Cu+Cu data of
other papers).  The 17 'mean_pT' records were the <pT> bin-centre columns
of those figures, not measurements: removed.

Run from the repository root:
    python scripts/curate_phenix_flow_cuau200.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1394897/HEPData-ins1394897-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 94, 054910 (2016), arXiv:1509.07784'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}]}
BBC_REF = {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
           'pT': {'min': 0.0, 'max': 100.0},
           'notes': 'Event plane of the combined South and North BBCs (Reference pT: all particles; the '
                    'charge weighting is not modelled).'}
SMD_REF = {'detector': 'SMD', 'plane': 'spectator',
           'notes': 'First-order spectator plane of the South SMD (Au spectator neutrons).'}
SYSTEM = {'energy_gev': 200.0, 'frame': 'nn_cms', 'rapidity_shift': 0.0, 'forward_beam': 'Cu'}
SPECIES = {'pion': 'pi+ + pi-', 'kaon': 'K+ + K-', 'proton': 'p + pbar'}
FLOW = ['Fig5', 'Fig6', 'Fig7a', 'Fig7b', 'Fig7c', 'Fig7d', 'Fig7e', 'Fig8a', 'Fig8b', 'Fig8c', 'Fig16', 'Fig17']
V1 = ['Fig4', 'Fig9a', 'Fig9b']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the v1 records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    public, staged = [], []
    for stem in FLOW + V1:
        hf = f'{stem}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name'].lower(): str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            cls = q['centrality']
            sp = SPECIES[q['particle']] if 'particle' in q else 'charged'
            n = int(re.search(r'v_?\{?(\d)', dv['header']['name']).group(1))
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            vals = [iv['values'][j] for j in rows]
            pairs = [[float(v['low']), float(v['high'])] for v in vals]
            if rows == list(range(rows[0], rows[-1] + 1)):
                pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [pairs[0][0]] + [p[1] for p in pairs]}
            else:
                pt = {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}
            d = {'observable_type': 'flow_vn', 'inspire_id': '1394897', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{' '.join(sub[hf]['description'].split())[:150]} | {cls}",
                 'collision_system': dict(SYSTEM, reaction='CU AU --> CHARGED X' if sp == 'charged'
                                          else 'CU AU --> X'),
                 'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': pt},
                 'harmonic': n, 'method': 'event_plane', 'value_type': dv['header']['name'],
                 'dimensionality': 'differential_pT'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if stem in V1:
                d['flow_reference'] = dict(SMD_REF)
                d['notes'] = {'value': f'{CITE}. v1(pT) of {sp}, {cls}, |eta| < 0.35, relative to the South SMD '
                                       'spectator plane. Errors: statistical and systematic.'}
                d['staging_reason'] = 'v1 relative to the spectator plane (South SMD): models have no spectators.'
                staged.append(d)
                continue
            d['flow_reference'] = dict(BBC_REF)
            d['notes'] = {'value': f'{CITE}. v{n}(pT) of {sp}, {cls}, |eta| < 0.35, combined BBC event plane '
                                   '(3.0 < |eta| < 3.9), computed with the scalar-product estimator. Errors: '
                                   'statistical and systematic.'}
            public.append(d)
    for old in ('data/flow_vn/phenix_flow_vn_measurements_of_directed_elliptic_and_CuCu_200gev.yaml',
                'data/mean_pT/phenix_mean_pT_measurements_of_directed_elliptic_and_CuCu_200gev.yaml'):
        if os.path.exists(old):
            os.remove(old)
    out = 'data/flow_vn/phenix_flow_vn_v2_v3_CuAu_200gev_ins1394897.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(
        (d['harmonic'], d['particle_id']['species']) for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
