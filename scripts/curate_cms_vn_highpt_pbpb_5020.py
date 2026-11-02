#!/usr/bin/env python3
"""Curate CMS v2, v3 of charged particles up to pT = 100 GeV in PbPb at 5.02 TeV, INSPIRE 1511868.

A. M. Sirunyan et al. (CMS), "Azimuthal anisotropy of charged particles
with transverse momentum up to 100 GeV/c in PbPb collisions at sqrt(sNN) =
5.02 TeV", PLB 776 (2017) 195, arXiv:1702.00630.

From the paper (Secs. 2-4):
- particles of interest: tracks |eta| < 1; centrality from the HF Sum E_T
  (2.9 < |eta| < 5.2);
- scalar product: Q_n^A from the HF on the side opposite the particle
  (3 < |eta| < 5, E_T-weighted; every particle), normalized with the
  other HF and the tracker (|eta| < 0.75): mirrored HF reference here,
  unit weights;
- v2{4}, {6}, {8} from multiparticle cumulants (reference particles not
  stated in the paper);
- errors: statistical and systematic.

Records (public): v2{SP}(pT) and v3{SP}(pT), 0-5 ... 50-60% (data1-14).
The tables give the mean pT of each bin, not its edges: bin edges at the
midpoints between neighbouring points (the outer edges mirrored).  Staged
(--staged OUT): v2{4}, v2{6}, v2{8} (data15-20; reference selection not
stated, differential v{6}, v{8} not implemented).  Not curated: v2(high
pT) vs v2(low pT) (data21-22).  The old records had |eta| < 2.4 and no
flow_reference.

Run from the repository root:
    python scripts/curate_cms_vn_highpt_pbpb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1511868/HEPData-ins1511868-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. M. Sirunyan et al. (CMS), PLB 776 (2017) 195, arXiv:1702.00630'
HF = {'estimator': {'name': 'HF', 'observable': 'ET', 'detector': 'HF'},
      'eta': [{'min': -5.2, 'max': -2.9}, {'min': 2.9, 'max': 5.2}]}
HF_REF = {'detector': 'HF', 'mirror': True, 'eta': [{'min': 3.0, 'max': 5.0}], 'pT': {'min': 0.0, 'max': 100.0},
          'notes': 'HF on the side opposite the particle, 3 < |eta| < 5 (E_T-weighted in the paper; unit weights '
                   'here). Reference pT: all particles.'}


def edges_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    lo = max(0.0, pts[0] - (mids[0] - pts[0]))
    hi = pts[-1] + (pts[-1] - mids[-1])
    return [round(x, 4) for x in [lo] + mids + [hi]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the cumulant records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = {x['data_file']: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    public, staged = [], []
    for k in range(1, 21):
        hf = f'data{k}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            cls = (q.get('CENTRALITY') or q['centrality']).replace(' PCT', '%')
            m = re.search(r'v_\{(\d)\}\\\{(SP|\d)\\\}', h)
            n, method = int(m.group(1)), m.group(2)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pts_all = [float(v['value']) for v in iv['values']]
            edges = edges_from_points(pts_all)
            pairs = [[edges[j], edges[j + 1]] for j in rows]
            d = {'observable_type': 'flow_vn', 'inspire_id': '1511868', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{' '.join(sub[hf]['description'].split())[:140]} | {h}",
                 'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
                 'centrality': {'classes': [cls], **HF}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                     'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'harmonic': n, 'value_type': h, 'dimensionality': 'differential_pT'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            edge_note = 'bin edges at the midpoints between the tabulated mean pT (edges not published)'
            if method == 'SP':
                d.update({'method': 'scalar_product', 'flow_reference': dict(HF_REF),
                          'notes': {'value': f'{CITE}. v{n}{{SP}}(pT) of charged particles, {cls}, |eta| < 1, HF '
                                             f'reference opposite the particle (|Delta eta| > 2); {edge_note}. Errors: '
                                             'statistical and systematic.'}})
                public.append(d)
            else:
                d.update({'method': 'cumulant', 'cumulant_order': int(method),
                          'notes': {'value': f'{CITE}. v{n}{{{method}}}(pT) of charged particles, {cls}, |eta| < 1, '
                                             f'multiparticle cumulants; {edge_note}. Errors: statistical and '
                                             'systematic.'},
                          'staging_reason': 'the cumulants\' reference particles are not stated in the paper'
                                            + ('' if method == '4' else '; differential v{6}, v{8} not implemented')})
                staged.append(d)
    old = 'data/flow_vn/cms_flow_vn_azimuthal_anisotropy_of_charged_particles_PbPb_5020gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/flow_vn/cms_flow_vn_v2_v3_highpt_PbPb_5020gev_ins1511868.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['harmonic'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
