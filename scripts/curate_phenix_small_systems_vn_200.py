#!/usr/bin/env python3
"""Curate PHENIX v2, v3 in 0-5% p+Au, d+Au, 3He+Au at 200 GeV, INSPIRE 1672133.

C. Aidala et al. (PHENIX), "Creating small circular, elliptical, and
triangular droplets of quark-gluon plasma", Nature Phys. 15 (2019) 214,
arXiv:1805.02973.

From the paper:
- charged hadrons at |eta| < 0.35, event-plane method;
- 2nd-order event plane in the Au-going direction: -3.0 < eta < -1.0
  (FVTX south) in p+Au and d+Au, -3.9 < eta < -3.1 (BBC south) in
  3He+Au; 3rd-order event plane: BBC south in all systems;
- centrality: total charge in the BBC south (-3.9 < eta < -3.1), 0-5%;
- Au goes toward eta < 0 (south), the light projectile toward +eta; RHIC
  beams of equal energy per nucleon: lab frame = nucleon-nucleon frame;
- errors: stat and sys.

Records (public): v2(pT) and v3(pT) of the three systems (Fig. 2).  Not
curated: Fig. 3 (SONIC, iEBE-VISHNU and MSTV model curves) and Fig. 4
(one series labelled "p+Au 20-40%"; the paper's Fig. 4 shows previously
published d+Au 20-40% and p+Au 0-5%).  The old records called every
system d+Au and the species K-, and included the model curves.

Run from the repository root:  python scripts/curate_phenix_small_systems_vn_200.py
"""
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1672133/HEPData-ins1672133-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Aidala et al. (PHENIX), Nature Phys. 15 (2019) 214, arXiv:1805.02973'
BBCS = [{'min': -3.9, 'max': -3.1}]
FVTXS = [{'min': -3.0, 'max': -1.0}]
SYSTEMS = {'fig2-1.yaml': ('P AU', 'p', 2), 'fig2-2.yaml': ('D AU', 'd', 2), 'fig2-3.yaml': ('HE3 AU', '3He', 2),
           'fig2-4.yaml': ('P AU', 'p', 3), 'fig2-5.yaml': ('D AU', 'd', 3), 'fig2-6.yaml': ('HE3 AU', '3He', 3)}
OLD = 'data/flow_vn/phenix_flow_vn_creating_small_circular_elliptical_and_AuAu_200gev.yaml'


def main():
    docs = []
    for hf, (beams, proj, n) in SYSTEMS.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        assert dv['header']['name'] == f'$v_{n}$ 0-5%', (hf, dv['header'])
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        contiguous = all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:]))
        pt = ({'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [p[0] for p in pairs] + [pairs[-1][1]]}
              if contiguous else {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs})
        fvtx = n == 2 and proj != '3He'
        ref = {'detector': 'FVTX' if fvtx else 'BBC', 'eta': FVTXS if fvtx else BBCS, 'pT': {'min': 0.0, 'max': 100.0},
               'notes': ('FVTX south' if fvtx else 'BBC south') + ' (Au-going side). Reference pT: all particles.'}
        system = proj + '+Au'
        d = {'observable_type': 'flow_vn', 'inspire_id': '1672133', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': 0, 'hepdata_table_label': f'v{n}(pT), 0-5% {system} ({hf})',
             'collision_system': {'energy_gev': 200.0, 'reaction': f'{beams} --> CHARGED X', 'frame': 'lab',
                                  'rapidity_shift': 0.0, 'forward_beam': proj},
             'centrality': {'classes': ['0-5%'],
                            'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
                            'eta': [dict(w) for w in BBCS],
                            'notes': 'Total charge in the south BBC (Au-going side).'},
             'particle_id': {'species': 'charged'},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35}, 'pT': pt},
             'harmonic': n, 'method': 'event_plane', 'value_type': f'v_{n}', 'dimensionality': 'differential_pT',
             'flow_reference': ref,
             'notes': {'value': f"{CITE}. v{n}(pT) of charged hadrons, 0-5% {system}, |eta| < 0.35, event plane "
                                f"{ref['notes'].split(' (')[0]} ({ref['eta'][0]['min']} < eta < {ref['eta'][0]['max']}); "
                                'computed with the scalar-product estimator. Lab frame (Au toward -eta). Errors: stat '
                                'and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        docs.append(d)
    if os.path.exists(OLD):
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_v2_v3_pAu_dAu_He3Au_200gev_ins1672133.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
