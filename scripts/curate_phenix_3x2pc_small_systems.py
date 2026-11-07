#!/usr/bin/env python3
"""Curate PHENIX v2, v3(pT) from three sets of two-particle correlations in p/d/3He+Au, INSPIRE 2026169.

U. Acharya et al. (PHENIX), "Kinematic dependence of azimuthal
anisotropies in p+Au, d+Au, and 3He+Au at sqrt(sNN) = 200 GeV", PRC 105
(2022) 024901, arXiv:2107.06634.

From the paper (Sec. II, Tables III, IV, Eqs. 3-6):
- 0-5% centrality from the BBC charge on the Au-going side (BBCS, -3.9 <
  eta < -3.1); the light nucleus toward +eta, Au toward -eta; lab frame =
  nucleon-nucleon centre of mass (equal beam energies per nucleon);
- POI: charged hadrons in the central arms (CNT), |eta| < 0.35, in pT bins;
- v_n^C(pT) = sqrt(c^AC(pT) c^BC(pT) / c^AB) (Eq. 6), the correlations of
  the CNT with two other detectors and theirs with each other, the latter
  over all tracks or hits; detector combinations BBCS-FVTXS-CNT and
  FVTXS-CNT-FVTXN;
- the 3x2PC uses FVTX tracks (1.2 < |eta| < 2.2, no momentum information;
  acceptance mostly pT > 0.5 GeV/c) and BBC tubes (light output normalized
  to a charged particle; about half the hits from secondaries).
- errors stat and sys (no nonflow estimate subtracted: 'all physics
  contributions included').

Records (public): v2 and v3 vs pT, 0-5%, in both combinations (12), flow_vn
method three_subevent_2pc (analysis/three_sub_2pc.py; references all pT;
BBC light weighting and secondaries not modelled).  Staged (--staged OUT):
the BBCS-FVTXS-CNT columns repeated in Figs. 5 and 7, the Fig. 6 ratios,
the c_n(eta) of Fig. 8, the correlation functions (Figs. 13-18) and the
supplemental c_n tables (detector pairs incl. p+p minimum bias, nonflow
included).

Run from the repository root:  python scripts/curate_phenix_3x2pc_small_systems.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2026169/HEPData-ins2026169-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'U. Acharya et al. (PHENIX), PRC 105 (2022) 024901, arXiv:2107.06634'
OUT = 'data/flow_vn/phenix_flow_vn_3x2pc_pAu_dAu_He3Au_200gev_ins2026169.yaml'
SYSTEMS = {'p+Au': ('P AU --> CHARGED X', 'p'), 'd+Au': ('D AU --> CHARGED X', 'd'),
           '3He+Au': ('HE3 AU --> CHARGED X', '3He')}
DET = {'BBCS': {'eta': [{'min': -3.9, 'max': -3.1}], 'pT': {'min': 0.0, 'max': 100.0}},
       'FVTXS': {'eta': [{'min': -2.2, 'max': -1.2}], 'pT': {'min': 0.0, 'max': 100.0}},
       'FVTXN': {'eta': [{'min': 1.2, 'max': 2.2}], 'pT': {'min': 0.0, 'max': 100.0}}}
COMBOS = {'BBCS-FVTXS-CNT': ('BBCS', 'FVTXS'), 'FVTXS-CNT-FVTXN': ('FVTXS', 'FVTXN')}
CENT = {'classes': ['0-5%'], 'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
        'eta': [{'min': -3.9, 'max': -3.1}], 'notes': 'Total charge in the BBC on the Au-going side.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    subs = {x['data_file']: ' '.join(x['description'].split())
            for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x}
    public, staged = [], []
    for hf, desc in subs.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        m = re.match(r'fig([3457])[abc]\.yaml', hf)
        for i, dv in enumerate(t['dependent_variables']):
            label = dv['qualifiers'][0]['value'] if dv.get('qualifiers') else desc
            if not m or (m.group(1) in '57' and i > 0):
                why = ('the BBCS-FVTXS-CNT values of Figs. 3, 4 repeated' if m else
                       'v2 ratio between detector combinations' if hf.startswith('fig6') else
                       'c_n(eta) of detector pairs' if hf.startswith('fig8') else
                       'two-particle correlation function C(Delta phi)' if hf.startswith('fig1') else
                       'supplemental c_n of detector pairs (p+p minimum bias included, nonflow included)')
                staged.append({'observable_type': 'flow_vn', 'inspire_id': '2026169', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{desc} ({label})',
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc}.'},
                               'staging_reason': why})
                continue
            sm = re.match(r'(p|d|3He)\+Au at 200 GeV, 0-5% central, ([A-Z-]+)$', label)
            assert sm, (hf, label)
            system, combo = f'{sm.group(1)}+Au', sm.group(2)
            n = int(dv['header']['name'].strip('$v_{}'))
            b, c = COMBOS[combo]
            bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'sys')}, hf
            rb = [bins[r] for r in rows]
            reaction, beam = SYSTEMS[system]
            d = {'observable_type': 'flow_vn', 'inspire_id': '2026169', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{desc} ({label})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': reaction, 'frame': 'lab',
                                      'rapidity_shift': 0.0, 'forward_beam': beam},
                 'centrality': dict(CENT), 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                     'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
                 'harmonic': n, 'method': 'three_subevent_2pc', 'dimensionality': 'differential_pT',
                 'flow_reference': {'subevents': {'B': {k: (list(v) if k == 'eta' else dict(v))
                                                        for k, v in DET[b].items()},
                                                  'C': {k: (list(v) if k == 'eta' else dict(v))
                                                        for k, v in DET[c].items()}},
                                    'detector': f'{b}, {c}',
                                    'notes': f'{combo}: v_{n}(CNT) = sqrt(c(CNT, {b}) c(CNT, {c}) / c({b}, {c})); '
                                             'FVTX tracks 1.2 < |eta| < 2.2, BBCS tubes -3.9 < eta < -3.1, all pT '
                                             '(BBC light weighting and secondaries not modelled).'},
                 'value_type': f'v{n}'}
            if rows != list(range(len(bins))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. v{n}(pT) of charged hadrons at |eta| < 0.35 from three two-particle '
                                   f'correlations ({combo}, Eq. 6), {system} 200 GeV 0-5% (BBCS); nonflow not '
                                   'subtracted. Errors: stat and sys.'}
            public.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
