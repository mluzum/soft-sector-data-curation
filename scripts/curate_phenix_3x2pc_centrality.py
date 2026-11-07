#!/usr/bin/env python3
"""Curate PHENIX v2{BB}, v2{BF} vs pT and centrality in p+Au, d+Au, 3He+Au, INSPIRE 2054927.

U. Acharya et al. (PHENIX), "Measurements of second-harmonic Fourier
coefficients from azimuthal anisotropies in p+p, p+Au, d+Au, and 3He+Au
collisions at sqrt(sNN) = 200 GeV", PRC 107 (2023) 024907, arXiv:2203.09894.

From the paper (Sec. II):
- centrality from the BBC on the Au-going side (BBCS, -3.9 < eta < -3.1);
  the light nucleus toward +eta; lab frame = nucleon-nucleon centre of mass;
- charged hadrons in the central arms (CNT), |eta| < 0.35, 0.2 < pT < 4
  GeV/c; FVTX tracks 1.2 < |eta| < 2.2 (pT > 0.3 GeV/c, no momentum
  measurement); BBCS tubes;
- three sets of two-particle correlations (3x2PC, Eqs. 5-8): v2{BB} from
  BBCS-FVTXS-CNT (two backward detectors), v2{BF} from FVTXS-CNT-FVTXN;
- v2 vs pT in centrality classes (Figs. 4-6) and vs centrality for 0.5 <
  pT < 1 and 2 < pT < 2.5 GeV/c (Fig. 8); errors stat and syst.

Records (public): flow_vn method three_subevent_2pc (analysis/
three_sub_2pc.py).  pT bins: 0.2, 0.5, 0.75, ..., 2.0, 2.5, 3.0, 4.0 GeV/c
(the tabulated points are their centres).  Staged (--staged OUT): p+p
minimum bias (Fig. 7, Fig. 9.3, 10.3), v2 vs dN_ch/deta (Fig. 9: the Fig. 8
values on a multiplicity axis) and the ratios v2{BF}/v2{BB} (Fig. 10).

Run from the repository root:  python scripts/curate_phenix_3x2pc_centrality.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2054927/HEPData-ins2054927-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'U. Acharya et al. (PHENIX), PRC 107 (2023) 024907, arXiv:2203.09894'
OUT = 'data/flow_vn/phenix_flow_vn_3x2pc_centrality_pAu_dAu_He3Au_200gev_ins2054927.yaml'
EDGES = [0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
FIG_SYS = {'4': 'p+Au', '5': 'd+Au', '6': '3He+Au', '8.0': 'p+Au', '8.1': 'd+Au', '8.2': '3He+Au'}
SYSTEMS = {'p+Au': ('P AU --> CHARGED X', 'p'), 'd+Au': ('D AU --> CHARGED X', 'd'),
           '3He+Au': ('HE3 AU --> CHARGED X', '3He')}
DET = {'BBCS': [{'min': -3.9, 'max': -3.1}], 'FVTXS': [{'min': -2.2, 'max': -1.2}],
       'FVTXN': [{'min': 1.2, 'max': 2.2}]}
COMBOS = {'BB': ('BBCS', 'FVTXS', 'BBCS-FVTXS-CNT'), 'BF': ('FVTXS', 'FVTXN', 'FVTXS-CNT-FVTXN')}
CENT = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
        'eta': [{'min': -3.9, 'max': -3.1}], 'notes': 'BBC hits on the Au-going side.'}


def base(hf, i, label, system, combo, classes):
    reaction, beam = SYSTEMS[system]
    b, c, name = COMBOS[combo]
    return {'observable_type': 'flow_vn', 'inspire_id': '2054927', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 200.0, 'reaction': reaction, 'frame': 'lab', 'rapidity_shift': 0.0,
                                 'forward_beam': beam},
            'centrality': {'classes': classes, **CENT}, 'particle_id': {'species': 'charged'},
            'harmonic': 2, 'method': 'three_subevent_2pc',
            'flow_reference': {'subevents': {'B': {'eta': [dict(e) for e in DET[b]], 'pT': {'min': 0.0, 'max': 100.0}},
                                             'C': {'eta': [dict(e) for e in DET[c]], 'pT': {'min': 0.0, 'max': 100.0}}},
                               'detector': f'{b}, {c}',
                               'notes': f'v2{{{combo}}}: {name}, v2(CNT) = sqrt(c(CNT, {b}) c(CNT, {c}) / c({b}, {c})); '
                                        'FVTX tracks 1.2 < |eta| < 2.2 (pT > 0.3 GeV/c in the detector), BBCS tubes '
                                        '-3.9 < eta < -3.1 (light weighting and secondaries not modelled).'},
            'value_type': f'v2{{{combo}}}'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    subs = {x['data_file']: ' '.join(x['description'].split())
            for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x}
    public, staged = [], []
    for hf, desc in subs.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        fig = re.match(r'Fig([\d.]+)_formatted\.yaml', hf).group(1)
        key = fig.split('.')[0] if fig.split('.')[0] in '456' else fig
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            label = f'{desc} ({name})'
            if key not in FIG_SYS:
                why = ('p+p minimum bias' if 'p$+$p' in desc else
                       'the Fig. 8 values vs dN_ch/deta' if fig.startswith('9') else
                       'ratio v2{BF}/v2{BB}' if fig.startswith('10') else 'not curated')
                staged.append({'observable_type': 'flow_vn', 'inspire_id': '2054927', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': label, 'value_type': name,
                               'notes': {'value': f'{CITE}. {desc}.'}, 'staging_reason': why})
                continue
            system = FIG_SYS[key]
            combo = re.search(r'\\\{(B[BF])\\\}', name).group(1)
            assert all({e['label'] for e in v['errors']} == {'stat', 'syst'} for v in dv['values']
                       if str(v['value']).strip() not in ('', '-')), (hf, i)
            if key in '456':
                cls = re.search(r'centrality (\d+-\d+%)', name).group(1)
                pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
                assert all(abs(p - (a + b) / 2) < 1e-9 for p, a, b in zip(pts, EDGES[:-1], EDGES[1:])), (hf, pts)
                rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
                rb = [[EDGES[r], EDGES[r + 1]] for r in rows]
                d = base(hf, i, label, system, combo, [cls])
                d['kinematic_range'] = {'eta': {'min': -0.35, 'max': 0.35},
                                        'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}}
                d['dimensionality'] = 'differential_pT'
                what = f'v2{{{combo}}}(pT), {system} 200 GeV {cls}'
            else:
                lo, hi = (float(x) for x in re.search(r'\$([\d.]+)<p_T<([\d.]+)\$', name).groups())
                cent = [(int(v['low']), int(v['high'])) for v in t['independent_variables'][0]['values']]
                rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
                d = base(hf, i, label, system, combo, [f'{a}-{b}%' for a, b in (cent[r] for r in rows)])
                d['kinematic_range'] = {'eta': {'min': -0.35, 'max': 0.35}, 'pT': {'min': lo, 'max': hi}}
                d['dimensionality'] = 'integrated'
                what = f'v2{{{combo}}} for {lo:g} < pT < {hi:g} GeV/c vs centrality, {system} 200 GeV'
            if rows != list(range(len(dv['values']))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. {what}: charged hadrons at |eta| < 0.35 from three two-particle '
                                   f'correlations ({COMBOS[combo][2]}, Eqs. 5-8); nonflow not subtracted. Errors: '
                                   'stat and syst.'}
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
