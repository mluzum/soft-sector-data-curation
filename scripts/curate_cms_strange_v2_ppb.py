#!/usr/bin/env python3
"""Curate CMS v2{4} and v2{SP} of charged hadrons, K0_S and Lambda in pPb at 8.16 TeV, INSPIRE 2075415.

CMS Collaboration, "Strange hadron collectivity in pPb and PbPb collisions",
JHEP 05 (2023) 007, arXiv:2205.00080 (the PbPb records:
scripts/curate_cms_strange_v2_pbpb.py).

From the paper (Sec. 2-4), pPb at 8.16 TeV (2016):
- both beam directions merged, "the proton-going direction defines
  positive pseudorapidity"; eta_cm = 0 is eta_lab = 0.465 toward the
  proton: the records are in the nn frame (rapidity_shift 0.465, the
  proton forward);
- classes of N_trk^offline (primary tracks |eta_lab| < 2.4, pT > 0.4 GeV,
  uncorrected): 120-150, 150-185, 185-250, >= 250, "identical to that used
  in Ref. [48]" (CMS PLB 724 (2013) 213), whose Table 1 gives the pPb
  <N_offline>/<N_corrected> = 0.83 (132/159, 162/195, ..., 232/280);
- charged hadrons |eta_lab| < 1, K0_S and Lambda + Lambdabar |y_lab| < 1
  (curated in eta, as the PbPb records);
- v2{4}: a particle of interest with three reference particles, charged
  |eta_lab| < 2.4, 0.5 < pT < 5 GeV; v2{SP}: the scalar product with the HF
  Q-vector (3 < |eta_lab| < 5) on the p-going (p-SP) or the Pb-going side
  (Pb-SP), separately;
- errors: stat and sys.

Records (public): v2{4}(pT), v2{p-SP}(pT) and v2{Pb-SP}(pT) of the three
species per class (Fig. 3), windows mapped to eta_cm = eta_lab - 0.465:
particles of interest -1.465 < eta_cm < 0.535, cumulant reference
-2.865 < eta_cm < 1.935, HF p-going 2.535-4.535 and Pb-going -5.465 to
-3.465, the estimator -2.865 < eta_cm < 1.935.  Staged (--staged OUT):
v2{6}, v2{8} (differential POI v{6}, v{8}: not computed), the jet-veto
cumulants and their ratios (events with jets removed: not modelled),
v2{4,Sub} and its ratio (subevent POI v{4}: not computed), sigma/<v2>
(derived).

Run from the repository root:  python scripts/curate_cms_strange_v2_ppb.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2075415/HEPData-ins2075415-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'CMS Collaboration, JHEP 05 (2023) 007, arXiv:2205.00080'
OUT = 'data/flow_vn/cms_flow_vn_strange_v2_pPb_8160gev_ins2075415.yaml'
SHIFT = 0.465


def cm(lo, hi):
    return {'min': round(lo - SHIFT, 3), 'max': round(hi - SHIFT, 3)}


EST = {'estimator': {'name': 'Ntrk_offline', 'observable': 'Nch', 'detector': 'Tracker'},
       'eta': [cm(-2.4, 2.4)], 'pT': {'min': 0.4, 'max': 100.0},
       'multiplicity_bins': {'efficiency': 0.83,
                             'notes': 'pPb <N_trk^offline>/<N_trk^corrected> of CMS PLB 724 (2013) 213 Table 1, the '
                                      'classification the paper uses (132/159 ... 232/280).'},
       'notes': 'N_trk^offline: primary tracks |eta_lab| < 2.4, pT > 0.4 GeV, uncorrected; lower bounds inclusive, the '
                'top class open (here to 1000); in the nn frame -2.865 < eta_cm < 1.935.'}
REF_CUMULANT = {'eta': [cm(-2.4, 2.4)], 'pT': {'min': 0.5, 'max': 5.0},
                'notes': 'reference particles of the cumulants: charged hadrons |eta_lab| < 2.4, 0.5 < pT < 5 GeV.'}
REF_HF = {'p': {'detector': 'HF', 'eta': [cm(3.0, 5.0)], 'pT': {'min': 0.0, 'max': 100.0},
                'notes': 'HF on the p-going side, 3 < eta_lab < 5 (E_T-weighted in the paper; unit weights here).'},
          'Pb': {'detector': 'HF', 'eta': [cm(-5.0, -3.0)], 'pT': {'min': 0.0, 'max': 100.0},
                 'notes': 'HF on the Pb-going side, -5 < eta_lab < -3 (E_T-weighted in the paper; unit weights '
                          'here).'}}
SPECIES = {'CHARGED': ('charged', 'CHARGED'), '$K_S^0$': ('K0_S', 'K0S'),
           '$\\Lambda$': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR')}
COMPUTED = {'$v_2\\{4\\}$': ('4', None), '$v_2\\{p-SP\\}$': ('SP', 'p'), '$v_2\\{Pb-SP\\}$': ('SP', 'Pb')}


def cls_of(s):
    m = re.search(r'(\d+)\s*\\le N_\{trk\}\^\{offline\}\s*<\s*(\d+)', s)
    if m:
        return f'{m.group(1)}-{m.group(2)}'
    m = re.search(r'N_\{trk\}\^\{offline\}\s*\\ge\s*(\d+)', s)
    return f'{m.group(1)}-1000'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        if 'pPb' not in desc:
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']] \
            if t['independent_variables'][0]['values'][0].get('low') is not None else None
        for k, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            q = {x['name']: str(x['value']) for x in dv.get('qualifiers', [])}
            if name not in COMPUTED:
                staged.append({'observable_type': 'flow_vn', 'inspire_id': '2075415', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': k,
                               'hepdata_table_label': f'{name}: {desc[:110]} ({hf} column {k})', 'value_type': name,
                               'notes': {'value': f'{CITE}. {desc}'},
                               'staging_reason': ('v2{6}, v2{8} of a particle of interest: not computed'
                                                  if re.search(r'\\\{[68]\\\}', name) and 'veto' not in name else
                                                  'jet-veto cumulant (events with jets removed): not modelled'
                                                  if 'veto' in name else
                                                  'v2{4,Sub} (subevent POI cumulant): not computed' if 'Sub' in name
                                                  else 'derived quantity (ratio or sigma/<v2>)')})
                continue
            meth, side = COMPUTED[name]
            assert q['SQRT(S_NN)'] == '8160.0', q
            species, prod = SPECIES[re.fullmatch(r'P PB --> (.+) X', q['RE']).group(1)]
            assert q.get('ABS(ETA_LAB)', q.get('ABS(RAPIDITY_LAB)')) == '<1.0', q
            cls = cls_of(q['$N_{trk}^{offline}$'])
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-', '--')]
            assert {tuple(e['label'] for e in dv['values'][j]['errors']) for j in rows} == {('stat', 'sys')}, hf
            label = {'4': 'v2{4}', 'SP': f'v2{{{side}-SP}}'}[meth]
            d = {'observable_type': 'flow_vn', 'inspire_id': '2075415', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': k,
                 'hepdata_table_label': f'{label}(pT) of {species}, pPb 8.16 TeV, N_trk^offline {cls} ({hf})',
                 'collision_system': {'energy_gev': 8160.0, 'reaction': f'P PB --> {prod} X', 'frame': 'nn_cms',
                                      'rapidity_shift': SHIFT, 'forward_beam': 'p'},
                 'centrality': {'classes': [cls], **EST},
                 'particle_id': {'species': species},
                 'kinematic_range': {'eta': cm(-1.0, 1.0),
                                     'pT': {'min': bins[rows[0]][0], 'max': bins[rows[-1]][1],
                                            'bin_ranges': [bins[j] for j in rows]}},
                 'harmonic': 2, 'value_type': 'v2', 'dimensionality': 'differential_pT'}
            if len(rows) < len(bins):
                d['table_rows'] = rows
            if meth == '4':
                d.update(method='cumulant', cumulant_order=4, flow_reference=dict(REF_CUMULANT))
            else:
                d.update(method='scalar_product', flow_reference=dict(REF_HF[side]))
            d['notes'] = {'value': f'{CITE}, Fig. 3. {label}(pT) of {species}, '
                                   f'{"|eta_lab|" if species == "charged" else "|y_lab|"} < 1 (here -1.465 < '
                                   f'eta_cm < 0.535), pPb 8.16 TeV, N_trk^offline {cls}; '
                                   + ('reference charged hadrons |eta_lab| < 2.4, 0.5 < pT < 5 GeV' if meth == '4'
                                      else f'HF reference on the {side}-going side') + '. Errors: stat and sys.'}
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
