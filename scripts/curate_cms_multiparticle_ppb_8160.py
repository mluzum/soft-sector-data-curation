#!/usr/bin/env python3
"""Curate CMS multiparticle cumulants in pPb at 8.16 TeV and PbPb at 5.02 TeV vs N_trk^offline, INSPIRE 1731568.

A. M. Sirunyan et al. (CMS), "Multiparticle correlation studies in pPb
collisions at sqrt(sNN) = 8.16 TeV", PRC 101 (2020) 014912,
arXiv:1904.11519.

From the paper (Secs. 2-4):
- tracks |eta_lab| < 2.4, 0.3 < pT < 3.0 GeV/c; classes of N_trk^offline
  (primary tracks |eta| < 2.4, pT > 0.4 GeV/c, uncorrected), the
  classification of CMS PLB 724 (2013) 213, whose Table 1 gives
  <N_trk^offline>/<N_trk^corrected> = 0.83 (pPb: 132/159 ...) and 0.80
  (PbPb: 134/168 ...); the PbPb sample reprocessed with the pPb
  reconstruction; both pPb beam directions;
- v_n{4}, v_n{6}, v_n{8} from the generic framework over all particles of
  the window (Eqs. 1-3, no gap);
- v_n{2}: two-particle correlations with |Delta eta| > 2 (CMS PRL 120
  (2018) 092301): v2^sub{2}, with the low-multiplicity (jet) subtraction,
  and v3{2} without it.

Records (public): v2{4}, v2{6}, v2{8}, v3{4} vs N_trk^offline in pPb and
PbPb (Tables 1-8); v2{4}/v2{2}, v3{4}/v3{2} (Tables 9-12) and
v2{6}/v2{4}, v2{8}/v2{6} (Tables 13-14, first column) as
cumulant_combination ratios, the v{2} term a pair gap |Delta eta| > 2.
pPb windows in the nn frame (p toward +eta, y_lab = y_cm + 0.465, as CMS
2075415 for the same data): |eta_lab| < 2.4 is -2.865 < eta_cm < 1.935.
Staged (--staged OUT): the v2{4}/v2{2} abscissa of Tables 13-14 (Fig. 3),
which differs from Table 11 (0.80 vs 0.88 at N_trk 90-120) with no stated
v2{2}.

Run from the repository root:  python scripts/curate_cms_multiparticle_ppb_8160.py --staged OUT.yaml
"""
import argparse
import copy
import glob
import os

import yaml

IID = '1731568'
RAW = glob.glob(f'hepdata_raw/records/ins{IID}/HEPData-ins{IID}-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. M. Sirunyan et al. (CMS), PRC 101 (2020) 014912, arXiv:1904.11519'
OUT = f'data/flow_vn/cms_flow_vn_multiparticle_pPb_8160gev_PbPb_5020gev_ins{IID}.yaml'
PT = {'min': 0.3, 'max': 3.0}
SYS = {
    'PbPb': {'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
             'eta': (-2.4, 2.4), 'eff': 0.80,
             'eff_note': 'PbPb <N_trk^offline>/<N_trk^corrected> of CMS PLB 724 (2013) 213 Table 1 (134/168, '
                         '109/137, ...: 0.80), the classification the paper uses.'},
    'pPb': {'collision_system': {'energy_gev': 8160.0, 'reaction': 'P PB --> CHARGED X', 'frame': 'nn_cms',
                                 'rapidity_shift': 0.465, 'forward_beam': 'p'},
            'eta': (-2.865, 1.935), 'eff': 0.83,
            'eff_note': 'pPb <N_trk^offline>/<N_trk^corrected> of CMS PLB 724 (2013) 213 Table 1 (132/159, '
                        '109/132, ...: 0.83), the classification the paper uses.'},
}
# file -> (system, kind, order(s), harmonic)
TABLES = {'AAV24': ('PbPb', 'v', 4, 2), 'AAV26': ('PbPb', 'v', 6, 2), 'AAV28': ('PbPb', 'v', 8, 2),
          'AAV34': ('PbPb', 'v', 4, 3), 'PAV24': ('pPb', 'v', 4, 2), 'PAV26': ('pPb', 'v', 6, 2),
          'PAV28': ('pPb', 'v', 8, 2), 'PAV34': ('pPb', 'v', 4, 3),
          'AAV242': ('PbPb', 'r', [4, 2], 2), 'AAV342': ('PbPb', 'r', [4, 2], 3),
          'PAV242': ('pPb', 'r', [4, 2], 2), 'PAV342': ('pPb', 'r', [4, 2], 3),
          'PAV2RS64': ('pPb', 'r', [6, 4], 2), 'PAV2RS86': ('pPb', 'r', [8, 6], 2)}
HEADER = {'v': 'v{h}{k}', 'r': None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = {x['data_file'][:-5]: x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml')))
           if x and 'data_file' in x}
    docs, staged = [], []
    for name, (sysname, kind, order, h) in TABLES.items():
        x = sub[name]
        t = yaml.safe_load(open(os.path.join(RAW, f'{name}.yaml')))
        iv = t['independent_variables'][0]
        assert iv['header']['name'] == 'Ntrk', name
        classes = [f"{b['low']:g}-{b['high']:g}" for b in iv['values']]
        S = SYS[sysname]
        lab = 'pPb 8.16 TeV' if sysname == 'pPb' else 'PbPb 5.02 TeV'
        cent = {'classes': classes,
                'estimator': {'name': 'Ntrk_offline', 'observable': 'Nch', 'detector': 'Tracker'},
                'eta': [{'min': S['eta'][0], 'max': S['eta'][1]}], 'pT': {'min': 0.4, 'max': 100.0},
                'multiplicity_bins': {'efficiency': S['eff'], 'notes': S['eff_note']},
                'notes': 'N_trk^offline: primary tracks |eta_lab| < 2.4, pT > 0.4 GeV, uncorrected; lower bounds '
                         'inclusive.' + (' In the nn frame -2.865 < eta_cm < 1.935.' if sysname == 'pPb' else '')}
        for j, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            assert q['PT'] == '0.3 - 3.0 GeV', (name, q)
            assert all(str(v['value']).strip() not in ('', '-') for v in dv['values']), name
            hname = dv['header']['name']
            base = {'observable_type': 'flow_vn', 'inspire_id': IID, 'version': VERSION,
                    'hepdata_file': f'{name}.yaml', 'dv_index': j,
                    'hepdata_table_label': f"{x['name']}: {' '.join(x['description'].split())} ({hname})",
                    'collision_system': copy.deepcopy(S['collision_system']), 'centrality': copy.deepcopy(cent),
                    'particle_id': {'species': 'charged'},
                    'kinematic_range': {'eta': {'min': S['eta'][0], 'max': S['eta'][1]}, 'pT': dict(PT)},
                    'harmonic': h, 'method': 'cumulant', 'dimensionality': 'integrated', 'value_type': hname}
            if kind == 'v':
                assert hname == f'v{h}{order}', (name, hname)
                base.update(cumulant_order=order,
                            notes={'value': f'{CITE}, Fig. 1. v{h}{{{order}}} of charged particles |eta_lab| < 2.4, '
                                            f'0.3 < pT < 3.0 GeV/c, all particles of the window, {lab}, vs '
                                            'N_trk^offline. Errors: stat and sys.'})
                docs.append(base)
                continue
            if hname == 'v2rs42' and name.startswith('PAV2RS'):
                staged.append(dict(base, cumulant_order=4,
                                   staging_reason='v2{4}/v2{2}, the abscissa of Fig. 3: differs from Table 11 '
                                                  '(0.80 vs 0.88 at N_trk 90-120) and the paper does not state its '
                                                  'v2{2}'))
                continue
            a, b = order
            assert hname == {(4, 2): f'v{h}42', (6, 4): 'v2rs64', (8, 6): 'v2rs86'}[(a, b)], (name, hname)
            comb = {'kind': 'ratio', 'orders': [a, b]}
            if b == 2:
                comb['two_particle'] = {'eta_gap': 2.0}
            gap_note = (' v2{2}: two-particle correlations |Delta eta| > 2 with the low-multiplicity (jet) '
                        'subtraction (v2^sub{2}), which leaves the flow a model without non-flow computes'
                        if h == 2 else ' v3{2}: two-particle correlations |Delta eta| > 2, without the '
                                       'low-multiplicity subtraction (residual away-side non-flow in the data)')
            base.update(cumulant_order=max(a, b), cumulant_combination=comb,
                        notes={'value': f'{CITE}, Fig. {2 if b == 2 else 3}. v{h}{{{a}}}/v{h}{{{b}}} of charged '
                                        f'particles |eta_lab| < 2.4, 0.3 < pT < 3.0 GeV/c, {lab}, vs N_trk^offline.'
                                        + (gap_note + '.' if b == 2 else '') + ' Errors: stat and sys.'})
            docs.append(base)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
