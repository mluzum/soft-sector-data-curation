#!/usr/bin/env python3
"""Curate ALICE v2 of h, pi, K, p in p-Pb at 5.02 TeV, INSPIRE 1242302.

B. Abelev et al. (ALICE), "Long-range angular correlations of pi, K and p
in p-Pb collisions at sqrt(sNN) = 5.02 TeV", PLB 726 (2013) 164,
arXiv:1307.3237.

From the paper (Secs. 2, 4):
- tracks 0.3 < pT < 4 GeV/c, |eta_lab| < 0.8; pi, K, p (charges summed)
  by TPC dE/dx and TOF;
- multiplicity classes from the VZERO-A amplitude (2.8 < eta_lab < 5.1,
  the Pb flight direction): 0-20, 20-40, 40-60, 60-100%;
- 2013 beams: the nucleon-nucleon cms moves by 0.465 toward the proton
  (y_lab = y_cms - 0.465), Pb toward +eta_lab;
- v2{SP}: the particle of interest (h, pi, K or p, one pT interval) in one
  subevent, -0.8 < eta < -0.4 or 0.4 < eta < 0.8, against the unidentified
  particles of the other over the full pT range, and vice versa
  (|Delta eta| > 0.8);
- v2{2PC}: both particles from the same pT interval, pair gap 0.8 <
  |Delta eta| < 1.6, V_2Delta^{h-i}/sqrt(V_2Delta^{h-h});
- V2{2PC,sub}: after subtracting the 60-100% per-trigger yield;
- errors: stat and sys.

Records (public): v2{SP}(pT) of h, pi, K, p in the four classes (Tables
5-20, second column), method scalar_product with the two subevents
(particles AB), lab frame.  Staged (--staged OUT): v2{2PC} (Tables 5-20,
first column: a pair gap within one pT interval, not two subevents) and
V2{2PC,sub} (Tables 1-4: the 60-100% subtraction).  The old records filed
v2{2PC} as a no-gap v2{2} with species 'charged' or 'p' and V0M classes.

Run from the repository root:
    python scripts/curate_alice_pid_v2_ppb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1242302/HEPData-ins1242302-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Abelev et al. (ALICE), PLB 726 (2013) 164, arXiv:1307.3237'
SYSTEM = {'energy_gev': 5020.0, 'frame': 'lab', 'rapidity_shift': -0.465, 'forward_beam': 'Pb'}
V0A = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
       'eta': [{'min': 2.8, 'max': 5.1}], 'notes': 'VZERO-A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side).'}
REF_PT = {'min': 0.3, 'max': 4.0}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': -0.8, 'max': -0.4}, 'pT': dict(REF_PT)}},
             'B': {'kinematic_range': {'eta': {'min': 0.4, 'max': 0.8}, 'pT': dict(REF_PT)}}}
SPECIES = {'CHARGED': 'charged', '( PI+ + PI- )': 'pi+ + pi-', '( K+ + K- )': 'K+ + K-', '( P + PBAR )': 'p + pbar'}
OLD = 'data/flow_vn/alice_flow_vn_long-range_angular_correlations_of_k_PbPb_5020gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 21):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            re_ = q['RE']
            sp = SPECIES[re_.split('-->')[1].rsplit(' X', 1)[0].strip()]
            h = dv['header']['name']
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            contiguous = all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:]))
            pt = ({'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [p[0] for p in pairs] + [pairs[-1][1]]}
                  if contiguous else {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs})
            cls_q = q['Muliplicity Class']
            m = re.fullmatch(r'([\d.]+)-([\d.]+)', cls_q)
            cls = f'{float(m.group(1)):g}-{float(m.group(2)):g}%' if m else None
            d = {'observable_type': 'flow_vn', 'inspire_id': '1242302', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"{' '.join(t.get('description', '').split())[:100]} | {h}",
                 'collision_system': {**SYSTEM, 'reaction': re_},
                 'centrality': {'classes': [cls] if cls else [cls_q], **V0A},
                 'particle_id': {'species': sp},
                 'kinematic_range': {'eta': {'min': -0.8, 'max': 0.8}, 'pT': pt},
                 'harmonic': 2, 'value_type': h, 'dimensionality': 'differential_pT'}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if h == 'V(SP)':
                d.update({'method': 'scalar_product', 'particles': 'AB', 'eta_gap': 0.8,
                          'subevents': {k: {'kinematic_range': {kk: dict(vv) for kk, vv in v['kinematic_range'].items()}}
                                        for k, v in SUBEVENTS.items()},
                          'notes': {'value': f'{CITE}. v2{{SP}}(pT) of {sp}, {cls} (VZERO-A), the particle in one '
                                             'subevent (-0.8 < eta_lab < -0.4 or 0.4 < eta_lab < 0.8) against the '
                                             'unidentified particles (0.3 < pT < 4 GeV/c) of the other; lab frame. '
                                             'Errors: stat and sys.'}})
                public.append(d)
            else:
                what = ('v2{2PC}: both particles in one pT interval, pair gap 0.8 < |Delta eta| < 1.6'
                        if h == 'V(2PC)' else 'V2{2PC,sub}: the 60-100% per-trigger yield subtracted')
                d.update({'method': 'cumulant', 'cumulant_order': 2, 'eta_gap': 0.8,
                          'notes': {'value': f'{CITE}. {what}; {sp}, {cls or cls_q}. Errors: stat and sys.'},
                          'staging_reason': ('a pair gap within one pT interval (V_2Delta^{h-i}/sqrt(V_2Delta^{h-h})), '
                                             'not two subevents' if h == 'V(2PC)' else
                                             'the 60-100% class subtracted (a difference of two classes)')})
                staged.append(d)
    if os.path.exists(OLD):
        os.remove(OLD)
    out = 'data/flow_vn/alice_flow_vn_pid_v2_SP_pPb_5020gev_ins1242302.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['particle_id']['species'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
