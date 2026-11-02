#!/usr/bin/env python3
"""Curate ALICE phi and K*0 production in p-Pb at 5.02 TeV, INSPIRE 1418181.

J. Adam et al. (ALICE), "Production of K*(892)0 and phi(1020) in p-Pb
collisions at sqrt(sNN) = 5.02 TeV", EPJC 76 (2016) 245, arXiv:1601.07868.

From the paper and tables:
- V0A multiplicity classes (2.8 < eta_lab < 5.1, Pb-going side); 2013
  beams: y_lab = y_cms - 0.465, Pb toward +eta;
- phi and (K*0 + K*0bar)/2 at -0.5 < y_cms < 0: d2N/(dpT dy) spectra, dN/dy
  and <pT> (fits extrapolated to all pT); NSD (non-single-diffractive)
  results besides the classes;
- <dNch/deta_lab> at |eta_lab| < 0.5 per class (Table 1);
- errors: stat and sys (sys,uncorr and sys,corr disjoint for dN/dy).

Records (public): phi spectra per class (Tables 9-14, value_type
dN_dpT_dy), phi dN/dy and <pT> vs class (Tables 18, 19, the NSD row left
out), <dNch/deta_lab> (Table 1: the 0-5 ... 80-100% rows and, separately,
0-20%).  Frames: nn_cms for the y_cms records (the V0A window then at
3.265 < eta < 5.565), lab for Table 1.  Staged (--staged OUT): K*0 (a
resonance not provided by the pipeline: Tables 2-7, 16, 17) and the NSD
phi spectra (Tables 8, 15: no class).  Not curated: <pT> vs mass across
systems (Tables 20, 21) and the yield ratios (Tables 22-30).  The old
records called phi and K*0 'p' and gave no frame.

Run from the repository root:
    python scripts/curate_alice_phi_kstar_ppb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1418181/HEPData-ins1418181-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (ALICE), EPJC 76 (2016) 245, arXiv:1601.07868'
SHIFT = -0.465
V0A_CMS = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
           'eta': [{'min': round(2.8 - SHIFT, 3), 'max': round(5.1 - SHIFT, 3)}],
           'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side), here in the nn_cms frame.'}
V0A_LAB = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
           'eta': [{'min': 2.8, 'max': 5.1}], 'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side).'}
Y = {'min': -0.5, 'max': 0.0}
OLD = ['data/particle_yields/alice_particle_yields_production_of_k_892_and_PbPb_5020gev.yaml',
       'data/mean_pT/alice_mean_pT_production_of_k_892_and_PbPb_multigev.yaml']


def system(frame, reaction):
    return {'energy_gev': 5020.0, 'reaction': reaction, 'frame': frame, 'rapidity_shift': SHIFT, 'forward_beam': 'Pb'}


def cls_of(v):
    return f"{float(v['low']):g}-{float(v['high']):g}%" if 'low' in v else str(v['value'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out = collections.defaultdict(list)
    staged = []

    def load(n):
        return yaml.safe_load(open(os.path.join(RAW, f'Table{n}.yaml')))

    # <dNch/deta_lab>, |eta_lab| < 0.5: the non-overlapping classes and 0-20% apart
    t = load(1)
    iv = t['independent_variables'][0]
    classes = [cls_of(v) for v in iv['values']]
    for rows in ([j for j, c in enumerate(classes) if c != '0-20%'], [classes.index('0-20%')]):
        out['particle_yields'].append({
            'observable_type': 'particle_yields', 'inspire_id': '1418181', 'version': VERSION,
            'hepdata_file': 'Table1.yaml', 'dv_index': 0,
            'hepdata_table_label': '<dNch/deta_lab> vs V0A class (Table1.yaml)',
            'collision_system': system('lab', 'P PB --> CHARGED X'),
            'centrality': {'classes': [classes[j] for j in rows], **V0A_LAB}, 'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'dNch_deta', 'value_type': 'dNch_deta', 'table_rows': rows,
            'notes': {'value': f'{CITE}. <dNch/deta_lab> at |eta_lab| < 0.5 per V0A class; lab frame.'}})
    # spectra: K*0 (Tables 2-7, staged), phi NSD (8, 15, staged), phi per class (9-14)
    for n in range(2, 16):
        t = load(n)
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        kstar = n <= 7
        sp = 'K*0 + K*0bar' if kstar else 'phi'
        c = q.get('class', '')
        cls = cls_of({'low': c.split('-')[0], 'high': c.split('-')[1]}) if '-' in c else None   # 'NSD'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'particle_spectra', 'inspire_id': '1418181', 'version': VERSION,
             'hepdata_file': f'Table{n}.yaml', 'dv_index': 0,
             'hepdata_table_label': f"d2N/(dpT dy) of {sp}, {cls or 'NSD'} (Table{n}.yaml)",
             'collision_system': system('nn_cms', q['RE']),
             'centrality': {'classes': [cls] if cls else ['NSD'], **V0A_CMS},
             'particle_id': {'species': sp, **({'charge_average': True} if kstar else {})},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy',
             'notes': {'value': f"{CITE}. 1/N_ev d2N/(dpT dy) of {'(K*0 + K*0bar)/2' if kstar else 'phi'}, "
                                f"{cls or 'NSD'} (V0A), -0.5 < y_cms < 0. Errors: stat and sys."}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if kstar:
            d['staging_reason'] = 'K*(892)0, a resonance not provided by the pipeline'
            staged.append(d)
        elif not cls:
            d['staging_reason'] = 'NSD (non-single-diffractive) events: no multiplicity class'
            staged.append(d)
        else:
            out['particle_spectra'].append(d)
    # dN/dy and <pT> vs class: K*0 (16, 17, staged), phi (18, 19; the NSD row left out)
    for n, otype in ((16, 'particle_yields'), (17, 'mean_pT'), (18, 'particle_yields'), (19, 'mean_pT')):
        t = load(n)
        iv = t['independent_variables'][0]
        dv = t['dependent_variables'][0]
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        kstar = n <= 17
        sp = 'K*0 + K*0bar' if kstar else 'phi'
        rows = [j for j, v in enumerate(iv['values']) if 'low' in v and str(dv['values'][j]['value']).strip() not in ('', '-')]
        classes = [cls_of(iv['values'][j]) for j in rows]
        what = 'dN/dy' if otype == 'particle_yields' else '<pT>'
        d = {'observable_type': otype, 'inspire_id': '1418181', 'version': VERSION, 'hepdata_file': f'Table{n}.yaml',
             'dv_index': 0, 'hepdata_table_label': f'{what} of {sp} vs V0A class (Table{n}.yaml)',
             'collision_system': system('nn_cms', q['RE']),
             'centrality': {'classes': classes, **V0A_CMS},
             'particle_id': {'species': sp, **({'charge_average': True} if kstar else {})},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
             'value_type': 'dN_dy' if otype == 'particle_yields' else 'mean_pT', 'table_rows': rows,
             'notes': {'value': f'{CITE}. {what} of {"(K*0 + K*0bar)/2" if kstar else "phi"} vs V0A class, -0.5 < '
                                'y_cms < 0, all pT (extrapolated); the NSD row left out.'}}
        d['observable'] = 'dN_dy' if otype == 'particle_yields' else 'mean_pT'
        if kstar:
            d['staging_reason'] = 'K*(892)0, a resonance not provided by the pipeline'
            staged.append(d)
        else:
            out[otype].append(d)
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '1418181' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    for otype, docs in out.items():
        path = f'data/{otype}/alice_{otype}_phi_pPb_5020gev_ins1418181.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
