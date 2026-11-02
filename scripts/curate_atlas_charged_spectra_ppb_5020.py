#!/usr/bin/env python3
"""Curate ATLAS charged-hadron spectra in p+Pb at 5.02 TeV, INSPIRE 1463284.

G. Aad et al. (ATLAS), "Transverse momentum, rapidity, and centrality
dependence of inclusive charged-particle production in sqrt(sNN) = 5.02
TeV p+Pb collisions measured by the ATLAS experiment", PLB 763 (2016) 313,
arXiv:1605.06436.

From the paper (Secs. 2-5):
- 2013 run: 4 TeV protons, 1.57 TeV/nucleon Pb; rapidity positive in the
  proton direction; y* = y_lab - 0.465;
- centrality from the FCal transverse energy on the Pb-going side, -4.9 <
  eta < -3.1 (lab); classes 0-1, 1-5, 5-10, 10-20, 20-30, 30-40, 40-60,
  60-90%; 0-90% for the integrated spectra;
- y* from the track momentum with the pion mass, corrected to the true
  rapidity with simulation (Eq. 7);
- errors: stat and sys.

Records (public): invariant yields 1/(2 pi pT) d2N/(dpT deta) in six eta_lab
intervals (Table 1; frame lab, proton toward +eta, rapidity_shift +0.465)
and 1/(2 pi pT) d2N/(dpT dy*) in six y* intervals (Table 2; frame nn_cms,
proton toward +y, the FCal window in that frame), 0-90%; points with value
and errors 0 left out.  Staged (--staged OUT): dN/dy* vs y* per class
(Table 3: rapidity-differential yields of unidentified hadrons are not
computed).  Not curated: R_pPb and R_CP (Tables 4-59: nuclear modification,
out of scope).  The 2 old records had no frame and called a spectrum dN/dy.

Run from the repository root:
    python scripts/curate_atlas_charged_spectra_ppb_5020.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1463284/HEPData-ins1463284-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'G. Aad et al. (ATLAS), PLB 763 (2016) 313, arXiv:1605.06436'
SHIFT = 0.465
FCAL_LAB = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'}, 'eta': [{'min': -4.9, 'max': -3.1}],
            'notes': 'FCal transverse energy on the Pb-going side (lab, proton toward +eta).'}
FCAL_CMS = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
            'eta': [{'min': round(-4.9 - SHIFT, 3), 'max': round(-3.1 - SHIFT, 3)}],
            'notes': 'FCal transverse energy, -4.9 < eta_lab < -3.1 (Pb-going side), here in the nn_cms frame.'}
OLD = ['data/particle_yields/atlas_particle_yields_transverse_momentum_rapidity_and_centrality_PbPb_200gev.yaml',
       'data/particle_spectra/atlas_particle_spectra_transverse_momentum_rapidity_and_centrality_PbPb_200gev.yaml']


def system(frame):
    return {'energy_gev': 5020.0, 'reaction': 'P PB --> CHARGED X', 'frame': frame, 'rapidity_shift': SHIFT,
            'forward_beam': 'p'}


# HEPData qualifier '-1.5-0.5' (Table 2) is the interval -1.5 < y* < -0.5 (the paper and the
# neighbouring intervals); every other qualifier reads lo-hi.
QUALIFIER_FIX = {'-1.5-0.5': (-1.5, -0.5)}


def window(text):
    text = text.strip()
    if text in QUALIFIER_FIX:
        lo, hi = QUALIFIER_FIX[text]
    else:
        m = re.fullmatch(r'(-?[\d.]+)\s*-\s*(-?[\d.]+)', text)
        lo, hi = float(m.group(1)), float(m.group(2))
    return {'min': lo, 'max': hi}


def kept(v):
    if str(v['value']).strip().lower() in ('', '-', 'none'):
        return False
    return not (float(v['value']) == 0.0 and all(float(e.get('symerror', 0) or 0) == 0.0 for e in v.get('errors') or []))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, var in (('Table1.yaml', 'eta'), ('Table2.yaml', 'y')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            w = window(q['ETARAP'] if var == 'eta' else q['YRAP'])
            rows = [j for j, v in enumerate(dv['values']) if kept(v)]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            lab = var == 'eta'
            what = f"{w['min']:g} < eta_lab < {w['max']:g}" if lab else f"{w['min']:g} < y* < {w['max']:g}"
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1463284', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'charged-hadron invariant yield, 0-90%, {what} ({hf})',
                 'collision_system': system('lab' if lab else 'nn_cms'),
                 'centrality': {'classes': ['0-90%'], **(FCAL_LAB if lab else FCAL_CMS)},
                 'particle_id': {'species': 'charged'},
                 'kinematic_range': {var: w, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}. 1/(2 pi pT) d2N/(dpT d{"eta" if lab else "y*"}) of charged hadrons, 0-90%, '
                                    f'{what}; ' + ('lab frame, the proton toward +eta' if lab else
                                                   'nn_cms frame, the proton toward +y; y* corrected from the pion-mass '
                                                   'assumption to the true rapidity')
                                    + '. Errors: stat and sys.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            fix = q.get('YRAP', '').strip()
            if var == 'y' and fix in QUALIFIER_FIX:
                d['hepdata_corrections'] = {'y': f"HEPData qualifier YRAP '{fix}'; the interval is -1.5 < y* < -0.5 "
                                                 '(the paper and the neighbouring intervals)'}
            public.append(d)
    t = yaml.safe_load(open(os.path.join(RAW, 'Table3.yaml')))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        cls = str({qq['name']: qq.get('value') for qq in dv['qualifiers']}['Centrality'])
        pairs = [[float(v['low']), float(v['high'])] for v in iv['values']]
        staged.append({'observable_type': 'particle_yields', 'inspire_id': '1463284', 'version': VERSION,
                       'hepdata_file': 'Table3.yaml', 'dv_index': i,
                       'hepdata_table_label': f'dN/dy* of charged hadrons vs y*, {cls} (Table3.yaml)',
                       'collision_system': system('nn_cms'), 'centrality': {'classes': [cls], **FCAL_CMS},
                       'particle_id': {'species': 'charged'},
                       'kinematic_range': {'y': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs},
                                           'pT': {'min': 0.1, 'max': 100.0}},
                       'observable': 'dN_dy', 'value_type': 'dN_dy',
                       'notes': {'value': f'{CITE}. dN/dy* of charged hadrons with pT > 0.1 GeV/c vs y*, {cls}.'},
                       'staging_reason': 'rapidity-differential yield of unidentified hadrons (true rapidity of a '
                                         'mixture): not computed'})
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '1463284' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    out = 'data/particle_spectra/atlas_particle_spectra_charged_pPb_5020gev_ins1463284.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
