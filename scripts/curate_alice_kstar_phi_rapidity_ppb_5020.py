#!/usr/bin/env python3
"""Curate ALICE K*0 and phi vs multiplicity and rapidity in p-Pb at 5.02 TeV, INSPIRE 2070441.

S. Acharya et al. (ALICE), "Multiplicity and rapidity dependence of
K*(892)0 and phi(1020) production in p-Pb collisions at sqrt(sNN) = 5.02
TeV", EPJC 83 (2023) 540, arXiv:2204.10263.

From the paper (Sec. 2):
- 2016 data; five rapidity slices -1.2 < y < -0.9 ... 0 < y < 0.3 in the
  nucleon-nucleon cms, with y < 0 the Pb-going and y > 0 the p-going
  direction (proton toward +y; the cms moves by 0.465 toward the proton);
- V0A multiplicity classes 0-10, 10-40, 40-100% and 0-100% (all minimum-
  bias events); V0A at 2.8 < eta_lab < 5.1, on the Pb-going side (ALICE A
  side), i.e. -5.565 < eta < -3.265 in this frame;
- phi -> K+K-, K*0 -> K pi (particle and antiparticle averaged);
  d2N/(dpT dy) per event, dN/dy and <pT> from fits extrapolated to all pT;
- errors: stat and sys.

Records (public): phi d2N/(dpT dy) per rapidity slice and class (Tables
6-10, value_type dN_dpT_dy), phi dN/dy and <pT> per class, one record per
rapidity slice (Tables 12, 14, table_rows).  Frame nn_cms with the proton
toward +eta (forward_beam p; rapidity_shift +0.465 in that orientation).
Staged (--staged OUT): K*0 (a resonance not provided by the pipeline:
Tables 1-5, 11, 13).  Not curated: ratios to y = 0 (Tables 15-18), the
rapidity asymmetry Y_asym (19, 20) and Q_CP (21-24), ratios of curated
yields (Q_CP with <N_coll>).  The 4 old records were filed as Pb-Pb, with
|eta| < 0.8 or -1.2 < eta < 0.3 and no frame.

Run from the repository root:
    python scripts/curate_alice_kstar_phi_rapidity_ppb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins2070441/HEPData-ins2070441-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), EPJC 83 (2023) 540, arXiv:2204.10263'
SHIFT = 0.465
V0A = {'estimator': {'name': 'V0A', 'observable': 'amplitude', 'detector': 'V0A'},
       'eta': [{'min': round(-5.1 - SHIFT, 3), 'max': round(-2.8 - SHIFT, 3)}],
       'notes': 'V0A amplitude, 2.8 < eta_lab < 5.1 (Pb-going side), here in the nn_cms frame with the proton '
                'toward +eta.'}
OLD = ['data/particle_yields/alice_particle_yields_multiplicity_and_rapidity_dependence_of_PbPb_5020gev.yaml',
       'data/mean_pT/alice_mean_pT_multiplicity_and_rapidity_dependence_of_PbPb_5020gev.yaml']
FRAME_NOTE = 'nn_cms frame, proton toward +y (the paper: y < 0 is the Pb-going direction)'


def system(sp):
    return {'energy_gev': 5020.0, 'reaction': f"P PB --> {'K*0 X' if sp != 'phi' else 'PHI X'}", 'frame': 'nn_cms',
            'rapidity_shift': SHIFT, 'forward_beam': 'p'}


def load(n):
    return yaml.safe_load(open(os.path.join(RAW, f'Table{n}.yaml')))


def rng(text):
    lo, hi = (float(x) for x in text.replace(' - ', ' ').split())
    return {'min': lo, 'max': hi}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out, staged = collections.defaultdict(list), []

    def keep(d, kstar):
        if kstar:
            d['staging_reason'] = 'K*(892)0, a resonance not provided by the pipeline'
            staged.append(d)
        else:
            out[d['observable_type']].append(d)

    # spectra: K*0 (1-5), phi (6-10), one table per rapidity slice, one DV per class
    for n in range(1, 11):
        kstar = n <= 5
        sp, pid = ('K*0', {'species': 'K*0 + K*0bar', 'charge_average': True}) if kstar else ('phi', {'species': 'phi'})
        t = load(n)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            cls, y = q['class'] + '%', rng(q['YRAP(C=CM)'])
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '2070441', 'version': VERSION,
                 'hepdata_file': f'Table{n}.yaml', 'dv_index': i,
                 'hepdata_table_label': f"d2N/(dpT dy) of {sp}, {cls}, {y['min']:g} < y < {y['max']:g} (Table{n}.yaml)",
                 'collision_system': system(sp), 'centrality': {'classes': [cls], **V0A}, 'particle_id': dict(pid),
                 'kinematic_range': {'y': y, 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy', 'dimensionality': 'differential_pT',
                 'notes': {'value': f"{CITE}. 1/N_ev d2N/(dpT dy) of {'(K*0 + K*0bar)/2' if kstar else 'phi'}, {cls} "
                                    f"(V0A), {y['min']:g} < y < {y['max']:g}; {FRAME_NOTE}. Errors: stat and sys."}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            keep(d, kstar)
    # dN/dy (11 K*0, 12 phi) and <pT> (13 K*0, 14 phi) vs rapidity: one record per class and slice
    for n, otype in ((11, 'particle_yields'), (12, 'particle_yields'), (13, 'mean_pT'), (14, 'mean_pT')):
        kstar = n in (11, 13)
        sp, pid = ('K*0', {'species': 'K*0 + K*0bar', 'charge_average': True}) if kstar else ('phi', {'species': 'phi'})
        what = 'dN/dy' if otype == 'particle_yields' else '<pT>'
        vt = 'dN_dy' if otype == 'particle_yields' else 'mean_pT'
        t = load(n)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            cls = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}['class'] + '%'
            for j, v in enumerate(iv['values']):
                if str(dv['values'][j]['value']).strip() in ('', '-'):
                    continue
                y = {'min': float(v['low']), 'max': float(v['high'])}
                keep({'observable_type': otype, 'inspire_id': '2070441', 'version': VERSION,
                      'hepdata_file': f'Table{n}.yaml', 'dv_index': i,
                      'hepdata_table_label': f"{what} of {sp}, {cls}, {y['min']:g} < y < {y['max']:g} (Table{n}.yaml)",
                      'collision_system': system(sp), 'centrality': {'classes': [cls], **V0A},
                      'particle_id': dict(pid), 'kinematic_range': {'y': y, 'pT': {'min': 0.0, 'max': 100.0}},
                      'observable': vt, 'value_type': vt, 'table_rows': [j],
                      'notes': {'value': f"{CITE}. {what} of {'(K*0 + K*0bar)/2' if kstar else 'phi'}, {cls} (V0A), "
                                         f"{y['min']:g} < y < {y['max']:g}, all pT (fits extrapolated); {FRAME_NOTE}. "
                                         'Errors: stat and sys.'}}, kstar)
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '2070441' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    for otype, docs in out.items():
        path = f'data/{otype}/alice_{otype}_phi_rapidity_pPb_5020gev_ins2070441.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
