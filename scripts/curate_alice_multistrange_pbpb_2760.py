#!/usr/bin/env python3
"""Curate ALICE Xi and Omega production in Pb-Pb at 2.76 TeV, INSPIRE 1243865.

B. B. Abelev et al. (ALICE), "Multi-strange baryon production at
mid-rapidity in Pb-Pb collisions at sqrt(sNN) = 2.76 TeV", PLB 728 (2014)
216, Erratum PLB 734 (2014) 409, arXiv:1307.5543.

From the paper:
- Xi-, Xibar+, Omega-, Omegabar+ from cascade decays, |y| < 0.5;
- centrality from the summed VZERO amplitudes: 0-10, 10-20, 20-40,
  40-60, 60-80%;
- spectra (1/N_ev) d2N/(dpT dy); yields integrated over all pT with fits
  extrapolated to the unmeasured region;
- errors: stat and sys.

Records (public): d2N/(dpT dy) of Xi-, Xibar+, Omega-, Omegabar+ in the five
classes (Tables 1-10; value_type dN_dpT_dy); dN/dy of each species and of
the particle + antiparticle sums vs centrality (Table 11).  Staged
(--staged OUT): (Xi + Xibar)/(pi + pi) and (Omega + Omegabar)/(pi + pi) vs
centrality (Table 14: particle ratios are not computed).  Not curated: the
enhancements relative to pp per participant (Table 12, N_part normalized)
and the pp ratios at 7 TeV (Table 13).  The old record read Table 11 as
dN/dy of charged particles in |eta| < 0.8, 0.15 < pT < 50 GeV/c.

Run from the repository root:
    python scripts/curate_alice_multistrange_pbpb_2760.py [--staged OUT.yaml]
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1243865/HEPData-ins1243865-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = ('B. B. Abelev et al. (ALICE), PLB 728 (2014) 216, Erratum PLB 734 (2014) 409, arXiv:1307.5543')
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
Y = {'min': -0.5, 'max': 0.5}
SPECIES = {'XI-': 'Xi-', 'XIBAR+': 'Xibar+', 'OMEGA-': 'Omega-', 'OMEGABAR+': 'Omegabar+',
           '(XI- + XIBAR+)': 'Xi- + Xibar+', '(OMEGA- + OMEGABAR+)': 'Omega- + Omegabar+'}
OLD = 'data/particle_yields/alice_particle_yields_multi-strange_baryon_production_PbPb_2760gev.yaml'
OUT_SPEC = 'data/particle_spectra/alice_particle_spectra_Xi_Omega_PbPb_2760gev_ins1243865.yaml'
OUT_YIELD = 'data/particle_yields/alice_particle_yields_Xi_Omega_PbPb_2760gev_ins1243865.yaml'
BLANK = ('', '-', '--', 'none')


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def species(reaction):
    return SPECIES[re.fullmatch(r'PB PB --> (.+) X', reaction.strip()).group(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, staged = [], [], []
    for k in range(1, 11):
        hf = f'Table{k}.yaml'
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
            sp = species(q['RE'])
            lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
            cls = f'{lo:g}-{hi:g}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1243865', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'{sp} d2N/(dpT dy), {cls} ({hf})',
                 'collision_system': {'energy_gev': 2760.0, 'reaction': f"PB PB --> {sp.upper()} X"},
                 'centrality': {'classes': [cls], **V0M}, 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}, Fig. 2. (1/N_ev) d2N/(dpT dy) of {sp}, {cls}, |y| < 0.5. Errors: stat '
                                    'and sys.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            spectra.append(d)
    t = load('Table11.yaml')
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        q = {qq['name']: str(qq.get('value')) for qq in dv['qualifiers']}
        sp = species(q['RE'])
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        classes = [f"{float(iv['values'][j]['low']):g}-{float(iv['values'][j]['high']):g}%" for j in rows]
        d = {'observable_type': 'particle_yields', 'inspire_id': '1243865', 'version': VERSION,
             'hepdata_file': 'Table11.yaml', 'dv_index': i, 'hepdata_table_label': f'{sp} dN/dy vs centrality (Table11.yaml)',
             'collision_system': {'energy_gev': 2760.0, 'reaction': f"PB PB --> {sp.upper()} X"},
             'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': sp},
             'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dN_dy', 'value_type': 'dN_dy',
             'notes': {'value': f'{CITE}, Table 2. dN/dy of {sp} at |y| < 0.5 vs centrality, all pT (fits '
                                'extrapolated). Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        yields.append(d)
    t = load('Table14.yaml')
    classes = ['60-80%', '40-60%', '20-40%', '10-20%', '0-10%']        # rows: increasing N_part
    for i, dv in enumerate(t['dependent_variables']):
        h = dv['header']['name']
        sp = 'Xi- + Xibar+' if 'XI-' in h else 'Omega- + Omegabar+'
        staged.append({'observable_type': 'particle_ratios', 'inspire_id': '1243865', 'version': VERSION,
                       'hepdata_file': 'Table14.yaml', 'dv_index': i, 'hepdata_table_label': f'{h} vs centrality (Table14.yaml)',
                       'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> X'},
                       'centrality': {'classes': classes, **V0M}, 'particle_id': {'species': sp},
                       'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                       'observable': f'{sp} / (pi+ + pi-)', 'value_type': h, 'numerator': sp,
                       'denominator': 'pi+ + pi-',
                       'notes': {'value': f'{CITE}, Fig. 5. {h} at |y| < 0.5 vs centrality (HEPData rows by N_part, '
                                          'peripheral first).'},
                       'staging_reason': 'particle ratio to pions: not computed'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '1243865' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    for out, docs in ((OUT_SPEC, spectra), (OUT_YIELD, yields)):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
