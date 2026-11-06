#!/usr/bin/env python3
"""Curate PHENIX phi -> l+ l- spectra in d+Au at 200 GeV, INSPIRE 1379995.

A. Adare et al. (PHENIX), "phi meson production in d+Au collisions at
sqrt(sNN) = 200 GeV", PRC 92 (2015) 044909, arXiv:1506.08181.

From the paper:
- phi -> mu+ mu- in the muon arms, d-going 1.2 < y < 2.2 and Au-going
  -2.2 < y < -1.2, 1 < pT < 7 GeV/c; phi -> e+ e- in the central arms,
  |y| < 0.35, pT < 7 GeV/c;
- invariant yields B_ll / (2 pi pT) d2N/(dy dpT) (Eq. 1), B_ll the
  dilepton branching ratio (not quoted), corrected for the BBC trigger
  bias;
- classes 0-20, 20-40, 40-60, 60-88% from the BBC (Ref. [47]'s
  definition: the charge in the Au-going, south BBC, -3.9 < eta < -3.1),
  and minimum bias, which the bias correction makes 0-100% (Figs. 8, 9);
- errors: stat. and type B sys. (point-to-point correlated); type C
  (global, 6% at forward and backward y) not tabulated.

Records (public): the spectra of Fig. 7, 15 (three rapidity windows x
five classes), with value_scale the PDG branching ratio (phi -> mu+ mu-:
2.86e-4 +- 0.19e-4; phi -> e+ e-: 2.974e-4 +- 0.034e-4; not in the
tabulated errors).  d+Au in the nn frame with d toward +y (rapidity_shift
0).  Staged (--staged OUT): R_dAu (Figs. 9, 11, 12; need the pp
reference).

Run from the repository root:  python scripts/curate_phenix_phi_dau.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1379995/HEPData-ins1379995-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRC 92 (2015) 044909, arXiv:1506.08181'
OUT = 'data/particle_spectra/phenix_particle_spectra_phi_dilepton_dAu_200gev_ins1379995.yaml'
BBC = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}],
       'notes': 'charge in the Au-going (south) BBC, -3.9 < eta < -3.1 (the definition of PHENIX PRC 90 034902).'}
SYSTEM = {'energy_gev': 200.0, 'reaction': 'D AU --> PHI X', 'frame': 'nn_cms', 'rapidity_shift': 0.0,
          'forward_beam': 'd'}
WINDOWS = {'fig7a.yaml': ('-2.2 < $y$ < -1.2', -2.2, -1.2, 'mu+ mu-', 2.86e-4, 'Au-going'),
           'fig7b.yaml': ('1.2 < $y$ < 2.2', 1.2, 2.2, 'mu+ mu-', 2.86e-4, 'd-going'),
           'fig7c.yaml': ('$|y|$ < 0.35', -0.35, 0.35, 'e+ e-', 2.974e-4, 'midrapidity')}
CLASSES = {'Minimum Bias': '0-100%', '0-20': '0-20%', '20-40': '20-40%', '40-60': '40-60%', '60-88': '60-88%'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if hf not in WINDOWS:
            for i, dv in enumerate(t['dependent_variables']):
                if 'N_{coll}' in dv['header']['name']:
                    continue
                staged.append({'observable_type': 'nuclear_modification', 'inspire_id': '1379995', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc[:200]}'},
                               'staging_reason': 'R_dAu: needs the pp reference'})
            continue
        qy, lo, hi, chan, br, side = WINDOWS[hf]
        bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            assert q['Rapidity'] == qy and 'B_{ll}/2 \\pi p_T' in dv['header']['name'], (hf, q)
            cls = CLASSES[q['Centrality (%)']]
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat.', 'B sys.')}
            rb = [bins[r] for r in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1379995', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'Fig. 7: phi -> {chan} invariant yield, {side}, {cls} ({hf} column {i})',
                 'collision_system': dict(SYSTEM), 'centrality': {'classes': [cls], **BBC},
                 'particle_id': {'species': 'phi'},
                 'kinematic_range': {'y': {'min': lo, 'max': hi},
                                     'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'value_scale': br}
            if rows != list(range(len(bins))):
                d['table_rows'] = rows
            mb = ' (minimum bias, corrected for the BBC trigger bias: 0-100%)' if cls == '0-100%' else ''
            d['notes'] = {'value': f'{CITE}, Fig. 7. 1/(2 pi pT) d2N/(dy dpT) of phi -> {chan}, {lo:g} < y < {hi:g} '
                                   f'({side}), d+Au 200 GeV {cls}{mb}. Tabulated times B_ll; value_scale = {br:g}, '
                                   f'the PDG phi -> {chan} branching ratio (its uncertainty not in the tabulated '
                                   'errors). Errors: stat. and type B sys.; type C (global) not tabulated.'}
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
