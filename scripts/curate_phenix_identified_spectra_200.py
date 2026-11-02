#!/usr/bin/env python3
"""Curate PHENIX identified charged-hadron spectra in Au+Au at 200 GeV, INSPIRE 624474.

S. S. Adler et al. (PHENIX), "Identified charged particle spectra and
yields in Au+Au collisions at sqrt(sNN) = 200 GeV", PRC 69, 034909 (2004),
arXiv:nucl-ex/0307022.

From the paper (Sec. II-III):
- pi+-, K+-, p, pbar in the east-arm TOF (|eta| < 0.35), yields per unit
  rapidity at mid-rapidity;
- centrality from the BBC charge (3.0 < |eta| < 3.9) versus the ZDC
  energy: 0-5, 5-10, 10-15, 15-20, 20-30 ... 70-80, 80-92%, and 60-92%;
  minimum bias 0-92%;
- p and pbar corrected for the weak-decay feed-down from Lambda, Xi and
  Sigma0 (HIJING), not from Sigma+-;
- spectra errors: statistical (systematic in the paper); dN/dy and <pT>:
  systematic.

Records (public): invariant yields of pi+, pi-, K+, K-, p, pbar in the 12
classes (Fig. 1), dN/dy and <pT> vs centrality of each species (Figs. 3,
4; <pT> in MeV/c), |y| < 0.35 used (the arm acceptance).  Staged
(--staged OUT): minimum-bias spectra (no class of the model events).  Not
curated: the fit inverse slopes and parameters (Fig. 2), particle ratios
(Figs. 5, 6), R_CP (Fig. 7) and the N_part table.  The 5 old records read
one column (pi+) for every species and the N_part table as <pT>.

Run from the repository root:
    python scripts/curate_phenix_identified_spectra_200.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins624474/HEPData-ins624474-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. S. Adler et al. (PHENIX), PRC 69, 034909 (2004), arXiv:nucl-ex/0307022'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge (with the ZDC energy in the paper).'}
Y = {'min': -0.35, 'max': 0.35}
SPECTRA = {'fig1-1.yaml': 'pi+', 'fig1-2.yaml': 'pi-', 'fig1-3.yaml': 'K+', 'fig1-4.yaml': 'K-', 'fig1-5.yaml': 'p',
           'fig1-6-1.yaml': 'pbar', 'fig1-6-2.yaml': 'pbar'}
NAMES = {'pi+': 'pi+', 'pi-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'proton': 'p', 'antiproton': 'pbar'}
FEED = '; corrected for weak-decay feed-down from Lambda, Xi, Sigma0 (not Sigma+-)'
OLD = ['data/particle_yields/phenix_particle_yields_identified_charged_particle_spectra_and_AuAu_200gev.yaml',
       'data/mean_pT/phenix_mean_pT_identified_charged_particle_spectra_and_AuAu_200gev.yaml']


def cls_of(v):
    return f"{float(v['low']):g}-{float(v['high']):g}%"      # the dN/dy, <pT> tables give 80-92.2%


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out, staged = collections.defaultdict(list), []
    for hf, sp in SPECTRA.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            cls = str(dv['qualifiers'][0]['value'])
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', 'none')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            mb = cls == 'minimum bias'
            d = {'observable_type': 'particle_spectra', 'inspire_id': '624474', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'invariant yield of {sp}, {cls} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': ['MB' if mb else cls], **BBC}, 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}, Fig. 1. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls}, mid-rapidity (|eta| < '
                                    '0.35; |y| < 0.35 used)' + (FEED if sp in ('p', 'pbar') else '')
                                    + '. Errors: statistical (systematic in the paper).'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if mb:
                d['staging_reason'] = 'minimum bias: no centrality class of the model events'
                staged.append(d)
            else:
                out['particle_spectra'].append(d)
    for hf, otype in (('fig4-1.yaml', 'particle_yields'), ('fig3-1.yaml', 'mean_pT')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        classes = [cls_of(v) for v in iv['values']]
        for i, dv in enumerate(t['dependent_variables']):
            sp = NAMES[str(dv['qualifiers'][0]['value'])]
            what, vt = ('dN/dy', 'dN_dy') if otype == 'particle_yields' else ('<pT>', 'mean_pT')
            out[otype].append({
                'observable_type': otype, 'inspire_id': '624474', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                'hepdata_table_label': f'{what} of {sp} vs centrality ({hf})',
                'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': sp},
                'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                'observable': vt, 'value_type': vt,
                'notes': {'value': f'{CITE}, Fig. {4 if otype == "particle_yields" else 3}. {what} of {sp} vs centrality, '
                                   'mid-rapidity, fits extrapolated to all pT' + (FEED if sp in ('p', 'pbar') else '')
                                   + ('; tabulated in MeV/c' if otype == 'mean_pT' else '')
                                   + '. Errors: systematic.'}})
    for f in OLD:
        if os.path.exists(f):
            assert all(str(x['inspire_id']) == '624474' for x in yaml.safe_load_all(open(f)) if x), f
            os.remove(f)
    for otype, docs in out.items():
        path = f'data/{otype}/phenix_{otype}_identified_AuAu_200gev_ins624474.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
