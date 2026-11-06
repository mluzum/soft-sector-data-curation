#!/usr/bin/env python3
"""Curate STAR inclusive antiproton spectra in Au+Au at 130 GeV, INSPIRE 564369.

C. Adler et al. (STAR), "Measurement of inclusive antiprotons from Au+Au
collisions at sqrt(sNN) = 130 GeV", PRL 87 (2001) 262302,
arXiv:nucl-ex/0110009.

From the paper:
- centrality from the charged multiplicity in |eta| < 0.75 (TPC), eight
  classes 0-6, 6-11, 11-18, 18-26, 26-34, 34-45, 45-58, 58-85%;
- antiprotons at |y| < 0.1 from TPC dE/dx, inclusive: they include the
  products of weak decays (efficiency identical for primary and secondary
  antiprotons within the 3 cm DCA cut);
- Fig. 2: invariant yields per event, in the pT bins HEPData gives;
  errors 'Uncorr' (systematic 10%, not tabulated).

Records (public): the spectra (8).  Staged (--staged OUT): the fit
parameters and the fiducial dN/dy and net-proton dN/dy vs dN_h-/deta
(Fig. 3: rows by dN_h-/deta, the spectra hold the fiducial yield).

Run from the repository root:  python scripts/curate_star_pbar_auau130.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins564369/HEPData-ins564369-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PRL 87 (2001) 262302, arXiv:nucl-ex/0110009'
OUT = 'data/particle_spectra/star_particle_spectra_pbar_AuAu_130gev_ins564369.yaml'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.75, 'max': 0.75}], 'notes': 'charged particle multiplicity in |eta| < 0.75 (TPC).'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    hf = 'antiPFig1.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    bins = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        cls = re.fullmatch(r'\$(\d+)-(\d+)\\%\$ centrality', dv['qualifiers'][0]['value']).groups()
        cls = f'{cls[0]}-{cls[1]}%'
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('Uncorr',)}, (hf, i)
        rb = [bins[r] for r in rows]
        d = {'observable_type': 'particle_spectra', 'inspire_id': '564369', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': f'Fig. 2: pbar invariant yield, {cls} ({hf} column {i})',
             'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> PBAR X'},
             'centrality': {'classes': [cls], **NCH}, 'particle_id': {'species': 'pbar'},
             'kinematic_range': {'y': {'min': -0.1, 'max': 0.1},
                                 'pT': {'min': rb[0][0], 'max': rb[-1][1], 'bin_ranges': rb}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield',
             'hepdata_corrections': {'pT': "HEPData names the pT axis after the yield and puts that name in the "
                                           "column's qualifier; the bins are pT"},
             'notes': {'value': f'{CITE}, Fig. 2. 1/(2 pi pT) d2N/(dpT dy) of inclusive antiprotons (weak-decay '
                                f'products included), |y| < 0.1, Au+Au 130 GeV {cls}. Errors: uncorrelated (10% '
                                'systematic not tabulated).'}}
        if rows != list(range(len(bins))):
            d['table_rows'] = rows
        public.append(d)
    for hf in ('antiPFig3.yaml', 'antiPFig3a.yaml', 'antiPFig3c.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '564369', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{name} ({hf} column {i})',
                           'value_type': name, 'notes': {'value': f'{CITE}. Fig. 3.'},
                           'staging_reason': 'a fit parameter' if hf == 'antiPFig3.yaml' else
                           'yields vs dN_h-/deta (no class labels; the spectra hold the fiducial yield)'})
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
