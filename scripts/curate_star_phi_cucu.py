#!/usr/bin/env python3
"""Curate STAR phi spectra in Cu+Cu at 62.4 and 200 GeV, INSPIRE 800796.

B. I. Abelev et al. (STAR), "Energy and system size dependence of phi meson
production in Cu+Cu and Au+Au collisions", PLB 673 (2009) 183,
arXiv:0810.4979.

From the paper:
- phi -> K+ K- at |y| < 0.5, invariant yields d2N/(2 pi pT dy dpT) per
  event in Cu+Cu at 200 and 62.4 GeV, classes 0-10, ..., 50-60%; the Au+Au
  data are STAR's earlier (PRC 79 064903);
- centrality from the uncorrected charged multiplicity in |eta| < 0.5
  (TPC);
- errors: stat and sys.

Records (public): the Cu+Cu spectra (Table 1a, 200 GeV; Table 1b, 62.4
GeV), 6 classes each.  HEPData gives bin centres: contiguous bins from
0.4 GeV/c, each point its bin's centre (0.4, 0.5, ..., 0.8, 1.0, 1.2,
1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0).  Staged (--staged OUT): R_AA
scaled by N_part or N_bin (Tables 2a-d: need the pp reference) and the
yields per N_part relative to pp (Tables 3a, 3b).

Run from the repository root:  python scripts/curate_star_phi_cucu.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins800796/HEPData-ins800796-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PLB 673 (2009) 183, arXiv:0810.4979'
OUT = 'data/particle_spectra/star_particle_spectra_phi_CuCu_62.4_200gev_ins800796.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'uncorrected charged multiplicity in |eta| < 0.5 (TPC).'}
ENERGY = {'Table1a.yaml': 200.0, 'Table1b.yaml': 62.4}


def centred_bins(pts):
    """Contiguous bins, each point its bin's centre, from half the first spacing below the first point."""
    edges = [round(pts[0] - (pts[1] - pts[0]) / 2, 6)]
    for c in pts:
        edges.append(round(2 * c - edges[-1], 6))
        assert edges[-1] > edges[-2] and edges[-2] < c < edges[-1], (pts, edges)
    return [[edges[i], edges[i + 1]] for i in range(len(pts))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if hf not in ENERGY:
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'nuclear_modification' if hf.startswith('Table2') else
                               'particle_yields', 'inspire_id': '800796', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc[:200]}'},
                               'staging_reason': ('R_AA (N_part or N_bin scaled): needs the pp reference'
                                                  if hf.startswith('Table2') else
                                                  'yields per N_part relative to pp')})
            continue
        energy = ENERGY[hf]
        assert f'={energy:g}$ GeV' in str(t['dependent_variables'][0]['qualifiers']), hf
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        bins = centred_bins(pts)
        for i, dv in enumerate(t['dependent_variables']):
            cls = re.fullmatch(r'centrality \$(\d+-\d+)\\%\$', dv['header']['name']).group(1) + '%'
            assert all(str(v['value']).strip() not in ('-', '', '--') for v in dv['values']), (hf, i)
            assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}, hf
            public.append({
                'observable_type': 'particle_spectra', 'inspire_id': '800796', 'version': VERSION,
                'hepdata_file': hf, 'dv_index': i,
                'hepdata_table_label': f'phi invariant yield, Cu+Cu {energy:g} GeV, {cls} ({hf} column {i})',
                'collision_system': {'energy_gev': energy, 'reaction': 'CU CU --> PHI X'},
                'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': 'phi'},
                'kinematic_range': {'y': {'min': -0.5, 'max': 0.5},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                'notes': {'value': f'{CITE}. d2N/(2 pi pT dy dpT) of phi (K+ K-), |y| < 0.5, Cu+Cu {energy:g} GeV '
                                   f'{cls}; bins contiguous from 0.4 GeV/c with the tabulated points at their centres. '
                                   'Errors: stat and sys.'}})
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
