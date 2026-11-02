#!/usr/bin/env python3
"""Curate PHENIX high-pT pi0 yields in d+Au at 200 GeV, INSPIRE 2645172.

N. J. Abdulameer et al. (PHENIX), "Disentangling centrality bias and
final-state effects in the production of high-pT pi0 using direct gamma
in d+Au collisions at sqrt(sNN) = 200 GeV", PRL 134, 022302 (2025),
arXiv:2303.12899.

From the paper and tables:
- pi0 and direct-photon invariant yields at high pT (7.5-18 GeV/c) in the
  central arms (|eta| < 0.35) for d+Au event-activity classes 0-100, 0-5,
  5-10, 10-20, 20-40, 40-60, 60-88%; centrality from the charge in the BBC
  on the Au-going side (-3.9 < eta < -3.0); the deuteron goes north (+eta);
- errors: stat and sys.

Records (public): pi0 invariant yields per class (fig1a), |y| < 0.35, lab
frame with the deuteron toward +eta; the tabulated points are the bin
centres of 0.5 GeV/c bins to 10 GeV/c and 2 GeV/c bins above (edges
inferred from the centres).  Not curated: direct photons and gamma/pi0
(figs. 1b, 1c: electromagnetic probes, out of scope), N_coll^EXP and
R_dAu (figs. 2, 3: nuclear modification).  The old record read the
direct-photon table in 60-88%.

Run from the repository root:  python scripts/curate_phenix_pi0_dau_highpt_200.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2645172/HEPData-ins2645172-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'N. J. Abdulameer et al. (PHENIX), PRL 134, 022302 (2025), arXiv:2303.12899'
BBCS = {'estimator': {'name': 'BBC-S', 'observable': 'amplitude', 'detector': 'BBC'}, 'eta': [{'min': -3.9, 'max': -3.0}],
        'notes': 'Charge in the south BBC (the Au-going side).'}
OLD = 'data/particle_spectra/phenix_particle_spectra_disentangling_centrality_bias_and_final-state_AuAu_200gev.yaml'
OUT = 'data/particle_spectra/phenix_particle_spectra_pi0_highpt_dAu_200gev_ins2645172.yaml'
BLANK = ('', '-', '--', 'none')


def bins(x):
    """Centres 7.75 ... 9.75 (width 0.5) and 11, 13, 15, 17 (width 2)."""
    return [[round(c - (0.25 if c < 10 else 1.0), 3), round(c + (0.25 if c < 10 else 1.0), 3)] for c in x]


def main():
    t = yaml.safe_load(open(os.path.join(RAW, 'fig1a.yaml')))
    x = [float(v['value']) for v in t['independent_variables'][0]['values']]
    b = bins(x)
    assert all(abs(p[1] - q[0]) < 1e-9 for p, q in zip(b, b[1:])), b
    public = []
    for i, dv in enumerate(t['dependent_variables']):
        lab = str(dv['qualifiers'][0]['value'])
        m = re.search(r'(\d+)\\?%?-(\d+)\\?%', lab.replace('$', ''))
        cls = f'{m.group(1)}-{m.group(2)}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
        pairs = [b[j] for j in rows]
        d = {'observable_type': 'particle_spectra', 'inspire_id': '2645172', 'version': VERSION, 'hepdata_file': 'fig1a.yaml',
             'dv_index': i, 'hepdata_table_label': f'pi0 invariant yield, d+Au {cls} (fig1a.yaml)',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'D AU --> PI0 X', 'frame': 'lab',
                                  'rapidity_shift': 0.0, 'forward_beam': 'd'},
             'centrality': {'classes': [cls], **BBCS}, 'particle_id': {'species': 'pi0'},
             'kinematic_range': {'y': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}, Fig. 1a. pi0 invariant yield, d+Au {cls} (BBC-S), |y| < 0.35; lab frame, the '
                                'deuteron toward +eta; pi0 computed by isospin. Bin edges inferred from the tabulated '
                                'centres (0.5 GeV/c to 10 GeV/c, 2 GeV/c above). Errors: stat and sys.'}}
        if rows != list(range(len(x))):
            d['table_rows'] = rows
        public.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '2645172' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records;', [d['centrality']['classes'][0] for d in public])


if __name__ == '__main__':
    main()
