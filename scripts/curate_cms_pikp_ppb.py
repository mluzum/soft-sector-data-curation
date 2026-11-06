#!/usr/bin/env python3
"""Curate CMS pi, K, p spectra in pPb at 5.02 TeV vs track multiplicity, INSPIRE 1242440.

CMS Collaboration, "Study of the production of charged pions, kaons, and
protons in pPb collisions at sqrt(sNN) = 5.02 TeV", EPJC 74 (2014) 2847,
arXiv:1307.3442 (CMS-HIN-12-016).

From the paper:
- the higher-energy proton beam travelled at theta = pi: y_lab - y_cm =
  -0.465 (the Pb beam toward +eta); spectra at laboratory |y| < 1;
- d2N/(dy dpT) per event of pi+-, K+-, p, pbar (dE/dx identification,
  pT ~ 0.1-1.7 GeV/c), corrected to particle level;
- event classes in N_rec, the reconstructed tracks with |eta_lab| < 2.4
  (this analysis's tracking, down to pT ~ 0.1 GeV/c), 19 classes 0-9,
  10-19, ..., 180-189; Table 3 gives <N_tracks>, the corrected charged
  multiplicity in |eta| < 2.4 at all pT: 8, 19, 32, ..., 235.  The class
  midpoints over <N_tracks> (14.5/19, 24.5/32, ..., 184.5/235) give the
  efficiency of N_rec, 0.76-0.79: 0.77;
- errors: stat, sys, and an overall normalization uncertainty.

Records (public): the spectra of the six species in the 19 classes, in
track-count classes (centrality.multiplicity_bins: N_rec, efficiency
0.77, all pT in |eta_lab| < 2.4, here -1.935 < eta_cm < 2.865);
the nn frame (rapidity_shift -0.465, forward_beam Pb): |y_lab| < 1 is
-0.535 < y_cm < 1.465.  pT bins midway between the tabulated points
(0.05 GeV/c apart).  Staged (--staged OUT): the inclusive (minimum-bias)
spectra; the Tsallis-fit dN/dy, n, T, <pT> per class (Tables 41-46: no
uncertainties, some fits degenerate).

Run from the repository root:  python scripts/curate_cms_pikp_ppb.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1242440/HEPData-ins1242440-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'CMS Collaboration, EPJC 74 (2014) 2847, arXiv:1307.3442'
OUT = 'data/particle_spectra/cms_particle_spectra_pi_K_p_multiplicity_pPb_5020gev_ins1242440.yaml'
SHIFT = -0.465
EFF = 0.77
SPECIES = {'PI+': 'pi+', 'K+': 'K+', 'P': 'p', 'PI-': 'pi-', 'K-': 'K-', 'PBAR': 'pbar'}
CENT = {'estimator': {'name': 'Nrec', 'observable': 'Nch', 'detector': 'Tracker'},
        'eta': [{'min': round(-2.4 - SHIFT, 3), 'max': round(2.4 - SHIFT, 3)}], 'pT': {'min': 0.0, 'max': 100.0},
        'multiplicity_bins': {'efficiency': EFF,
                              'notes': 'N_rec over the corrected <N_tracks> of Table 3 (|eta| < 2.4, all pT): '
                                       'class midpoints 14.5/19 ... 184.5/235 = 0.76-0.79.'},
        'notes': 'N_rec: reconstructed tracks, |eta_lab| < 2.4, classes 0-9 ... 180-189 (integer counts, here '
                 'half-open 0-10 ... 180-190), in the nn frame -1.935 < eta_cm < 2.865.'}


def bins_from_points(pts):
    mids = [0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    edges = [pts[0] - (mids[0] - pts[0])] + mids + [pts[-1] + (pts[-1] - mids[-1])]
    return [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n in range(1, 41):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert t['independent_variables'][0]['header']['name'] == 'PT', hf
        x = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            assert dv['header']['name'] == '(1/Nev)*D2(N)/DYRAP/DPT', (hf, dv['header'])
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            assert q['ABS(YRAP)'] == '< 1', q
            prod = re.fullmatch(r'P PB --> (\S+) X', q['RE']).group(1)
            species = SPECIES[prod]
            rows = [r for r, v in enumerate(dv['values']) if v['value'] not in ('-', '', None)]
            assert rows == list(range(rows[0], rows[-1] + 1)), (hf, i)
            labels = {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows}
            assert labels == {('stat', 'sys', 'sys,Overall Normalization')}, labels
            pts = [x[r] for r in rows]
            assert all(abs(b - a - 0.05) < 1e-6 for a, b in zip(pts, pts[1:])), pts
            bins = bins_from_points(pts)
            if q['MULTBIN'] == 'Inclusive':
                cls, cent, label = 'MB', {'classes': ['MB'], **{k: v for k, v in CENT.items()
                                                                if k not in ('multiplicity_bins', 'notes')}}, 'inclusive'
            else:
                lo, hi = (int(float(v)) for v in q['Nrec'].split('-'))
                cls = f'{lo}-{hi + 1}'
                cent = {'classes': [cls], **CENT}
                label = f"N_rec {lo}-{hi} (<N_tracks> {q['MEAN(Ntracks)']})"
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1242440', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'{species} d2N/(dy dpT), pPb 5.02 TeV, {label} ({hf} column {i})',
                 'collision_system': {'energy_gev': 5020.0, 'reaction': f'P PB --> {prod} X', 'frame': 'nn_cms',
                                      'rapidity_shift': SHIFT, 'forward_beam': 'Pb'},
                 'centrality': cent, 'particle_id': {'species': species},
                 'kinematic_range': {'y': {'min': round(-1.0 - SHIFT, 3), 'max': round(1.0 - SHIFT, 3)},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy',
                 'notes': {'value': f'{CITE}. d2N/(dy dpT) per event of {species}, |y_lab| < 1 (-0.535 < y_cm < '
                                    f'1.465), pPb 5.02 TeV, {label}. pT bins midway between the tabulated points. '
                                    'Errors: stat, sys and the overall normalization.'}}
            if rows != list(range(len(x))):
                d['table_rows'] = rows
            if cls == 'MB':
                d['staging_reason'] = 'minimum-bias (inclusive) spectrum: no event class'
                staged.append(d)
            else:
                public.append(d)
    for n in range(41, 47):
        hf = f'Table{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv['qualifiers']}
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '1242440', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"Tsallis fit {q['Fit']}: {dv['header']['name']} vs class, "
                                                  f"{q['RE']} ({hf} column {i})",
                           'value_type': dv['header']['name'],
                           'notes': {'value': f'{CITE}. Tsallis-Pareto fit {q["Fit"]} per N_rec class: '
                                              f'{dv["header"]["name"]}.'},
                           'staging_reason': 'Tsallis-fit results per class (dN/dy and <pT> extrapolated by the '
                                             'fit), given without uncertainties; some fits degenerate (n ~ 1e6)'})
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
