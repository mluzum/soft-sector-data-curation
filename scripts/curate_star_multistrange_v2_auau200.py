#!/usr/bin/env python3
"""Curate STAR Xi and Omega v2 in Au+Au at 200 GeV, INSPIRE 681161.

J. Adams et al. (STAR), "Multi-strange baryon elliptic flow in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRL 95 (2005) 122301,
arXiv:nucl-ex/0504022.

From the paper:
- Xi- + Xibar+ and Omega- + Omegabar+ from their weak decays, minimum
  bias (0-80%) Au+Au at 200 GeV (STAR's centrality: the charged
  multiplicity in |eta| < 0.5);
- v2(pT) relative to the second-order event plane of charged primary
  tracks with 0.2 < pT < 2.0 GeV/c and |eta| < 1.0, the candidates' tracks
  excluded; resolution from random subevents (1/0.72);
- points at the mean pT of each bin (no edges); statistical errors
  (absolute systematics 0.02 for the lowest Xi bin, < 0.01 otherwise,
  0.04 for Omega; not tabulated); non-flow not subtracted.  The baryons'
  rapidity window is not stated: the TPC acceptance |eta| < 1 here.

Records (public): v2(pT) of Xi- + Xibar+ and Omega- + Omegabar+ (Fig. 2)
at the tabulated pT, 0.1 GeV/c windows.  Staged (--staged OUT): the
invariant-mass and phi - Psi raw distributions (Fig. 1) and the
n_q-scaled v2 (Fig. 3).

Run from the repository root:  python scripts/curate_star_multistrange_v2_auau200.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins681161/HEPData-ins681161-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 95 (2005) 122301, arXiv:nucl-ex/0504022'
OUT = 'data/flow_vn/star_flow_vn_multistrange_v2_AuAu_200gev_ins681161.yaml'
WINDOW = 0.1
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': "STAR's charged multiplicity in |eta| < 0.5."}
SPECIES = {'$\\Xi+\\bar{\\Xi}$': ('Xi- + Xibar+', 'XI- XIBAR+'),
           '$\\Omega+\\bar{\\Omega}$': ('Omega- + Omegabar+', 'OMEGA- OMEGABAR+')}
TPC_EP = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.2, 'max': 2.0},
          'notes': 'Event plane of charged primary tracks, 0.2 < pT < 2.0 GeV/c, |eta| < 1.0, the candidates\' '
                   'tracks excluded.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if hf != 'Fig2.yaml':
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'flow_vn', 'inspire_id': '681161', 'version': VERSION,
                               'hepdata_file': hf, 'dv_index': i,
                               'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc[:200]}'},
                               'staging_reason': 'v2/n_q vs pT/n_q: a scaling display of Fig. 2' if hf == 'Fig3.yaml'
                               else 'raw invariant-mass or azimuthal distributions (detector level)'})
            continue
        xs = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            species, prod = SPECIES[dv['qualifiers'][0]['value']]
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            assert rows == list(range(rows[0], rows[-1] + 1)), rows
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat',)}
            pts = [xs[r] for r in rows]
            assert pts == sorted(pts)
            public.append({
                'observable_type': 'flow_vn', 'inspire_id': '681161', 'version': VERSION, 'hepdata_file': hf,
                'dv_index': i, 'hepdata_table_label': f'Fig. 2: v2(pT) of {species}, 0-80% ({hf} column {i})',
                'collision_system': {'energy_gev': 200.0, 'reaction': f'AU AU --> {prod} X'},
                'centrality': {'classes': ['0-80%'], **REFMULT}, 'particle_id': {'species': species},
                'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0},
                                    'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                           'points': pts, 'point_window': WINDOW}},
                'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(TPC_EP), 'value_type': 'v_2',
                'dimensionality': 'differential_pT', 'table_rows': rows,
                'notes': {'value': f'{CITE}, Fig. 2. v2 of {species} at the tabulated mean pT (0.1 GeV/c windows), '
                                   'minimum-bias (0-80%) Au+Au 200 GeV, relative to the TPC event plane; the rapidity '
                                   'window is not stated (|eta| < 1 here); non-flow not subtracted. Errors: '
                                   'statistical (absolute systematics 0.01-0.02 for Xi, 0.04 for Omega, not '
                                   'tabulated).'}})
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
