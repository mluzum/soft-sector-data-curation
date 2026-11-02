#!/usr/bin/env python3
"""Curate PHENIX v2 of identified hadrons, phi and deuterons in Au+Au at 200 GeV, INSPIRE 746499.

S. Afanasiev et al. (PHENIX), "Elliptic flow for phi mesons and
(anti)deuterons in Au+Au collisions at sqrt(sNN) = 200 GeV", PRL 99,
052301 (2007), arXiv:nucl-ex/0703024 (PPG073).

From the paper:
- 2004 data; centrality from the BBC charge versus the ZDC energy; 20-60%;
- particles at |eta| <= 0.35 (central arms; phi -> K+ K-);
- reaction-plane method: Phi_2 from both BBCs (3 < |eta| < 3.9),
  v2 = <cos(2 (phi - Phi_2))> / resolution;
- errors: stat and sys.

Records (public): v2(pT) of pi+ + pi-, K+ + K-, p + pbar and phi in 20-60%
(Fig. 2a-d), BBC event plane, computed with the scalar-product estimator.
Staged (--staged OUT): d + dbar (Fig. 2e: light nuclei are not provided by
the pipeline).  Not curated: the invariant-mass and m^2 distributions
(Fig. 1) and v2 vs KE_T and the n_q-scaled forms (Fig. 3, the Fig. 2
values on other axes).  The 5 old records had the species charged (or p
for p + pbar) and no reference.

Run from the repository root:
    python scripts/curate_phenix_phi_deuteron_v2_ppg073.py [--staged OUT.yaml]
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins746499/HEPData-ins746499-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Afanasiev et al. (PHENIX), PRL 99, 052301 (2007), arXiv:nucl-ex/0703024'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge (with the ZDC energy in the paper).'}
REF = {'detector': 'BBC', 'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}], 'pT': {'min': 0.0, 'max': 100.0},
       'notes': 'Second-order event plane of both BBCs (charge weighting is not modelled).'}
TABLES = {'ppg073fig2a.yaml': 'pi+ + pi-', 'ppg073fig2b.yaml': 'K+ + K-', 'ppg073fig2c.yaml': 'p + pbar',
          'ppg073fig2d.yaml': 'phi', 'ppg073fig2e.yaml': 'd + dbar'}
OLD = 'data/flow_vn/phenix_flow_vn_elliptic_flow_AuAu_200gev.yaml'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, sp in TABLES.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv, dv = t['independent_variables'][0], t['dependent_variables'][0]
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        d = {'observable_type': 'flow_vn', 'inspire_id': '746499', 'version': VERSION, 'hepdata_file': hf, 'dv_index': 0,
             'hepdata_table_label': f'v2(pT) of {sp}, 20-60% ({hf})',
             'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
             'centrality': {'classes': ['20-60%'], **BBC}, 'particle_id': {'species': sp},
             'kinematic_range': {'eta': {'min': -0.35, 'max': 0.35},
                                 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
             'harmonic': 2, 'method': 'event_plane', 'flow_reference': dict(REF), 'value_type': 'v_2',
             'dimensionality': 'differential_pT',
             'notes': {'value': f'{CITE}, Fig. 2. v2(pT) of {sp}, 20-60%, |eta| < 0.35, BBC event plane (3 < |eta| '
                                '< 3.9), computed with the scalar-product estimator. Errors: stat and sys.'}}
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        if sp == 'd + dbar':
            d['staging_reason'] = '(anti)deuterons: light nuclei are not provided by the pipeline'
            staged.append(d)
        else:
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '746499' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_pid_phi_v2_AuAu_200gev_ins746499.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
