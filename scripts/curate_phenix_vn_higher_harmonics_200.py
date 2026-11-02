#!/usr/bin/env python3
"""Curate PHENIX v2, v3, v4 of charged hadrons in Au+Au at 200 GeV, INSPIRE 900703.

A. Adare et al. (PHENIX), "Measurements of Higher-Order Flow Harmonics in
Au+Au Collisions at sqrt(sNN) = 200 GeV", PRL 107, 252301 (2011),
arXiv:1105.3928 (PPG132).

From the paper:
- 2007 run; centrality from the BBC multiplicity (3.1 < |eta| < 3.9);
- v_n{Psi_n}: tracks in the central arms (|eta| < 0.35) relative to the
  n-th order event plane of the RXN (1.0 < |eta| < 2.8, the average of the
  North and South subevents), which has the best resolution; BBC and MPC
  planes used for the systematic uncertainties;
- v4 is v4{Psi_4}, relative to the fourth-order plane;
- errors: stat and sys.

Records (public): v2{Psi_2}, v3{Psi_3} (0-10 ... 50-60%) and v4{Psi_4}
(0-10 ... 40-50%) vs pT (Fig. 2), computed with the scalar-product
estimator against the RXN windows.  Not curated: v_n vs centrality at 0.75
< pT < 1 and 1.75 < pT < 2 GeV/c (Fig. 3: the same points as Fig. 2), the
<pT> of each pT bin (Fig. 2.6: bin information) and N_part (Fig. 3.2,
Glauber).  The old public record read N_part as v2; the 8 staged ones were
pT-differential tables curated as integrated v2.

Run from the repository root:  python scripts/curate_phenix_vn_higher_harmonics_200.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins900703/HEPData-ins900703-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'A. Adare et al. (PHENIX), PRL 107, 252301 (2011), arXiv:1105.3928'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.1}, {'min': 3.1, 'max': 3.9}]}
ETA = {'min': -0.35, 'max': 0.35}
OLD = 'data/flow_vn/phenix_flow_vn_measurements_of_higher-order_flow_harmonics_200gev.yaml'


def rxn(n):
    return {'detector': 'RXN', 'eta': [{'min': -2.8, 'max': -1.0}, {'min': 1.0, 'max': 2.8}],
            'pT': {'min': 0.0, 'max': 100.0},
            'notes': f'Order-{n} event plane of the RXN, North and South averaged, 1.0 < |eta| < 2.8 (scintillator '
                     'weighting is not modelled).'}


def main():
    public = []
    for c in range(6):
        hf = f'Fig2.{c}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            m = re.fullmatch(r'v(\d) in centrality (\d+)-(\d+)%', h)
            n, cls = int(m.group(1)), f'{m.group(2)}-{m.group(3)}%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in ('', '-', '--', 'none')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            d = {'observable_type': 'flow_vn', 'inspire_id': '900703', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'v{n}(pT), {cls} ({hf})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'particle_id': {'species': 'charged'}, 'value_type': h, 'method': 'event_plane',
                 'flow_reference': rxn(n), 'centrality': {'classes': [cls], **BBC}, 'harmonic': n,
                 'dimensionality': 'differential_pT',
                 'kinematic_range': {'eta': dict(ETA), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'notes': {'value': f'{CITE}, Fig. 2. v{n}{{Psi_{n}}}(pT) of charged hadrons, {cls} (BBC), |eta| < 0.35, '
                                    f'relative to the RXN order-{n} event plane (1.0 < |eta| < 2.8); computed with the '
                                    'scalar-product estimator. Errors: stat and sys.'}}
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            public.append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '900703' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    out = 'data/flow_vn/phenix_flow_vn_v2_v3_v4_charged_AuAu_200gev_ins900703.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', [(d['harmonic'], d['centrality']['classes'][0]) for d in public])


if __name__ == '__main__':
    main()
