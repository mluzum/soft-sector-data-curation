#!/usr/bin/env python3
"""Curate ATLAS v_n(pT) without event-shape selection, INSPIRE 1357991.

G. Aad et al. (ATLAS), "Measurement of the correlation between flow
harmonics of different order in lead-lead collisions at sqrt(sNN) = 2.76
TeV", PRC 92, 034903 (2015), arXiv:1504.01289.  Tables "v_n data for
various q_2 bins, Centrality X%": v_n(pT), n = 2..5, in 19 centrality
classes, for q_2 fractions 0-0.1, 0.1-0.2, 0.7-0.8, 0.9-1.0 and 0-1 (all
events).

From the paper (Sec. III):
- tracks |eta| < 2.5, pT > 0.5 GeV; two-particle correlations of pairs
  with 2 < |Delta eta| < 5; v_n(pT) = v_nn(pT, pT_b) / sqrt(v_nn(pT_b,
  pT_b)) with the reference 0.5 < pT_b < 2 GeV;
- centrality from the FCal Sigma E_T, 3.2 < |eta| < 4.9;
- the tables give pT points (no bin edges) and statistical errors only
  (systematics of 1-5% are listed in Table II of the paper).

Records: the all-events column (q_2 fraction 0-1, dv_index 2); the q_3
tables repeat it (agreeing to 1e-4) and are not used.  method cumulant,
order 2, pT points with 0.1 GeV windows, reference pT from
flow_reference.pT.  The pair gap |Delta eta| > 2 is eta_gap, applied on 3D
containers (on boost-invariant 2D ones it only removes self-pairs).  Not curated: the event-shape-selected
columns (the pipeline has no q_n selection) and the v_m-v_n correlation
tables.

Run from the repository root:  python scripts/curate_atlas_ese_vn_pt.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1357991/HEPData-ins1357991-v*-yaml')[0]
CITE = 'G. Aad et al. (ATLAS), PRC 92, 034903 (2015), arXiv:1504.01289'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
WINDOW = 0.1


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    tables = {}
    for x in sub:
        m = re.match(r"\$v_\{(\d)\}\$ data for various \$q_2\$ bins, Centrality ([0-9]+-[0-9]+)%", x['description'].strip())
        if m:
            tables[(int(m.group(1)), m.group(2))] = x['data_file']
    assert len(tables) == 76, len(tables)
    docs = []
    for (n, cls), hf in sorted(tables.items(), key=lambda kv: (kv[0][0], int(kv[0][1].split('-')[0]), kv[0][1])):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = next(i for i, d in enumerate(t['dependent_variables']) if d['header']['name'].endswith('fraction 0-1'))
        assert dv == 2, (hf, dv)
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '1357991', 'version': 1, 'hepdata_file': hf,
            'dv_index': dv, 'hepdata_table_label': f'v_{n}(pT), all events (q_2 fraction 0-1), {cls}% Pb+Pb at 2.76 TeV',
            'collision_system': {'energy_gev': 2760.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [f'{cls}%'], **FCAL},
            'particle_id': {'species': 'charged'},
            'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5},
                                'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                       'points': pts, 'point_window': WINDOW}},
            'harmonic': n, 'method': 'cumulant', 'cumulant_order': 2, 'value_type': f'v_{n}',
            'dimensionality': 'differential_pT', 'eta_gap': 2.0,
            'flow_reference': {'pT': {'min': 0.5, 'max': 2.0},
                               'notes': 'Two-particle correlations, reference 0.5 < pT_b < 2 GeV, pairs with '
                                        '2 < |Delta eta| < 5 (the pair gap, eta_gap, is applied on 3D containers).'},
            'notes': {'value': (f'{CITE}. v_{n}(pT) of all events (the q_2 fraction 0-1 column) from two-particle '
                                'correlations, |eta| < 2.5; pT points without edges (0.1 GeV windows). Statistical '
                                'errors only (systematics 1-5%, Table II of the paper, not tabulated).')},
        })
    out = 'data/flow_vn/atlas_flow_vn_correlation_between_flow_harmonics_of_PbPb_2760gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
