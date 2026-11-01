#!/usr/bin/env python3
"""Curate PHOBOS v2(eta) in 0-40% Cu+Cu at 62.4 and 200 GeV, PRL 98, 242302 (2007).

B. Alver et al. (PHOBOS), "System size, energy, pseudorapidity, and
centrality dependence of elliptic flow", Phys. Rev. Lett. 98, 242302 (2007),
arXiv:nucl-ex/0610037, INSPIRE 729951.  Table of Fig. 1 from the PHOBOS web
page (hepdata_raw/phobos/CuCu_v2/Fig1.txt).

From the paper: hit-based method as for Au+Au (PRL 94, 122303), event plane
from the sub-event on the opposite side of midrapidity, sub-events
0.1 < |eta| < 3.0 at both energies; centrality from the Paddle counters;
1 sigma statistical and 90% C.L. systematic errors (boxes).

The table gives each point's mean eta only.  The published figure's
systematic boxes (arXiv source figs/v2eta.eps) span the same 18 eta bins as
PRL 94, 122303 (to its 0.01 resolution), and its markers reproduce the
table's values to 5e-5: those bins are the records' bin_ranges.  At 62.4 GeV
the first point's lower systematic error equals v2 (a box clipped at zero).

Not curated: the Au+Au 200 GeV curve of Fig. 1 (published in PRL 94,
122303), Fig. 2 (Fig. 1 against eta - ybeam), Figs. 3 and 5 (v2 against
<Npart>, no centrality classes) and Fig. 4 (model eccentricities).

Run from the repository root:  python scripts/curate_phobos_v2_cucu.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/CuCu_v2'
SRC = 'CuCu_v2/Fig1.txt'
CITE = 'B. Alver et al. (PHOBOS), PRL 98, 242302 (2007), arXiv:nucl-ex/0610037'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
REF = {'detector': 'Octagon', 'mirror': True, 'eta': [{'min': 0.1, 'max': 3.0}],
       'notes': 'Hit-based method: for v2 at positive (negative) eta, the event plane from the sub-event on '
                'the opposite side of midrapidity; sub-events 0.1 < |eta| < 3.0.'}
BINS = [[-5.25, -4.57], [-4.5, -3.85], [-3.7, -3.15], [-3.0, -2.5], [-2.5, -2.0], [-2.0, -1.5],
        [-1.5, -1.0], [-1.0, -0.5], [-0.5, -0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0],
        [2.0, 2.5], [2.5, 3.0], [3.13, 3.72], [3.85, 4.52], [4.6, 5.31]]


def series():
    text = open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1').read().replace('\r', '')
    out = {}
    for m in re.finditer(r'// Cu-Cu ([0-9.]+)GeV \(0-40% centrality\)\n[^\n]*\n((?:\s*-?[0-9][^\n]*\n?)+)', text):
        out[float(m.group(1))] = [[float(t) for t in ln.split()] for ln in m.group(2).strip().splitlines()]
    return out           # rows: <eta>, eta err, v2, stat, -syst, +syst


def main():
    data = series()
    assert sorted(data) == [62.4, 200.0] and all(len(r) == 18 for r in data.values())
    docs = []
    for e, rows in sorted(data.items()):
        assert all(lo < r[0] < hi for (lo, hi), r in zip(BINS, rows))
        table = f'tables/fig1_v2_eta_CuCu_0-40_{e:g}GeV.yaml'
        write_value_table(
            os.path.join(RAW, table),
            f'{CITE}, Fig. 1: v2 vs eta, Cu+Cu 0-40%, sqrt(sNN) = {e:g} GeV.  eta bins from the published '
            f'systematic boxes; mean eta {[r[0] for r in rows]}.',
            {'name': 'ETA', 'bins': BINS},
            [{'name': 'V2', 'qualifiers': {'CENTRALITY': '0-40%', 'SQRT(S)/NUCLEON': f'{e:g} GeV'},
              'values': [r[2] for r in rows],
              'errors': [[{'label': 'stat', 'sym': r[3]},
                          {'label': 'sys,90% C.L.', 'plus': r[5], 'minus': r[4]}] for r in rows]}])
        clipped = [r[0] for r in rows if abs(r[4] - r[2]) < 1e-9]
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '729951', 'version': 1,
            'hepdata_file': SRC, 'value_table': f'../../{RAW}/{table}', 'dv_index': 0,
            'hepdata_table_label': f'Fig. 1: v2 vs eta, 0-40% Cu+Cu at sqrt(sNN) = {e:g} GeV',
            'collision_system': {'energy_gev': e, 'reaction': 'CU CU --> CHARGED X'},
            'centrality': {'classes': ['0-40%'], **PADDLE},
            'kinematic_range': {'eta': {'min': -5.25, 'max': 5.31, 'bin_ranges': BINS},
                                'pT': {'min': 0.0, 'max': 5000000000.0}},
            'particle_id': {'species': 'charged'},
            'method': 'event_plane', 'harmonic': 2, 'value_type': 'v2', 'dimensionality': 'differential_eta',
            'flow_reference': REF,
            'notes': {'value': (
                f'{CITE}, Fig. 1. Hit-based method, all charged hadrons, 0-40%. eta bins from the published '
                'systematic boxes (the table gives mean eta). Errors: 1 sigma statistical and 90% C.L. '
                'systematic, as published.'
                + (f' The lower systematic error at eta = {clipped} equals v2 (box clipped at zero).'
                   if clipped else ''))},
        })
    out = 'data/flow_vn/phobos_flow_vn_v2_eta_CuCu_0-40_ins729951.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
