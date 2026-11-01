#!/usr/bin/env python3
"""Curate PHOBOS charged-hadron pT spectra in Au+Au at 200 GeV, PLB 578, 297 (2004).

B. B. Back et al. (PHOBOS), "Charged hadron transverse momentum distributions
in Au+Au collisions at sqrt(sNN) = 200 GeV", Phys. Lett. B 578, 297 (2004),
arXiv:nucl-ex/0302015, INSPIRE 613385.  Table of Fig. 1 from the PHOBOS web
page (hepdata_raw/phobos/AuAu_Spectra_Npart_200/AuAu_Spectra_Fig1.txt).

From the paper:
- invariant yield (2 pi pT)^-1 d2N/(dy_pi dpT) of charged hadrons, the
  AVERAGE of positive and negative hadrons, (h+ + h-)/2;
- 0.2 < y_pi < 1.4, rapidity computed with the pion mass for every particle
  (kinematic_range.y.mass);
- six centrality classes from the Paddle counters (3 < |eta| < 4.5);
- corrected for acceptance, efficiency, secondaries and feed-down, ghost
  tracks, and for the finite bin width and momentum resolution: the values
  are the yields at the listed pT;
- the table's errors include the systematic ones (total, asymmetric).

The table gives the pT of each point: 0.25 ... 1.95 in steps of 0.1, then
0.2, 0.25 and 0.5.  The bins are the cells around them (edges halfway,
boundaries at 2.0, 3.0, 4.0 where the spacing changes); since the values
are corrected to the listed pT, a bin average differs from them by the
spectrum's curvature over the bin (< 1% for the 0.1-0.25 bins).

Not curated: Figs. 2-4 and R_AA (ratios to scaled yields and to p+p).

Run from the repository root:  python scripts/curate_phobos_spectra_200gev.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/AuAu_Spectra_Npart_200'
SRC = 'AuAu_Spectra_Npart_200/AuAu_Spectra_Fig1.txt'
CITE = 'B. B. Back et al. (PHOBOS), PLB 578, 297 (2004), arXiv:nucl-ex/0302015'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
M_PI = 0.13957


def blocks():
    """{class: rows of (pT, inv. yield, err up, err down)} under each 'Centrality X%' heading."""
    out, cur = {}, None
    for raw in open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1'):
        line = raw.replace('\r', '').strip()
        m = re.match(r'Centrality ([0-9]+-[0-9]+)%', line)
        if m:
            cur = out.setdefault(m.group(1), [])
        elif cur is not None and re.match(r'^[0-9.]+(\s+[0-9.eE+-]+){3}$', line):
            cur.append([float(t) for t in line.split()])
    return out


def edges(pts):
    """Bin edges with every point at the centre of its bin: the first bin is
    as wide as the first spacing, each further edge mirrors the previous one
    through the point."""
    e = [round(pts[0] - (pts[1] - pts[0]) / 2, 4)]
    for p in pts:
        e.append(round(2 * p - e[-1], 4))
    assert all(lo < p < hi for lo, hi, p in zip(e, e[1:], pts)), (pts, e)
    return e


def main():
    data = blocks()
    assert sorted(data) == sorted(['45-50', '35-45', '25-35', '15-25', '6-15', '0-6']), sorted(data)
    pts = [r[0] for r in data['0-6']]
    assert all([r[0] for r in rows] == pts for rows in data.values())
    pt_edges = edges(pts)
    classes = ['0-6', '6-15', '15-25', '25-35', '35-45', '45-50']
    write_value_table(
        os.path.join(RAW, 'tables/fig1_spectra.yaml'),
        f'{CITE}, Fig. 1: invariant yield of (h+ + h-)/2, 0.2 < y_pi < 1.4, by centrality; values at the '
        f'listed pT {pts} (bin-width corrected); errors include the systematic ones.',
        {'name': 'PT', 'units': 'GEV', 'bins': list(zip(pt_edges, pt_edges[1:]))},
        [{'name': '(2*PI*PT)**-1*D2N/DYPI/DPT', 'units': 'GEV**-2',
          'qualifiers': {'CENTRALITY': f'{c}%', 'PARTICLE': '(h+ + h-)/2'},
          'values': [r[1] for r in data[c]],
          'errors': [[{'label': 'stat+sys', 'plus': r[2], 'minus': r[3]}] for r in data[c]]} for c in classes])
    docs = []
    for i, c in enumerate(classes):
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '613385', 'version': 1,
            'hepdata_file': SRC, 'value_table': f'../../{RAW}/tables/fig1_spectra.yaml', 'dv_index': i,
            'hepdata_table_label': f'Fig. 1: charged-hadron invariant yield, {c}% Au+Au at 200 GeV',
            'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': [f'{c}%'], **PADDLE},
            'kinematic_range': {'y': {'min': 0.2, 'max': 1.4, 'mass': M_PI},
                                'pT': {'min': pt_edges[0], 'max': pt_edges[-1], 'bins': pt_edges}},
            'particle_id': {'species': 'charged', 'charge_average': True},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{CITE}, Fig. 1. (h+ + h-)/2 in 0.2 < y_pi < 1.4 (pion mass for every '
                                'particle). Corrected for secondaries, feed-down, bin width and momentum '
                                'resolution: values at the listed pT; bins reconstructed from the spacing of the '
                                'points. Errors: total (statistical and systematic), as published.')},
        })
    out = 'data/particle_spectra/phobos_particle_spectra_charged_AuAu_200gev_ins613385.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records; pT edges {pt_edges}')


if __name__ == '__main__':
    main()
