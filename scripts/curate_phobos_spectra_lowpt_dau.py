#!/usr/bin/env python3
"""Curate PHOBOS very-low-pT identified yields (Au+Au 200 GeV) and d+Au charged-hadron spectra.

- B. B. Back et al. (PHOBOS), "Particle production at very low transverse
  momenta in Au+Au collisions at sqrt(sNN) = 200 GeV", PRC 70, 051901(R)
  (2004), arXiv:nucl-ex/0401006, INSPIRE 642288.  Table
  hepdata_raw/phobos/Stopping/stopping_data.txt: invariant yields
  1/(2 pi pT) d2N/(dy dpT) of (pi+ + pi-), (K+ + K-), (p + pbar) -- sums of
  both charges -- in the 15% most central collisions, at mid-rapidity
  -0.1 < y < 0.4, in narrow pT ranges: 30-50 MeV/c (pions), 90-130 (kaons),
  140-210 (protons).  Particles stopping in the spectrometer, identified by
  their energy loss; corrected for secondaries, feed-down from weak decays
  and ghost tracks.  Statistical and systematic errors.  Centrality and
  vertex as in the 200 GeV analyses (Paddle counters).  Points quoted at
  their pT, no bin edges: windows of 5, 10 and 15 MeV/c.
- B. B. Back et al. (PHOBOS), "Centrality dependence of charged hadron
  transverse momentum spectra in d+Au collisions at sqrt(sNN) = 200 GeV",
  PRL 91, 072302 (2003), arXiv:nucl-ex/0306025, INSPIRE 621495.  Table of
  Fig. 1 (hepdata_raw/phobos/dAu_first/dAu1_Figure1.txt): invariant yield
  (2 pi pT)^-1 d2N/(deta dpT) of (h+ + h-)/2 over 0.2 < eta < 1.4 in the
  deuteron direction (positive eta), in four centrality classes from the
  energy in the Rings (3.0 < |eta| < 5.4); corrected for binning,
  momentum resolution, secondaries and feed-down (values at the listed pT;
  bins centred on them); systematic errors at 90% C.L.

Not curated: R_dAu, relative yields (ratios to p+pbar data).

Run from the repository root:  python scripts/curate_phobos_spectra_lowpt_dau.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.curate_phobos_spectra_200gev import edges  # noqa: E402
from scripts.value_tables import write_value_table  # noqa: E402

PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
RINGS = {'estimator': {'name': 'Rings', 'observable': 'energy', 'detector': 'Ring counters'},
         'eta': [{'min': -5.4, 'max': -3.0}, {'min': 3.0, 'max': 5.4}]}
LOWPT = 'B. B. Back et al. (PHOBOS), PRC 70, 051901 (2004), arXiv:nucl-ex/0401006'
DAU = 'B. B. Back et al. (PHOBOS), PRL 91, 072302 (2003), arXiv:nucl-ex/0306025'
LOWPT_SPECIES = {'(pi^+ + pi^-)': ('pi+ + pi-', 0.005), '(K^+ + K^-)': ('K+ + K-', 0.01),
                 '(p + pbar)': ('p + pbar', 0.015)}


def lowpt():
    raw = 'hepdata_raw/phobos/Stopping'
    blocks, cur = {}, None
    for ln in open(os.path.join(raw, 'stopping_data.txt'), encoding='latin-1'):
        ln = ln.replace('\r', '').strip()
        if ln in LOWPT_SPECIES:
            cur = blocks.setdefault(ln, [])
        elif cur is not None and re.match(r'^[0-9.]+(\s+[0-9.]+){4}$', ln):
            cur.append([float(t) for t in ln.split()])       # pT, mT, yield, stat, syst
    assert sorted(blocks) == sorted(LOWPT_SPECIES) and all(len(v) == 4 for v in blocks.values())
    docs = []
    for name, rows in blocks.items():
        sp, w = LOWPT_SPECIES[name]
        pts = [r[0] for r in rows]
        tag = sp.split()[0].replace('+', '')
        table = f'tables/lowpt_{tag}.yaml'
        write_value_table(os.path.join(raw, table),
                          f'{LOWPT}: invariant yield of {sp} (sum of charges), 0-15% Au+Au at 200 GeV, '
                          f'-0.1 < y < 0.4, at the quoted pT.',
                          {'name': 'PT', 'units': 'GEV', 'values': pts},
                          [{'name': '1/(2*PI*PT)*D2N/DY/DPT', 'units': 'GEV**-2',
                            'qualifiers': {'CENTRALITY': '0-15%', 'PARTICLE': sp},
                            'values': [r[2] for r in rows],
                            'errors': [[{'label': 'stat', 'sym': r[3]}, {'label': 'sys', 'sym': r[4]}] for r in rows]}])
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '642288', 'version': 1,
            'hepdata_file': 'Stopping/stopping_data.txt', 'value_table': f'../../{raw}/{table}', 'dv_index': 0,
            'hepdata_table_label': f'invariant yield of {sp} at very low pT, 0-15% Au+Au at 200 GeV',
            'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': ['0-15%'], **PADDLE},
            'kinematic_range': {'y': {'min': -0.1, 'max': 0.4},
                                'pT': {'min': round(pts[0] - w / 2, 4), 'max': round(pts[-1] + w / 2, 4),
                                       'points': pts, 'point_window': w}},
            'particle_id': {'species': sp},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{LOWPT}. Sum of both charges; particles stopping in the spectrometer, identified '
                                'by energy loss; -0.1 < y < 0.4 (true rapidity); corrected for secondaries, feed-down '
                                f'and ghost tracks. pT points without bin edges, windows of {w} GeV/c. Errors: '
                                'statistical and systematic. On a 3D grid coarser than these windows the result '
                                'rests on the within-cell interpolation (constant invariant yield).')},
        })
    out = 'data/particle_spectra/phobos_particle_spectra_identified_lowpt_AuAu_200gev_ins642288.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


def dau():
    raw = 'hepdata_raw/phobos/dAu_first'
    blocks, cur = {}, None
    for ln in open(os.path.join(raw, 'dAu1_Figure1.txt'), encoding='latin-1'):
        ln = ln.replace('\r', '').strip()
        m = re.match(r'Centrality: ([0-9]+-[0-9]+)%', ln)
        if m:
            cur = blocks.setdefault(m.group(1), [])
        elif cur is not None and re.match(r'^[0-9.]+(\s+[0-9.eE+-]+){3}$', ln):
            cur.append([float(t) for t in ln.split()])          # pT, yield, stat, syst
    classes = ['0-20', '20-40', '40-70', '70-100']
    assert sorted(blocks) == sorted(classes), sorted(blocks)
    pts = [r[0] for r in blocks['0-20']]
    assert all([r[0] for r in blocks[c]] == pts for c in classes)
    pt_edges = edges(pts)
    write_value_table(
        os.path.join(raw, 'tables/fig1_spectra.yaml'),
        f'{DAU}, Fig. 1: invariant yield of (h+ + h-)/2, 0.2 < eta < 1.4 (deuteron direction), d+Au at 200 GeV, '
        f'by centrality; values at the listed pT {pts}; systematic errors at 90% C.L.',
        {'name': 'PT', 'units': 'GEV', 'bins': list(zip(pt_edges, pt_edges[1:]))},
        [{'name': '(2*PI*PT)**-1*D2N/DETA/DPT', 'units': 'GEV**-2',
          'qualifiers': {'CENTRALITY': f'{c}%', 'PARTICLE': '(h+ + h-)/2'},
          'values': [r[1] for r in blocks[c]],
          'errors': [[{'label': 'stat', 'sym': r[2]}, {'label': 'sys,90% C.L.', 'sym': r[3]}] for r in blocks[c]]}
         for c in classes])
    docs = []
    for i, c in enumerate(classes):
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '621495', 'version': 1,
            'hepdata_file': 'dAu_first/dAu1_Figure1.txt', 'value_table': f'../../{raw}/tables/fig1_spectra.yaml',
            'dv_index': i,
            'hepdata_table_label': f'Fig. 1: charged-hadron invariant yield, {c}% d+Au at 200 GeV',
            'collision_system': {'energy_gev': 200.0, 'reaction': 'DEUT AU --> CHARGED X'},
            'centrality': {'classes': [f'{c}%'], **RINGS},
            'kinematic_range': {'eta': {'min': 0.2, 'max': 1.4},
                                'pT': {'min': pt_edges[0], 'max': pt_edges[-1], 'bins': pt_edges}},
            'particle_id': {'species': 'charged', 'charge_average': True},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{DAU}, Fig. 1. (h+ + h-)/2 in 0.2 < eta < 1.4, eta positive in the deuteron '
                                'direction; centrality from the Rings (fractions of the unbiased HIJING cross '
                                'section). Corrected for binning and momentum resolution: values at the listed pT, '
                                'bins centred on them. Errors: statistical, and systematic at 90% C.L.')},
        })
    out = 'data/particle_spectra/phobos_particle_spectra_charged_dAu_200gev_ins621495.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records; pT edges {pt_edges[:3]} ... {pt_edges[-3:]}')


if __name__ == '__main__':
    lowpt()
    dau()
