#!/usr/bin/env python3
"""Curate PHOBOS identified-hadron pT spectra in Au+Au at 62.4 GeV, PRC 75, 024910 (2007).

B. B. Back et al. (PHOBOS), "Identified hadron transverse momentum spectra in
Au+Au collisions at sqrt(sNN) = 62.4 GeV", Phys. Rev. C 75, 024910 (2007),
arXiv:nucl-ex/0610001, INSPIRE 727645.  Table from the PHOBOS web page
(hepdata_raw/phobos/AuAu_PID_62/spectra_allspecies_allcbins.txt): pi+, pi-,
K+, K-, p, pbar in three centrality classes.

From the paper:
- invariant yields 1/(2 pi pT) d2N/(dpT dy) evaluated at a common rapidity
  y = 0.8 (Fig. 4); particle identification by dE/dx in the silicon
  spectrometer and by time of flight; points from several pT bins are merged
  and quoted at a single pT (no bin edges);
- corrected for acceptance, efficiency, occupancy, ghost tracks, momentum
  resolution, secondaries and feed-down from weak decays of Lambda and Sigma
  (protons are primary);
- centrality classes 0-15%, 15-30%, 30-50% from the Paddle counters
  (3.2 <= |eta| <= 4.5);
- statistical and systematic errors.

Records: species as identified; rapidity window 0.7 < y < 0.9 around the
published y = 0.8 (true rapidity of the species); pT as points with a
0.05 GeV/c window.

Run from the repository root:  python scripts/curate_phobos_pid_62.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/AuAu_PID_62'
SRC = 'AuAu_PID_62/spectra_allspecies_allcbins.txt'
CITE = 'B. B. Back et al. (PHOBOS), PRC 75, 024910 (2007), arXiv:nucl-ex/0610001'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.2}, {'min': 3.2, 'max': 4.5}]}
SPECIES = {'Positive pions': 'pi+', 'Negative pions': 'pi-', 'Positive kaons': 'K+', 'Negative kaons': 'K-',
           'Protons': 'p', 'Antiprotons': 'pbar'}
WINDOW = 0.05


def blocks():
    """{(class, species): rows (pT, yield, stat, syst)}."""
    out, cls, sp = {}, None, None
    for raw in open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1'):
        ln = raw.replace('\r', '').strip()
        m = re.match(r'CENTRALITY: ([0-9]+-[0-9]+)%', ln)
        if m:
            cls = m.group(1)
        elif ln in SPECIES:
            sp = SPECIES[ln]
        elif cls and sp and re.match(r'^[0-9.]+(\s+[0-9.eE+-]+){3}$', ln):
            out.setdefault((cls, sp), []).append([float(t) for t in ln.split()])
    return out


def main():
    data = blocks()
    classes = ['0-15', '15-30', '30-50']
    assert sorted(data) == sorted((c, s) for c in classes for s in SPECIES.values()), sorted(data)
    docs = []
    for c in classes:
        for sp in SPECIES.values():
            rows = data[(c, sp)]
            pts = [r[0] for r in rows]
            assert pts == sorted(pts)
            tag = sp.replace('+', 'plus').replace('-', 'minus')
            table = f'tables/spectra_{tag}_{c}.yaml'
            write_value_table(
                os.path.join(RAW, table),
                f'{CITE}: invariant yield of {sp} at y = 0.8, {c}% Au+Au at 62.4 GeV, at the quoted pT.',
                {'name': 'PT', 'units': 'GEV', 'values': pts},
                [{'name': '1/(2*PI*PT)*D2N/DPT/DY', 'units': 'GEV**-2',
                  'qualifiers': {'CENTRALITY': f'{c}%', 'PARTICLE': sp, 'YRAP': 0.8},
                  'values': [r[1] for r in rows],
                  'errors': [[{'label': 'stat', 'sym': r[2]}, {'label': 'sys', 'sym': r[3]}] for r in rows]}])
            docs.append({
                'observable_type': 'particle_spectra', 'inspire_id': '727645', 'version': 1,
                'hepdata_file': SRC, 'value_table': f'../../{RAW}/{table}', 'dv_index': 0,
                'hepdata_table_label': f'invariant yield of {sp} at y = 0.8, {c}% Au+Au at 62.4 GeV',
                'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> CHARGED X'},
                'centrality': {'classes': [f'{c}%'], **PADDLE},
                'kinematic_range': {'y': {'min': 0.7, 'max': 0.9},
                                    'pT': {'min': round(pts[0] - WINDOW / 2, 4), 'max': round(pts[-1] + WINDOW / 2, 4),
                                           'points': pts, 'point_window': WINDOW}},
                'particle_id': {'species': sp},
                'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                'notes': {'value': (f'{CITE}. Identified by dE/dx and time of flight; yields evaluated at y = 0.8 '
                                    '(computed over 0.7 < y < 0.9) and quoted at merged pT values without bin edges '
                                    f'(points, {WINDOW} GeV/c windows). Corrected for feed-down from Lambda and Sigma '
                                    'weak decays. Errors: statistical and systematic.')},
            })
    out = 'data/particle_spectra/phobos_particle_spectra_identified_AuAu_62.4gev_ins727645.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
