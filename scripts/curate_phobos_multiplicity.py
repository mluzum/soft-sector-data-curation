#!/usr/bin/env python3
"""Curate PHOBOS dNch/deta(eta) by centrality from PRC 83, 024913 (2011).

B. Alver et al. (PHOBOS), "Phobos results on charged particle multiplicity
and pseudorapidity distributions in Au+Au, Cu+Cu, d+Au, and p+p collisions
at ultra-relativistic energies", Phys. Rev. C 83, 024913 (2011),
arXiv:1011.1940, INSPIRE 876609.  Tables from the PHOBOS web page of the
paper (Publications/Physics/BigMult/Data/AA), kept in hepdata_raw/phobos/BigMult/.

From the paper:
- all charged particles excluding weak-decay products, corrected for
  missing low-pT particles (Sec. IV), in bins of 0.2 in eta over |eta| < 5.4
  (centres -5.3 ... 5.3);
- centrality from the energy deposited in the Paddle counters,
  3 < |eta| < 4.5, for Au+Au at 62.4-200 GeV and Cu+Cu at 62.4 and 200 GeV;
  in the Octagon, -3 < eta < 3, for Au+Au at 19.6 GeV and Cu+Cu at
  22.4 GeV; in the six Ring counters, 3 < |eta| < 5.4, for d+Au (Sec. III A);
- Au+Au (Figs. 15-18) and Cu+Cu (Figs. 19-21): asymmetric systematic errors
  at 90% C.L.; d+Au (Fig. 27): 1 sigma, eta "relative to the deuteron
  beam, which travels in the positive z direction" (the lab, which is the
  nucleon-nucleon frame at RHIC: collision_system.frame lab, shift 0,
  forward_beam d);
- the 19.6 GeV Au+Au data are a re-analysis, up to 10% above the earlier
  PRL 91, 052303 (2003) result (Sec. IV A).

Writes one value table per system and energy (hepdata_raw/phobos/BigMult/
tables/, one dependent variable per centrality class) and one record per
class (data/particle_yields/).  The d+Au minimum-bias and the p+p
distributions (no centrality classes) are not curated here.

Run from the repository root:  python scripts/curate_phobos_multiplicity.py
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/BigMult'
CITE = 'B. Alver et al. (PHOBOS), PRC 83, 024913 (2011), arXiv:1011.1940'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
OCTAGON = {'estimator': {'name': 'Octagon', 'observable': 'energy', 'detector': 'Octagon'},
           'eta': [{'min': -3.0, 'max': 3.0}]}
RINGS = {'estimator': {'name': 'Rings', 'observable': 'energy', 'detector': 'Ring counters'},
         'eta': [{'min': -5.4, 'max': -3.0}, {'min': 3.0, 'max': 5.4}]}

AA = ['0-3', '3-6', '6-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45',
      '45-50', '50-55']
SETS = [  # (file prefix, reaction, label, sqrt(sNN), classes, centrality, figure, 90% C.L.)
    ('AuAu_20GeV', 'AU AU --> CHARGED X', 'AuAu', 19.6, AA[:10], OCTAGON, 15, True),
    ('AuAu_63GeV', 'AU AU --> CHARGED X', 'AuAu', 62.4, AA[:11], PADDLE, 16, True),
    ('AuAu_130GeV', 'AU AU --> CHARGED X', 'AuAu', 130.0, AA[:11], PADDLE, 17, True),
    ('AuAu_200GeV', 'AU AU --> CHARGED X', 'AuAu', 200.0, AA[:11], PADDLE, 18, True),
    ('CuCu_22GeV', 'CU CU --> CHARGED X', 'CuCu', 22.4, AA, OCTAGON, 19, True),
    ('CuCu_62GeV', 'CU CU --> CHARGED X', 'CuCu', 62.4, AA, PADDLE, 20, True),
    ('CuCu_200GeV', 'CU CU --> CHARGED X', 'CuCu', 200.0, AA, PADDLE, 21, True),
    ('dAu_200GeV', 'DEUT AU --> CHARGED X', 'dAu', 200.0, ['0-20', '20-40', '40-60', '60-80', '80-100'],
     RINGS, 27, False),
]
ETA_EDGES = [round(-5.4 + 0.2 * i, 1) for i in range(55)]


def parse(path):
    """(eta, dN/deta, +err, -err) rows: 4 columns, or 5 with a zero eta error."""
    rows = []
    for raw in open(path, encoding='latin-1'):
        line = raw.replace('\r', '').replace('&', ' ').strip()
        if not line or line.startswith('#'):
            continue
        nums = [float(t) for t in line.split()]
        if len(nums) == 5:
            if nums[1] != 0:
                raise ValueError(f'{path}: eta error {nums[1]}')
            nums = [nums[0]] + nums[2:]
        if len(nums) != 4:
            raise ValueError(f'{path}: row {line!r}')
        rows.append((nums[0], nums[1], abs(nums[2]), abs(nums[3])))
    centres = [round(0.5 * (a + b), 1) for a, b in zip(ETA_EDGES, ETA_EDGES[1:])]
    if [round(r[0], 1) for r in rows] != centres:
        raise ValueError(f'{path}: eta centres differ from the 0.2 grid')
    return rows


# d+Au: eta relative to the deuteron beam (+z); equal beam energies per
# nucleon, so the lab is the nucleon-nucleon frame
DAU_FRAME = {'frame': 'lab', 'rapidity_shift': 0.0, 'forward_beam': 'd'}


def main():
    os.makedirs('data/particle_yields', exist_ok=True)
    for prefix, reaction, label, energy, classes, cent, fig, cl90 in SETS:
        err = 'sys,90% C.L.' if cl90 else 'sys,1 sigma'
        series = []
        for c in classes:
            rows = parse(os.path.join(RAW, f'{prefix}_Centrality_{c}.txt'))
            series.append({'name': 'DN/DETA', 'qualifiers': {
                'RE': reaction, 'SQRT(S)/NUCLEON': f'{energy} GeV', 'CENTRALITY': f'{c}%'},
                'values': [r[1] for r in rows],
                'errors': [[{'label': err, 'plus': r[2], 'minus': r[3]}] for r in rows]})
        table = f'tables/{label}_{energy:g}GeV_dNdeta.yaml'
        write_value_table(
            os.path.join(RAW, table),
            f'{CITE}, Fig. {fig}: dNch/deta vs eta, {label} at sqrt(sNN) = {energy:g} GeV, '
            f'by centrality.  Converted from the PHOBOS tables {prefix}_Centrality_<class>.txt.',
            {'name': 'ETA', 'bins': list(zip(ETA_EDGES, ETA_EDGES[1:]))}, series)
        docs = []
        for i, c in enumerate(classes):
            docs.append({
                'observable_type': 'particle_yields',
                'inspire_id': '876609',
                'version': 1,
                'hepdata_file': f'BigMult/{prefix}_Centrality_{c}.txt',
                'value_table': f'../../{RAW}/{table}',
                'dv_index': i,
                'hepdata_table_label': f'Fig. {fig}: dNch/deta vs eta, {c}% central {label} at '
                                       f'sqrt(sNN) = {energy:g} GeV',
                'collision_system': {'energy_gev': energy, 'reaction': reaction,
                                     **(DAU_FRAME if label == 'dAu' else {})},
                'centrality': {'classes': [f'{c}%'], **cent},
                'kinematic_range': {'eta': {'min': -5.4, 'max': 5.4, 'bins': ETA_EDGES},
                                    'pT': {'min': 0.0, 'max': 5000000000.0}},
                'particle_id': {'species': 'charged'},
                'observable': 'dNch_deta',
                'value_type': 'dN_deta',
                'notes': {'value': (
                    f'{CITE}.  Single-layer analysis (hit counting and energy deposition in '
                    f'the Octagon and Rings): all charged particles excluding weak-decay products, '
                    f'corrected for missing low-pT particles (no pT cut).  Errors are the published '
                    f'asymmetric systematic errors at {"90% C.L." if cl90 else "1 sigma"}.'
                    + (' Eta is measured along the deuteron direction.' if label == 'dAu' else '')
                    + (' A re-analysis superseding PRL 91, 052303 (2003) at this energy.'
                       if energy == 19.6 else ''))},
            })
        out = f'data/particle_yields/phobos_particle_yields_dNch_deta_eta_{label}_{energy:g}gev_ins876609.yaml'
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records; {RAW}/{table}')


if __name__ == '__main__':
    main()
