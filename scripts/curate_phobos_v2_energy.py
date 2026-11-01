#!/usr/bin/env python3
"""Curate PHOBOS v2(eta) in 0-40% Au+Au at 19.6-200 GeV, PRL 94, 122303 (2005).

B. B. Back et al. (PHOBOS), "Energy dependence of elliptic flow over a large
pseudorapidity range in Au+Au collisions at RHIC", Phys. Rev. Lett. 94,
122303 (2005), arXiv:nucl-ex/0406021, INSPIRE 652825.  Table of Fig. 1 from
the PHOBOS web page (hepdata_raw/phobos/Flow_v2_limfrag/v2LimFrag.txt).

From the paper:
- hit-based method (octagon, rings), collisions within +-10 cm of the
  nominal vertex; event plane from the sub-event on the opposite side of
  midrapidity, sub-events 0.1 < |eta| < 3.0 at all four energies;
- v2 averaged over the 40% most central collisions (<Npart> = 201, 201, 209
  and 211); no pT selection;
- errors: 1 sigma statistical; systematic at 90% C.L. (the boxes).
Centrality: the paper refers to its Refs. [7, 9]; PRC 83, 024913 (Sec. III
A) gives the PHOBOS estimators, Octagon energy (-3 < eta < 3) at 19.6 GeV and
the Paddle counters (3 < |eta| < 4.5) above.

The table gives each point's systematic box: its horizontal bounds are the
eta bin (the v1 table of PRL 97, 012301 lists the same ranges as eta_low,
eta_hi), its vertical bounds the systematic range.  The bins are not
contiguous, so the records give kinematic_range.eta.bin_ranges.  At 19.6 GeV
eight lower box bounds equal -0.0050, which looks like the figure's lower
edge: for those points the lower systematic error may be larger.

Figs. 2 (averages over |eta| ranges) and 3 (Fig. 1 against eta -+ ybeam) are
derived from Fig. 1 and not curated.

Run from the repository root:  python scripts/curate_phobos_v2_energy.py
"""
import os
import re
import sys

import yaml

# Forward-detector references (calorimeter/scintillator) see every particle: reference pT 0-100 GeV/c.
ALL_PT_REF = {'min': 0.0, 'max': 100.0}
FWD_NOTE = (' Reference pT: all particles (a calorimeter/scintillator reference has no pT selection; its E_T or '
            'hit weighting is not modelled).')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/Flow_v2_limfrag'
SRC = 'Flow_v2_limfrag/v2LimFrag.txt'
CITE = 'B. B. Back et al. (PHOBOS), PRL 94, 122303 (2005), arXiv:nucl-ex/0406021'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
OCTAGON = {'estimator': {'name': 'Octagon', 'observable': 'energy', 'detector': 'Octagon'},
           'eta': [{'min': -3.0, 'max': 3.0}]}
REF = {'detector': 'Octagon', 'mirror': True, 'eta': [{'min': 0.1, 'max': 3.0}],
       'notes': 'Hit-based method: for v2 at positive (negative) eta, the event plane from the sub-event on '
                'the opposite side of midrapidity; sub-events 0.1 < |eta| < 3.0 at all four energies.' + FWD_NOTE,
       'pT': dict(ALL_PT_REF)}
ENERGIES = [19.6, 62.4, 130.0, 200.0]


def fig1_blocks():
    """{energy: rows of (eta, eta_e, v2, v2_e, sy_l, sy_r, sy_d, sy_u)} of Fig. 1."""
    text = open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1').read().replace('\r', '')
    fig1 = text.split('Figure  2:')[0]
    out = {}
    for m in re.finditer(r'\n([0-9.]+) GeV v2 vs\. eta\n(.*?)(?=\n-{10,}|\Z)', fig1, re.S):
        rows = [[float(t) for t in ln.split()] for ln in m.group(2).splitlines()
                if re.match(r'^\s*-?[0-9]', ln)]
        assert all(len(r) == 8 for r in rows)
        out[float(m.group(1))] = rows
    return out


def main():
    blocks = fig1_blocks()
    assert sorted(blocks) == ENERGIES and all(len(r) == 18 for r in blocks.values())
    docs = []
    for k, e in enumerate(ENERGIES):
        rows = blocks[e]
        ranges = [[r[4], r[5]] for r in rows]
        assert all(lo < r[0] < hi for (lo, hi), r in zip(ranges, rows)), e
        table = f'tables/fig1_v2_eta_0-40_{e:g}GeV.yaml'
        write_value_table(
            os.path.join(RAW, table),
            f'{CITE}, Fig. 1: v2 vs eta, Au+Au 0-40%, sqrt(sNN) = {e:g} GeV.  eta bins are the horizontal '
            f'bounds of the systematic boxes; mean eta {[r[0] for r in rows]}.',
            {'name': 'ETA', 'bins': ranges},
            [{'name': 'V2', 'qualifiers': {'CENTRALITY': '0-40%', 'SQRT(S)/NUCLEON': f'{e:g} GeV'},
              'values': [r[2] for r in rows],
              'errors': [[{'label': 'stat', 'sym': r[3]},
                          {'label': 'sys,90% C.L.', 'plus': r[7] - r[2], 'minus': r[2] - r[6]}] for r in rows]}])
        clipped = sum(1 for r in rows if r[6] == -0.005)
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '652825', 'version': 1,
            'hepdata_file': SRC, 'value_table': f'../../{RAW}/{table}', 'dv_index': 0,
            'hepdata_table_label': f'Fig. 1: v2 vs eta, 0-40% Au+Au at sqrt(sNN) = {e:g} GeV',
            'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': ['0-40%'], **(OCTAGON if e == 19.6 else PADDLE)},
            'kinematic_range': {'eta': {'min': -5.25, 'max': 5.31, 'bin_ranges': ranges},
                                'pT': {'min': 0.0, 'max': 5000000000.0}},
            'particle_id': {'species': 'charged'},
            'method': 'event_plane', 'harmonic': 2, 'value_type': 'v2', 'dimensionality': 'differential_eta',
            'flow_reference': REF,
            'notes': {'value': (
                f'{CITE}, Fig. 1. Hit-based method, all charged hadrons, 0-40% (<Npart> as published). '
                'eta bins from the systematic boxes (disjoint, so bin_ranges). Errors: 1 sigma statistical '
                'and 90% C.L. systematic, as published.  Centrality estimator per PRC 83, 024913 Sec. III A.'
                + (f' {clipped} lower systematic bounds equal -0.005, apparently the figure edge: their lower '
                   'systematic error may be larger.' if clipped else ''))},
        })
    out = 'data/flow_vn/phobos_flow_vn_v2_eta_AuAu_0-40_ins652825.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
