#!/usr/bin/env python3
"""Curate PHOBOS directed flow v1(eta), 0-40% Au+Au at 19.6-200 GeV, PRL 97, 012301 (2006).

B. B. Back et al. (PHOBOS), "Energy dependence of directed flow over a wide
range of pseudorapidity in Au+Au collisions at RHIC", Phys. Rev. Lett. 97,
012301 (2006), arXiv:nucl-ex/0511045, INSPIRE 698620.  Table of Fig. 2 from
the PHOBOS web page (hepdata_raw/phobos/AuAu_v1/v1Eta_AuAu_040Cent.txt):
v1 per eta bin (eta_low, eta_hi), 1 sigma statistical and asymmetric 90% C.L.
systematic errors.

From the paper:
- hit-based method, the data sets of the elliptic-flow study PRL 94,
  122303 (0-40%, collisions within +-10 cm; centrality from the Octagon at
  19.6 GeV and the Paddle counters above, as there);
- symmetric sub-events, each made of equal halves at negative and positive
  eta: octagon 1.5 < |eta| < 3 and rings 3 < |eta| < 5.  The octagon
  sub-event gives v1 in the rings region (|eta| > 3), the rings sub-event v1
  in the octagon region (|eta| < 3).  For directed flow the two halves
  enter with opposite signs: flow_reference.odd, with two segments;
- sign: "Historically, v1 has been defined to be positive (negative) at high
  positive (negative) eta ... it is necessary to invert the sign of v1 at
  130 and 200 GeV in order to preserve the sign convention from the lower
  energies".  So the records fix it at 19.6 and 62.4 GeV by v1 > 0 in the
  highest-eta bin (4.6-5.31), and at 130 and 200 GeV, where PHOBOS matched
  the shape of the lower energies, by v1 < 0 over 1.5 < eta < 3 (true at all
  four energies).

Not curated: Fig. 1 (method comparison in other centralities) and Fig. 3
(Fig. 2 against eta' = |eta| - ybeam).

Run from the repository root:  python scripts/curate_phobos_v1.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/AuAu_v1'
SRC = 'AuAu_v1/v1Eta_AuAu_040Cent.txt'
CITE = 'B. B. Back et al. (PHOBOS), PRL 97, 012301 (2006), arXiv:nucl-ex/0511045'
PADDLE = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
          'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
OCTAGON = {'estimator': {'name': 'Octagon', 'observable': 'energy', 'detector': 'Octagon'},
           'eta': [{'min': -3.0, 'max': 3.0}]}
RINGS_REF = [{'min': -5.0, 'max': -3.0}, {'min': 3.0, 'max': 5.0}]
OCT_REF = [{'min': -3.0, 'max': -1.5}, {'min': 1.5, 'max': 3.0}]
SIGN = {19.6: ({'min': 4.6, 'max': 5.31}, 'positive'), 62.4: ({'min': 4.6, 'max': 5.31}, 'positive'),
        130.0: ({'min': 1.5, 'max': 3.0}, 'negative'), 200.0: ({'min': 1.5, 'max': 3.0}, 'negative')}


def blocks():
    text = open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1').read().replace('\r', '')
    out = {}
    for m in re.finditer(r'\n([0-9.]+) GeV\s*\n<eta>[^\n]*\n((?:-?[0-9][^\n]*\n?)+)', text):
        out[float(m.group(1))] = [[float(t) for t in ln.split()] for ln in m.group(2).strip().splitlines()]
    return out           # rows: <eta>, eta_low, eta_hi, v1, stat, +syst, -syst


def main():
    data = blocks()
    assert sorted(data) == sorted(SIGN) and all(len(r) == 18 for r in data.values()), {k: len(v) for k, v in data.items()}
    docs = []
    for e, rows in sorted(data.items()):
        bins = [[r[1], r[2]] for r in rows]
        assert all(lo < r[0] < hi for (lo, hi), r in zip(bins, rows)), e
        sw, sign = SIGN[e]
        sel = [r[3] for r in rows if r[1] >= sw['min'] - 1e-9 and r[2] <= sw['max'] + 1e-9]
        assert sel and (sum(sel) > 0) == (sign == 'positive'), (e, sel)     # the published values obey it
        table = f'tables/fig2_v1_eta_0-40_{e:g}GeV.yaml'
        write_value_table(
            os.path.join(RAW, table),
            f'{CITE}, Fig. 2: v1 vs eta, Au+Au 0-40%, sqrt(sNN) = {e:g} GeV; mean eta {[r[0] for r in rows]}.',
            {'name': 'ETA', 'bins': bins},
            [{'name': 'V1', 'qualifiers': {'CENTRALITY': '0-40%', 'SQRT(S)/NUCLEON': f'{e:g} GeV'},
              'values': [r[3] for r in rows],
              'errors': [[{'label': 'stat', 'sym': r[4]},
                          {'label': 'sys,90% C.L.', 'plus': r[5], 'minus': r[6]}] for r in rows]}])
        docs.append({
            'observable_type': 'flow_vn', 'inspire_id': '698620', 'version': 1,
            'hepdata_file': SRC, 'value_table': f'../../{RAW}/{table}', 'dv_index': 0,
            'hepdata_table_label': f'Fig. 2: v1 vs eta, 0-40% Au+Au at sqrt(sNN) = {e:g} GeV',
            'collision_system': {'energy_gev': e, 'reaction': 'AU AU --> CHARGED X'},
            'centrality': {'classes': ['0-40%'], **(OCTAGON if e == 19.6 else PADDLE)},
            'kinematic_range': {'eta': {'min': -5.25, 'max': 5.31, 'bin_ranges': bins},
                                'pT': {'min': 0.0, 'max': 5000000000.0}},
            'particle_id': {'species': 'charged'},
            'method': 'event_plane', 'harmonic': 1, 'value_type': 'v1', 'dimensionality': 'differential_eta',
            'flow_reference': {
                'odd': True,
                'segments': [
                    {'applies_to_eta': {'min': 0.0, 'max': 3.0}, 'detector': 'Rings', 'eta': RINGS_REF,
                     'notes': 'v1 in the octagon region from the symmetric rings sub-event (3 < |eta| < 5).'},
                    {'applies_to_eta': {'min': 3.0, 'max': 5.5}, 'detector': 'Octagon', 'eta': OCT_REF,
                     'notes': 'v1 in the rings region from the symmetric octagon sub-event (1.5 < |eta| < 3).'}],
                'sign_convention': {'eta': sw, 'sign': sign},
                'notes': ('Symmetric sub-events, halves at negative and positive eta with opposite signs for '
                          'directed flow (momentum-conservation recoil cancels).  '
                          + ('Sign: v1 > 0 at the highest positive eta, the historical convention PHOBOS keeps '
                             'at this energy.' if sign == 'positive' else
                             'Sign: PHOBOS inverted v1 at 130 and 200 GeV to match the lower energies; '
                             'operationally v1 < 0 over 1.5 < eta < 3.'))},
            'notes': {'value': (f'{CITE}, Fig. 2. Hit-based method, all charged hadrons, 0-40%. eta bins as '
                                'published (eta_low, eta_hi). Errors: 1 sigma statistical and 90% C.L. systematic. '
                                'Centrality estimator as in PRL 94, 122303 (PRC 83, 024913 Sec. III A).')},
        })
    out = 'data/flow_vn/phobos_flow_vn_v1_eta_AuAu_0-40_ins698620.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
