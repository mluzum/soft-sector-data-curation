#!/usr/bin/env python3
"""Curate PHOBOS v2 of charged hadrons in Au+Au at 200 GeV, PRC 72, 051901(R) (2005).

B. B. Back et al. (PHOBOS), "Centrality and pseudorapidity dependence of
elliptic flow for charged hadrons in Au+Au collisions at sqrt(sNN) = 200 GeV",
Phys. Rev. C 72, 051901 (2005), arXiv:nucl-ex/0407012, INSPIRE 654475.
Values from the PHOBOS table of the paper (hepdata_raw/phobos/Flow_v2_cent_pt/
final_ascii_table.txt), except where noted.

From the paper:
- centrality from the Paddle counters (3 < |eta| < 4.5), Table I classes
  3-15%, 15-25%, 25-50%; the track-based minimum-bias sample is 0-50%;
- track-based method: tracks in the spectrometer correlated with the event
  plane from the octagon sub-events at 2.05 < |eta| < 3.2 (both combined);
  straight-line tracks cover 0 < eta < 1.8 (no momentum measurement),
  curved tracks 0 < eta < 1.5 (Fig. 3, v2 vs pT);
- hit-based method "as described in" PRL 89, 222301 (2002): vertex offset to
  -34 cm, event plane from the octagon sub-event on the opposite side of
  midrapidity, 0.1 < |eta| < 2.0 in that paper;
- errors: 1 sigma statistical; systematic at 90% C.L. (asymmetric for the
  hit-based method); Fig. 5 shows combined 90% C.L. errors.

Points are published at their mean eta without bin edges (the figures carry
no horizontal error bars), so the eta-differential records give
kinematic_range.eta.points with a 0.2 window.  The Fig. 3 pT bins are the
horizontal error bars of the published figure (arXiv source
fig3_v2pt.eps, PostScript x = 249 + 459 pT): [0.04, 0.2, ..., 3.0, 4.0].

Fig. 5 (combined v2(eta)): the PHOBOS table's value column is shifted
against its eta column -- by one row for 25-50%, two for 15-25% and three
for 3-15%, padded with zeros -- while its errors line up with eta (they
match the published error boxes).  The values here are the published
markers (fig5_v2etacomb_inset_ratio_new.eps, v2 = (y - 213) / 18500,
resolution 5e-5); the paper's own combination rule confirms them (e.g. the
0.31 point combines the track-based 0.0638 and 0.0631 into 0.0635).

Not curated here (they stay with the records outside this database): Fig. 1
(hit-based "minimum bias", every triggered event with a valid vertex, not a
percentile class) and Fig. 2 (v2 against <Npart> only).

Run from the repository root:  python scripts/curate_phobos_v2_200gev.py
"""
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.value_tables import write_value_table  # noqa: E402

RAW = 'hepdata_raw/phobos/Flow_v2_cent_pt'
SRC = 'Flow_v2_cent_pt/final_ascii_table.txt'
CITE = 'B. B. Back et al. (PHOBOS), PRC 72, 051901 (2005), arXiv:nucl-ex/0407012'
CENT = {'estimator': {'name': 'Paddle', 'observable': 'energy', 'detector': 'Paddle counters'},
        'eta': [{'min': -4.5, 'max': -3.0}, {'min': 3.0, 'max': 4.5}]}
ALL_PT = {'min': 0.0, 'max': 5000000000.0}
WINDOW = 0.2
# Octagon (silicon) references see every particle: reference pT 0-100 GeV/c.
ALL_PT_REF = {'min': 0.0, 'max': 100.0}
FWD_NOTE = (' Reference pT: all particles (a calorimeter/scintillator reference has no pT selection; its E_T or '
            'hit weighting is not modelled).')
CLASSES = ['25-50', '15-25', '3-15']            # table order: peripheral, mid-central, central
TRACK_REF = {'detector': 'Octagon', 'eta': [{'min': -3.2, 'max': -2.05}, {'min': 2.05, 'max': 3.2}],
             'notes': 'Track-based method: event plane from the two octagon sub-events at 2.05 < |eta| < 3.2 '
                      '(parts with full azimuthal acceptance), combined; resolution from the sub-event '
                      'technique, by centrality and vertex.' + FWD_NOTE,
             'pT': dict(ALL_PT_REF)}
HIT_REF = {'detector': 'Octagon', 'mirror': True, 'eta': [{'min': 0.1, 'max': 2.0}],
           'notes': 'Hit-based method as described in PRL 89, 222301 (2002): for v2 at positive (negative) '
                    'eta, the event plane from the octagon sub-event on the opposite side, 0.1 < |eta| < 2.0 '
                    'in that paper (this paper does not restate the ranges); suppression corrected with '
                    'HIJING + GEANT.' + FWD_NOTE,
           'pT': dict(ALL_PT_REF)}
PT_EDGES = [0.04, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0]
FIG5_ETA = [0.31, 0.76, 1.22, 1.69, 3.04, 3.67, 4.13, 4.96]
FIG5_Y = {'25-50': [1388, 1345, 1268, 1208, 1042, 903, 722, 600],     # squares
          '15-25': [1168, 1144, 1086, 1031, 870, 737, 657, 557],      # triangles
          '3-15': [798, 807, 774, 753, 627, 642, 481, 483]}           # circles


def sections():
    """The table's numeric blocks, in order, each with the heading lines above it."""
    blocks, heads, cur = [], [], None
    for raw in open(os.path.join('hepdata_raw/phobos', SRC), encoding='latin-1'):
        line = raw.replace('\r', '').strip()
        if re.match(r'^-?[0-9.]+(\s+-?[0-9.]+){3,4}$', line):
            if cur is None:
                cur = {'heads': heads, 'rows': []}
                blocks.append(cur)
                heads = []
            cur['rows'].append([float(t) for t in line.split()])
        else:
            cur = None
            if line and not set(line) <= set('-'):
                heads.append(line)
    return blocks


def stat_sys(rows, sys_order):
    """errors per point: 1 sigma stat, then systematic (90% C.L.)."""
    out = []
    for r in rows:
        if sys_order == 'sym':
            out.append([{'label': 'stat', 'sym': r[2]}, {'label': 'sys,90% C.L.', 'sym': r[3]}])
        else:                                    # hit-based: Estat Elower Eupper
            out.append([{'label': 'stat', 'sym': r[2]},
                        {'label': 'sys,90% C.L.', 'plus': r[4], 'minus': r[3]}])
    return out


def record(name, fig, cls, kr, dim, ref, table, note):
    return {
        'observable_type': 'flow_vn', 'inspire_id': '654475', 'version': 1,
        'hepdata_file': SRC, 'value_table': f'../../{RAW}/tables/{table}', 'dv_index': 0,
        'hepdata_table_label': f'Fig. {fig}: {name}',
        'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'},
        'centrality': {'classes': [f'{cls}%'], **CENT},
        'kinematic_range': kr, 'particle_id': {'species': 'charged'},
        'method': 'event_plane', 'harmonic': 2, 'value_type': 'v2', 'dimensionality': dim,
        'flow_reference': ref,
        'notes': {'value': f'{CITE}. {note} Errors: 1 sigma statistical and 90% C.L. systematic, as published.'},
    }


def main():
    b = sections()
    # Order in the table: Fig.1 hit; Fig.2 track, hit; Fig.3; Fig.4 hit x3; Fig.4 track x3; Fig.5 x3.
    assert [len(x['rows']) for x in b] == [14, 11, 7, 12, 14, 14, 14, 6, 6, 6, 8, 8, 8], [len(x['rows']) for x in b]
    docs = []
    fig3 = b[3]['rows']
    write_value_table(os.path.join(RAW, 'tables/fig3_v2_pT_0-50.yaml'),
                      f'{CITE}, Fig. 3: track-based v2 vs pT, 0 < eta < 1.5, 0-50%; points at the mean pT '
                      f'of each bin (mean pT {[r[0] for r in fig3]}).',
                      {'name': 'PT', 'units': 'GEV', 'bins': list(zip(PT_EDGES, PT_EDGES[1:]))},
                      [{'name': 'V2', 'qualifiers': {'CENTRALITY': '0-50%'}, 'values': [r[1] for r in fig3],
                        'errors': stat_sys(fig3, 'sym')}])
    docs.append(record('track-based v2 vs pT, 0-50%', 3, '0-50',
                       {'eta': {'min': 0.0, 'max': 1.5}, 'pT': {'min': 0.04, 'max': 4.0, 'bins': PT_EDGES}},
                       'differential_pT', TRACK_REF, 'fig3_v2_pT_0-50.yaml',
                       'Fig. 3: track-based method, curved tracks through the spectrometer field, 0 < eta < 1.5. '
                       'pT bins from the published figure; the lowest edge reads 0.04 GeV/c.'))
    for k, cls in enumerate(CLASSES):
        rows = b[4 + k]['rows']
        t = f'fig4_hit_v2_eta_{cls}.yaml'
        write_value_table(os.path.join(RAW, 'tables', t),
                          f'{CITE}, Fig. 4: hit-based v2 vs eta, {cls}%, at the mean eta of each point.',
                          {'name': 'ETA', 'values': [r[0] for r in rows]},
                          [{'name': 'V2', 'qualifiers': {'CENTRALITY': f'{cls}%'}, 'values': [r[1] for r in rows],
                            'errors': stat_sys(rows, 'asym')}])
        docs.append(record(f'hit-based v2 vs eta, {cls}%', 4, cls,
                           {'eta': {'min': -5.4, 'max': 5.4, 'points': [r[0] for r in rows], 'point_window': WINDOW},
                            'pT': ALL_PT}, 'differential_eta', HIT_REF, t,
                           'Fig. 4: hit-based method (octagon and rings), all charged hadrons. Points at the published '
                           f'mean eta, computed in windows of {WINDOW}.'))
    for k, cls in enumerate(CLASSES):
        rows = b[7 + k]['rows']
        t = f'fig4_track_v2_eta_{cls}.yaml'
        write_value_table(os.path.join(RAW, 'tables', t),
                          f'{CITE}, Fig. 4: track-based v2 vs eta, {cls}%, at the mean eta of each point.',
                          {'name': 'ETA', 'values': [r[0] for r in rows]},
                          [{'name': 'V2', 'qualifiers': {'CENTRALITY': f'{cls}%'}, 'values': [r[1] for r in rows],
                            'errors': stat_sys(rows, 'sym')}])
        docs.append(record(f'track-based v2 vs eta, {cls}%', 4, cls,
                           {'eta': {'min': 0.0, 'max': 1.8, 'points': [r[0] for r in rows], 'point_window': WINDOW},
                            'pT': ALL_PT}, 'differential_eta', TRACK_REF, t,
                           'Fig. 4: track-based method, straight-line tracks (0 < eta < 1.8, no momentum '
                           f'measurement). Points at the published mean eta, computed in windows of {WINDOW}.'))
    for k, cls in enumerate(CLASSES):
        rows = b[10 + k]['rows']                 # eta, (shifted) v2, Eupper, Elower
        assert [r[0] for r in rows] == FIG5_ETA
        values = [round((y - 213) / 18500, 4) for y in FIG5_Y[cls]]
        t = f'fig5_combined_v2_eta_{cls}.yaml'
        write_value_table(os.path.join(RAW, 'tables', t),
                          f'{CITE}, Fig. 5: combined (hit- and track-based) v2 vs eta, {cls}%, eta > 0. Values '
                          f'read from the published figure (the PHOBOS table misaligns them); errors from the '
                          f'table (combined statistical and systematic, 90% C.L.).',
                          {'name': 'ETA', 'values': FIG5_ETA},
                          [{'name': 'V2', 'qualifiers': {'CENTRALITY': f'{cls}%'}, 'values': values,
                            'errors': [[{'label': 'stat+sys,90% C.L.', 'plus': r[2], 'minus': r[3]}] for r in rows]}])
        docs.append(record(f'combined v2 vs eta, {cls}%', 5, cls,
                           {'eta': {'min': 0.0, 'max': 5.4, 'points': FIG5_ETA, 'point_window': WINDOW},
                            'pT': ALL_PT}, 'differential_eta',
                           {'segments': [dict(TRACK_REF, applies_to_eta={'min': 0.0, 'max': 1.8}),
                                         dict(HIT_REF, applies_to_eta={'min': 1.8, 'max': 5.5})]}, t,
                           'Fig. 5: hit-based results reflected about midrapidity and averaged with the '
                           'track-based ones; eta > 0. It combines the Fig. 4 records: use one or the other. '
                           'Values read from the published figure, because the PHOBOS table shifts them against '
                           f'eta (see scripts/curate_phobos_v2_200gev.py). Windows of {WINDOW} at the mean eta.'))
        docs[-1]['notes']['value'] = docs[-1]['notes']['value'].replace(
            'Errors: 1 sigma statistical and 90% C.L. systematic', 'Errors: combined statistical and systematic at 90% C.L.')
    out = 'data/flow_vn/phobos_flow_vn_v2_AuAu_200gev_ins654475.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
