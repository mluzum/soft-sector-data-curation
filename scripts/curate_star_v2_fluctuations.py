#!/usr/bin/env python3
"""Curate STAR v2{2}, v2{4}, v2{6} vs centrality, INSPIRE 2017211.

M. S. Abdallah et al. (STAR), "Collision-system and beam-energy dependence
of anisotropic flow fluctuations", Phys. Rev. Lett. 129, 252301 (2022),
arXiv:2201.10365.  HEPData tables 200_GeV, 54_GeV, 39_GeV, 27_GeV, 19_GeV,
11_GeV (Au+Au, Fig. 3; 200 GeV also Fig. 1), U_U (193 GeV) and Cu_Au
(200 GeV) (Fig. 4): charged hadrons; 200_pi, 200_k, 200_p (Fig. 2):
identified hadrons in Au+Au at 200 GeV.

From the paper:
- charged hadrons 0.2 < pT < 4.0 GeV/c, identified pi, K, p (TPC dE/dx and
  TOF m^2) 0.2 < pT < 2.0 GeV/c; all particles |eta| < 1;
- v2{2} from the two-subevent method, |Delta eta| > 0.6: eta_a > 0.3 and
  eta_b < -0.3; v2{4} and v2{6} "traditional" cumulants with particle
  weights over the full acceptance;
- identified hadrons: for v2{4} "one identified hadron and three inclusive
  charged hadrons"; v2{2} likewise with the identified hadron in one
  subevent and charged hadrons in the other (vn2_gap_poi: POI-reference
  correlation over the charged v2{2}ab).  The charge sign is not given
  (both charges assumed), nor the reference pT range (0.2-4.0 GeV/c, the
  charged selection, assumed; with factorized flow it cancels);
- centrality from the RefMult multiplicity (|eta| < 0.5) with a Glauber
  fit (refs. [90, 94]: PRC 86, 054908; PRC 79, 034909).  The tables give
  one centrality value per point, no class edges, listed peripheral first;
  for Fig. 1 "the range 1-5% is used instead of 0-5%".  Records keep the
  values as centrality.points with windows as classes: 2% wide, or the
  closest spacing of a table's points when smaller (1% at 54.4 GeV, with
  points at 11% and 12%), so that classes never overlap.
- errors: statistical and systematic (the latter includes the residual
  non-flow estimate, 1% central to 13% peripheral at 200 GeV).

Blank cells (v2{6} of the most peripheral or central points at 39, 27 GeV
and in Cu+Au) are left out of the classes; the loader drops blank rows, so
the remaining classes stay aligned with the values.

Not curated: the ratios v2{4}/v2{2} and v2{6}/v2{4} (same-system quantity
ratios have no computation yet), and v2{4} of identified hadrons (a
differential four-particle cumulant with an identified particle of
interest is not implemented).

Run from the repository root:  python scripts/curate_star_v2_fluctuations.py
"""
import glob
import os

import yaml

CITE = 'M. S. Abdallah et al. (STAR), PRL 129, 252301 (2022), arXiv:2201.10365'
RAW = glob.glob('hepdata_raw/records/ins2017211/HEPData-ins2017211-v1-yaml')[0]
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}]}
WINDOW = 2.0          # at most; narrower when a table's points are closer
CHARGED_PT = {'min': 0.2, 'max': 4.0}
PID_PT = {'min': 0.2, 'max': 2.0}
# table -> (file tag, energy, reaction)
CHARGED = {'200_GeV': ('AuAu_200gev', 200.0, 'AU AU --> CHARGED X'),
           '54_GeV': ('AuAu_54.4gev', 54.4, 'AU AU --> CHARGED X'),
           '39_GeV': ('AuAu_39gev', 39.0, 'AU AU --> CHARGED X'),
           '27_GeV': ('AuAu_27gev', 27.0, 'AU AU --> CHARGED X'),
           '19_GeV': ('AuAu_19.6gev', 19.6, 'AU AU --> CHARGED X'),
           '11_GeV': ('AuAu_11.5gev', 11.5, 'AU AU --> CHARGED X'),
           'U_U': ('UU_193gev', 193.0, 'U U --> CHARGED X'),
           'Cu_Au': ('CuAu_200gev', 200.0, 'CU AU --> CHARGED X')}
IDENTIFIED = {'200_pi': ('pi', 'pi+ + pi-'), '200_k': ('K', 'K+ + K-'), '200_p': ('p', 'p + pbar')}
SUBEVENTS = {'A': {'eta': {'min': 0.3, 'max': 1.0}}, 'B': {'eta': {'min': -1.0, 'max': -0.3}}}


def width(points):
    srt = sorted(points)
    return min([WINDOW] + [b - a for a, b in zip(srt, srt[1:])])


def window(p, w):
    return f'{max(p - w / 2, 0):g}-{p + w / 2:g}%'


def table(name):
    return yaml.safe_load(open(os.path.join(RAW, f'{name}.yaml')))


def centrality(points, w):
    return {'classes': [window(p, w) for p in points], **REFMULT, 'points': points, 'point_window': w,
            'notes': ('RefMult (|eta| < 0.5) with a Glauber fit.  The table gives one centrality value per point '
                      f'and no class edges; classes are {w:g}%-wide windows centred on them.')}


def base(name, dv, cent_pts, energy, reaction, label, w):
    return {'observable_type': 'flow_vn', 'inspire_id': '2017211', 'version': 1,
            'hepdata_file': f'{name}.yaml', 'dv_index': dv, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': energy, 'reaction': reaction},
            'centrality': centrality(cent_pts, w)}


def columns(t):
    """{order: (dv index, [row indices with a value])} for the v2{k} columns."""
    out = {}
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        for k in (2, 4, 6):
            if f'({k}-PC)' in name and 'Ratio' not in name:
                rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
                out[k] = (i, rows)
    return out


def charged_docs(name):
    tag, energy, reaction = CHARGED[name]
    t = table(name)
    pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
    system = reaction.split(' -->')[0].title().replace(' ', '+')
    w = width(pts)
    docs = []
    for k, (dv, rows) in sorted(columns(t).items()):
        rows_pts = [pts[j] for j in rows]
        assert rows == list(range(rows[0], rows[-1] + 1)), (name, k, rows)      # blanks only at the ends
        d = base(name, dv, rows_pts, energy, reaction,
                 f'v2{{{k}}} of charged hadrons vs centrality, {system} at {energy:g} GeV', w)
        d.update({'particle_id': {'species': 'charged'},
                  'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': dict(CHARGED_PT)},
                  'harmonic': 2, 'method': 'cumulant', 'cumulant_order': k, 'value_type': f'v2{{{k}}}',
                  'dimensionality': 'integrated'})
        if k == 2:
            d.update({'particles': 'AB', 'eta_gap': 0.6,
                      'subevents': {lab: {'kinematic_range': {**s, 'pT': dict(CHARGED_PT)}}
                                    for lab, s in SUBEVENTS.items()}})
            method = 'two subevents, eta_a > 0.3 and eta_b < -0.3 (|Delta eta| > 0.6)'
        else:
            method = f'{k}-particle cumulant with particle weights over |eta| < 1'
        skipped = len(pts) - len(rows)
        d['notes'] = {'value': (f'{CITE}. Charged hadrons, 0.2 < pT < 4 GeV/c; v2{{{k}}}: {method}. Errors: '
                                'statistical and systematic.'
                                + (f' {skipped} blank table row(s) at the ends left out.' if skipped else '')
                                + (' Fig. 1: the most central point is the 1-5% range.'
                                   if name == '200_GeV' else ''))}
        docs.append(d)
    return f'data/flow_vn/star_flow_vn_v2_cumulants_{tag}_ins2017211.yaml', docs


def identified_docs(name):
    tag, species = IDENTIFIED[name]
    t = table(name)
    pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
    dv, rows = columns(t)[2]
    assert rows == list(range(len(pts)))
    d = base(name, dv, pts, 200.0, 'AU AU --> CHARGED X',
             f'v2{{2}} of {species} vs centrality, Au+Au at 200 GeV', width(pts))
    d.update({'particle_id': {'species': species},
              'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': dict(PID_PT)},
              'harmonic': 2, 'method': 'cumulant', 'cumulant_order': 2, 'value_type': 'v2{2}',
              'dimensionality': 'integrated', 'particles': 'AB', 'eta_gap': 0.6,
              'subevents': {lab: {'kinematic_range': {**s, 'pT': dict(CHARGED_PT)},
                                  'particle_id': {'species': 'charged'}} for lab, s in SUBEVENTS.items()},
              'notes': {'value': (f'{CITE}, Fig. 2. {species} with 0.2 < pT < 2 GeV/c (both charges assumed; the '
                                  'paper says pions, kaons, protons) in one subevent, charged hadrons in the other '
                                  '(eta_a > 0.3, eta_b < -0.3), over the charged v2{2}ab.  Reference pT range not '
                                  'stated: 0.2 < pT < 4 GeV/c (the charged selection) assumed.  Errors: statistical '
                                  'and systematic.')}})
    return f'data/flow_vn/star_flow_vn_v2_identified_{tag}_AuAu_200gev_ins2017211.yaml', [d]


def main():
    for name in list(CHARGED) + list(IDENTIFIED):
        out, docs = charged_docs(name) if name in CHARGED else identified_docs(name)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
