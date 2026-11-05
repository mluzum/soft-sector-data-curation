#!/usr/bin/env python3
"""Curate STAR phi meson production in Au+Au at 62.4, 200 GeV and d+Au at 200 GeV, INSPIRE 797805.

B. I. Abelev et al. (STAR), "Measurements of phi meson production in
relativistic heavy-ion collisions at RHIC", PRC 79 (2009) 064903,
arXiv:0809.4737.

From the paper (Sec. II-III, Table III):
- phi -> K+ K- at |y| < 0.5 (spectra, <pT>), TPC tracks |eta| < 1;
- centrality: Au+Au from the raw charged multiplicity in |eta| < 0.5;
  d+Au from the Au-side FTPC (-3.8 < eta < -2.8); classes: Au+Au 62.4
  GeV 0-20 ... 60-80% and 0-80%; Au+Au 200 GeV 0-5, 0-10, 10-20 ... 70-80%;
  d+Au 0-20, 20-40, 40-100%, 0-100%;
- invariant yields 1/(2 pi m_T) d2N/(dm_T dy) vs m_T - m0 (Fig. 10), per
  event: the HEPData tables are not the display-scaled curves (their
  integrals reproduce Table III's dN/dy); Fig10c is the 200 GeV panel
  (HEPData labels it 62.4 GeV; its classes and integrals are 200 GeV's);
- <pT> from the m_T fits (Fig. 13, MeV/c);
- v2(pT) (Figs. 20, 21) relative to the TPC event plane (pT-weighted
  tracks |eta| < 1, the phi daughters excluded; resolution from random
  subevents), encoded as the event plane with the TPC reference, as
  STAR 620309;
- errors: stat (spectra, <pT>); stat and syst (v2).

Records (public): spectra (Fig. 10a, c, d d+Au), <pT> vs class (Fig. 13:
Au+Au 62.4, 200, d+Au), v2(pT) in 0-80% (Fig. 20, invariant-mass method)
and in 0-5, 10-40, 40-80% (Fig. 21).  Not curated: the 130 GeV panels
(STAR PLB 612 (2005) 181, its own record), p+p (no centrality), dN/dy
(tabulated only over 0.5 N_part), the reaction-plane-method v2 of Fig. 20
(the same v2), mass/width, ratios, R_CP, R_dAu, efficiencies.

Run from the repository root:  python scripts/curate_star_phi_production.py
"""
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins797805/HEPData-ins797805-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PRC 79 (2009) 064903, arXiv:0809.4737'
M_PHI = 1.019461
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'raw charged multiplicity in the TPC, |eta| < 0.5.'}
FTPC = {'estimator': {'name': 'FTPC-East', 'observable': 'Nch', 'detector': 'FTPC'}, 'eta': [{'min': -3.8, 'max': -2.8}],
        'notes': 'charged multiplicity in the Au-side FTPC.'}
AUAU = lambda e: {'energy_gev': e, 'reaction': 'AU AU --> PHI X'}  # noqa: E731
DAU = {'energy_gev': 200.0, 'reaction': 'D AU --> PHI X', 'frame': 'lab', 'rapidity_shift': 0.0, 'forward_beam': 'd'}
# spectra tables: file -> (collision_system, estimator, label energy expected in HEPData)
SPECTRA = {'Fig10a.yaml': (AUAU(62.4), REFMULT, '62.4'), 'Fig10c.yaml': (AUAU(200.0), REFMULT, '62.4'),
           'Fig10d_1.yaml': (DAU, FTPC, '200')}
# <pT> tables: file -> (collision_system, estimator, classes in table order)
MEANPT = {'Fig13top_AuAu200.yaml': (AUAU(200.0), REFMULT, ['0-5%', '0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                                                           '50-60%', '60-70%', '70-80%']),
          'Fig13top_AuAu62.yaml': (AUAU(62.4), REFMULT, ['0-20%', '20-40%', '40-60%', '60-80%']),
          'Fig13top_dAu200.yaml': (DAU, FTPC, ['0-20%', '20-40%', '40-100%'])}
TPC_EP = {'detector': 'TPC', 'eta': [{'min': -1.0, 'max': 1.0}], 'pT': {'min': 0.15, 'max': 2.0},
          'notes': 'TPC event plane, |eta| < 1, pT-weighted tracks, the phi daughters excluded.'}
OUT = {'particle_spectra': 'data/particle_spectra/star_particle_spectra_phi_AuAu_dAu_ins797805.yaml',
       'mean_pT': 'data/mean_pT/star_mean_pT_phi_AuAu_dAu_ins797805.yaml',
       'flow_vn': 'data/flow_vn/star_flow_vn_phi_v2_AuAu_200gev_ins797805.yaml'}


def load(f):
    return yaml.safe_load(open(os.path.join(RAW, f)))


def mid_edges(points):
    mids = [0.5 * (a + b) for a, b in zip(points[:-1], points[1:])]
    return [max(0.0, points[0] - (mids[0] - points[0]))] + mids + [points[-1] + (points[-1] - mids[-1])]


def pt_of(dm):
    mt = dm + M_PHI
    return math.sqrt(mt * mt - M_PHI * M_PHI)


def head(otype, hf, k, cs, label):
    return {'observable_type': otype, 'inspire_id': '797805', 'version': VERSION, 'hepdata_file': hf, 'dv_index': k,
            'hepdata_table_label': f'{label} ({hf})', 'collision_system': dict(cs),
            'particle_id': {'species': 'phi'}}


def system_label(cs):
    return f'{"d+Au" if cs["reaction"].startswith("D") else "Au+Au"} {cs["energy_gev"]:g} GeV'


def spectra():
    docs = []
    for hf, (cs, est, lab_e) in SPECTRA.items():
        t = load(hf)
        dm = [float(v['value']) for v in t['independent_variables'][0]['values']]
        edges = [pt_of(x) for x in mid_edges(dm)]
        for k, dv in enumerate(t['dependent_variables']):
            q = dv['qualifiers'][0]['value'].replace('$', '').replace('\\', '')
            m = re.search(r'([\d.]+) GeV \((\d+-\d+)%\)', q)
            assert m and m.group(1) == lab_e, (hf, q)
            cls = m.group(2) + '%'
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            bins = [[round(edges[j], 5), round(edges[j + 1], 5)] for j in rows]
            d = head('particle_spectra', hf, k, cs, f'phi invariant yield vs m_T - m0, {system_label(cs)} {cls}')
            d['centrality'] = {'classes': [cls], **est}
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            if len(rows) < len(dm):
                d['table_rows'] = rows
            d.update(observable='invariant_yield', value_type='invariant_yield')
            if hf == 'Fig10c.yaml':
                d['hepdata_corrections'] = {'energy': 'HEPData labels the panel 62.4 GeV; its classes (0-5 ... 70-80%) '
                                                      'and integrals (dN/dy of Table III) are the 200 GeV ones'}
            d['notes'] = {'value': f'{CITE}, Fig. 10. 1/(2 pi m_T) d2N/(dm_T dy) of phi, |y| < 0.5, {system_label(cs)} '
                                   f'{cls}, as 1/(2 pi pT) d2N/(dpT dy): pT bins from midpoints between the tabulated '
                                   'm_T - m0 (outer edges mirrored). Errors: stat (sys in Table III, not tabulated).'}
            docs.append(d)
    return docs


def mean_pt():
    docs = []
    for hf, (cs, est, classes) in MEANPT.items():
        t = load(hf)
        npart = [float(v['value']) for v in t['independent_variables'][0]['values']]
        assert len(npart) == len(classes), (hf, npart)
        dv = t['dependent_variables'][0]
        assert '(MeV/c)' in dv['header']['name'], dv['header']
        # overlapping classes (0-5% and 0-10% at 200 GeV): one record per non-overlapping set
        sets = [list(range(1, len(classes))), [0]] if classes[:2] == ['0-5%', '0-10%'] else [None]
        for rows in sets:
            cl = classes if rows is None else [classes[j] for j in rows]
            d = head('mean_pT', hf, 0, cs, f'phi <pT> vs centrality, {system_label(cs)}')
            d['centrality'] = {'classes': cl, **est}
            if rows is not None:
                d['table_rows'] = rows
            d['kinematic_range'] = {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}}
            d.update(observable='mean_pT', value_type='mean_pT')
            d['notes'] = {'value': f'{CITE}, Fig. 13 and Table III. <pT> of phi from the m_T fits (all pT), |y| < '
                                   f'0.5, {system_label(cs)}, classes {", ".join(cl)} (HEPData: their N_part), MeV/c. '
                                   'Errors: stat (sys in Table III, not tabulated).'}
            docs.append(d)
    return docs


def flow():
    docs = []
    for hf, k, cls in (('Fig20.yaml', 0, '0-80%'), ('Fig21_1.yaml', 0, '0-5%'), ('Fig21_2.yaml', 0, '10-40%'),
                       ('Fig21_2.yaml', 1, '40-80%')):
        t = load(hf)
        dv = t['dependent_variables'][k]
        q = dv['qualifiers'][0]['value']
        assert (cls.replace('%', '') in q.replace('$', '')) or (hf == 'Fig20.yaml' and 'invariant' in q), (hf, q)
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        edges = mid_edges(pts)
        bins = [[round(edges[i], 4), round(edges[i + 1], 4)] for i in range(len(pts))]
        d = head('flow_vn', hf, k, AUAU(200.0), f'phi v2(pT), Au+Au 200 GeV {cls}')
        d['centrality'] = {'classes': [cls], **REFMULT}
        d['kinematic_range'] = {'eta': {'min': -1.0, 'max': 1.0},
                                'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
        d.update(harmonic=2, method='event_plane', flow_reference=dict(TPC_EP), value_type='v2',
                 dimensionality='differential_pT')
        d['notes'] = {'value': f'{CITE}, Fig. {"20 (invariant-mass method)" if hf == "Fig20.yaml" else "21"}. v2(pT) of '
                               f'phi, Au+Au 200 GeV {cls}, relative to the TPC event plane (pT-weighted, daughters '
                               'excluded), computed with the scalar-product estimator; bin edges midway between the '
                               'tabulated pT. Errors: stat and syst.'}
        docs.append(d)
    return docs


def main():
    for otype, docs in (('particle_spectra', spectra()), ('mean_pT', mean_pt()), ('flow_vn', flow())):
        out = OUT[otype]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
