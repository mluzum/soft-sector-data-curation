#!/usr/bin/env python3
"""Curate STAR kaon production in Au+Au at 130 GeV, INSPIRE 588342.

C. Adler et al. (STAR), "Kaon production and kaon to pion ratio in Au+Au
collisions at sqrt(sNN) = 130 GeV", PLB 595 (2004) 143, nucl-ex/0206008.

From the paper:
- invariant yields 1/(2 pi m_T) d2N/(dm_T dy) of K+, K- and K0_S vs m_T - m
  (Fig. 1), per event (unscaled in HEPData), equal to 1/(2 pi pT)
  d2N/(dpT dy) at pT = sqrt(m_T^2 - m^2); charged kaons from dE/dx at
  |y| < 0.1 (the first 13 rows of each table) and from kink decays at
  |y| < 0.5 (the last 15, 11 rows filled below 11%), K0_S from V0 decays at
  |y| < 0.5; corrected for acceptance, decay, efficiency;
- errors: uncorrelated (dE/dx: stat and point-to-point sys; kink, K0_S:
  stat); overall systematics 5% (dE/dx), 10% (kink, K0_S) plus 5% common,
  not tabulated;
- centrality from the charged multiplicity in the TPC, the classes of STAR
  PRL 87 262302: 0-6, 6-11, 11-18, 18-26, 26-34, 34-45, 45-58, 58-85%
  (Table I, which gives their dN_h-/deta);
- mid-rapidity dN/dy of K+, K-, K0_S per class (Table I: m_T exponential
  fits, extrapolated; errors uncorrelated and correlated sys, plus 5% common).
  HEPData's K0_S dN/dy and slopes differ from Table I of arXiv v1 (2002;
  1.82, 5.40, ... there, 1.76, 5.83, ... in HEPData; K+- agree): the
  published version (2004) was revised; the records use HEPData.

Records (public): m_T spectra per species, method and class as pT-binned
invariant yields (bins from midpoints between the m_T points, the outer
edges mirrored; table_rows: the filled rows), dN/dy vs class (3 records).
Not curated: the inverse slopes T (fit parameters) and Fig. 3 (K/pi vs
sqrt(s): a compilation of other experiments' points).  The headers of the
spectra tables read 'K+/pi' (HEPData label; the description and Fig. 1 say
invariant yields).

Run from the repository root:  python scripts/curate_star_kaons_auau_130.py
"""
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins588342/HEPData-ins588342-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'C. Adler et al. (STAR), PLB 595 (2004) 143, nucl-ex/0206008'
CLASSES = ['0-6%', '6-11%', '11-18%', '18-26%', '26-34%', '34-45%', '45-58%', '58-85%']
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.75, 'max': 0.75}],
       'notes': 'Charged multiplicity in the TPC, |eta| < 0.75 (STAR 130 GeV; the classes of PRL 87 262302).'}
M_K, M_K0 = 0.493677, 0.497611
SPECTRA = {'k2piFig1.yaml': ('K+', M_K, 'AU AU --> K+ X'), 'k2piFig1_K-.yaml': ('K-', M_K, 'AU AU --> K- X'),
           'k2piFig1_neutral.yaml': ('K0_S', M_K0, 'AU AU --> K0S X')}
OUT_S = 'data/particle_spectra/star_particle_spectra_kaons_AuAu_130gev_ins588342.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_kaons_AuAu_130gev_ins588342.yaml'


def load(f):
    return yaml.safe_load(open(os.path.join(RAW, f)))


def pt_of(dm, m):
    mt = dm + m
    return math.sqrt(max(mt * mt - m * m, 0.0))


def mt_edges(points):
    mids = [0.5 * (a + b) for a, b in zip(points[:-1], points[1:])]
    return [max(0.0, points[0] - (mids[0] - points[0]))] + mids + [points[-1] + (points[-1] - mids[-1])]


def base(hf, k, reaction, species, label):
    return {'observable_type': 'particle_spectra', 'inspire_id': '588342', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': k, 'hepdata_table_label': f'{label} ({hf})',
            'collision_system': {'energy_gev': 130.0, 'reaction': reaction}, 'particle_id': {'species': species}}


def spectra():
    docs = []
    for hf, (species, m, reaction) in SPECTRA.items():
        t = load(hf)
        dm = [float(v['value']) for v in t['independent_variables'][0]['values']]
        groups = {'dEdx': list(range(13)), 'kink': list(range(13, 28))} if species != 'K0_S' else \
            {'V0': list(range(len(dm)))}
        assert t['independent_variables'][0]['header']['name'].startswith('$m_{T}-m_{K}'), hf
        for k, dv in enumerate(t['dependent_variables']):
            cls = re.fullmatch(r'\$(\d+-\d+)\\%\$ centrality', dv['qualifiers'][0]['value']).group(1) + '%'
            assert cls in CLASSES, cls
            method = 'V0' if species == 'K0_S' else ('dEdx' if k < 8 else 'kink')
            assert cls == CLASSES[k % 8], (hf, k, cls)
            rows_all = groups[method]
            rows = [j for j in rows_all if str(dv['values'][j]['value']).strip() != '']
            others = [j for j in range(len(dm)) if j not in rows_all]
            assert rows and all(str(dv['values'][j]['value']).strip() == '' for j in others), (hf, k)
            edges = mt_edges([dm[j] for j in rows_all])
            bins = [[round(pt_of(edges[i], m), 5), round(pt_of(edges[i + 1], m), 5)]
                    for i, j in enumerate(rows_all) if j in rows]
            ywin = 0.1 if method == 'dEdx' else 0.5
            how = {'dEdx': 'dE/dx', 'kink': 'kink decays', 'V0': 'K0_S -> pi+ pi- decays'}[method]
            d = base(hf, k, reaction, species, f'{species} invariant yield vs m_T - m, {cls}, {how}')
            d['centrality'] = {'classes': [cls], **NCH}
            d['kinematic_range'] = {'y': {'min': -ywin, 'max': ywin},
                                    'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
            d.update(table_rows=rows, observable='invariant_yield', value_type='invariant_yield')
            d['notes'] = {'value': f'{CITE}, Fig. 1. 1/(2 pi m_T) d2N/(dm_T dy) of {species} ({how}, |y| < {ywin:g}), '
                                   f'Au+Au 130 GeV {cls}, as 1/(2 pi pT) d2N/(dpT dy): pT bins from the midpoints '
                                   'between the tabulated m_T - m (outer edges mirrored). Errors: uncorrelated '
                                   '(overall 5-10% plus 5% common sys not tabulated).'}
            docs.append(d)
    return docs


def yields():
    t = load('k2piFig2_dndy.yaml')
    h = [float(v['value']) for v in t['independent_variables'][0]['values']]
    assert h == [17.9, 47.3, 78.9, 115.0, 154.0, 196.0, 236.0, 290.0], h
    classes = CLASSES[::-1]                     # Table I, peripheral first
    docs = []
    for k, (species, reaction) in enumerate((('K+', 'AU AU --> K+ X'), ('K-', 'AU AU --> K- X'),
                                             ('K0_S', 'AU AU --> K0S X'))):
        dv = t['dependent_variables'][k]
        assert [e['label'] for e in dv['values'][0]['errors']] == ['Uncorr', 'Corr'], dv['values'][0]
        d = {'observable_type': 'particle_yields', 'inspire_id': '588342', 'version': VERSION,
             'hepdata_file': 'k2piFig2_dndy.yaml', 'dv_index': k,
             'hepdata_table_label': f'{species} dN/dy vs centrality (k2piFig2_dndy.yaml)',
             'collision_system': {'energy_gev': 130.0, 'reaction': reaction},
             'centrality': {'classes': list(classes), **NCH}, 'particle_id': {'species': species},
             'kinematic_range': {'y': {'min': -0.5, 'max': 0.5}, 'pT': {'min': 0.0, 'max': 100.0}},
             'observable': 'dN_dy', 'value_type': 'dN_dy',
             'notes': {'value': f'{CITE}, Table I. Mid-rapidity dN/dy of {species} from m_T exponential fits '
                                '(extrapolated to all pT), classes 58-85 ... 0-6% (HEPData: their dN_h-/deta). '
                                'Errors: uncorrelated and correlated sys (plus 5% common, not tabulated).'
                                + (' HEPData (the published values) differs from arXiv v1 Table I for K0_S.'
                                   if species == 'K0_S' else '')}}
        docs.append(d)
    return docs


def main():
    for out, docs in ((OUT_S, spectra()), (OUT_Y, yields())):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
