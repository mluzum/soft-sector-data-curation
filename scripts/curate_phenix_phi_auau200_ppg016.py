#!/usr/bin/env python3
"""Curate PHENIX phi-meson production in Au+Au at 200 GeV, INSPIRE 661505.

S. S. Adler et al. (PHENIX), "Production of phi mesons at mid-rapidity in
sqrt(sNN) = 200 GeV Au+Au collisions at RHIC", PRC 72, 014903 (2005),
nucl-ex/0410012 (PPG016).

From the paper:
- phi -> K+ K- in the central arms (|eta| < 0.35), yields per unit
  rapidity at mid-rapidity;
- centrality from the BBC (3.0 < |eta| < 3.9) and ZDC correlation: 0-10,
  10-40, 40-92% and minimum bias (0-92%);
- invariant yields 1/(2 pi mT) d2N/(dmT dy) in mT bins (Appendix table:
  centres and widths matching the mT edges of HEPData fig7); dN/dy and the
  inverse slope from exponential fits in mT (Table V);
- errors: stat and sys.

Records (public): phi invariant yields in 0-10, 10-40, 40-92% (fig7) and
0-92% (fig11, whose axis is mT - 1 GeV/c^2 labelled mT: the first three
columns equal fig7 shifted by 1.0), pT = sqrt(mT^2 - m_phi^2), |y| < 0.35;
dN/dy in 0-10, 10-40, 40-92% and, separately, 0-92% (fig5).  Staged
(--staged OUT): phi/pi and phi/K- vs centrality and in minimum bias (figs.
14b, 16: ratios to the pion and kaon yields of INSPIRE 624474; particle
ratios are not computed).  Not curated: Glauber values (fig1), the mass
centroid and width (figs. 2, 9), subsystem cross-checks (figs. 4, 10), the
inverse slopes and blast-wave fits (figs. 5 T, 6, 13, 17), the same dN/dy
per participant pair (fig12), other experiments (fig15), K/pi of INSPIRE
624474 (fig14a) and R_CP (fig23).  The old record read fig15a (dN/dy per
participant pair) as dN/dy.

Run from the repository root:
    python scripts/curate_phenix_phi_auau200_ppg016.py [--staged OUT.yaml]
"""
import argparse
import glob
import math
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins661505/HEPData-ins661505-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. S. Adler et al. (PHENIX), PRC 72, 014903 (2005), nucl-ex/0410012'
BBC = {'estimator': {'name': 'BBC', 'observable': 'amplitude', 'detector': 'BBC'},
       'eta': [{'min': -3.9, 'max': -3.0}, {'min': 3.0, 'max': 3.9}],
       'notes': 'BBC charge with the ZDC energy.'}
Y = {'min': -0.35, 'max': 0.35}
M_PHI = 1.019461
OLD = 'data/particle_yields/phenix_particle_yields_production_of_phi_mesons_AuAu_200gev.yaml'
OUT_SPEC = 'data/particle_spectra/phenix_particle_spectra_phi_AuAu_200gev_ins661505.yaml'
OUT_YIELD = 'data/particle_yields/phenix_particle_yields_phi_AuAu_200gev_ins661505.yaml'
FIG11_FIX = "HEPData fig11 labels its axis mT; the edges are mT - 1.0 GeV/c^2 (fig7 and the paper's Appendix table)"


def load(hf):
    return yaml.safe_load(open(os.path.join(RAW, hf)))


def pt(mt):
    return round(math.sqrt(max(mt * mt - M_PHI * M_PHI, 0.0)), 4)


def spectrum(hf, i, cls, mt_edges, extra=None):
    pairs = [[pt(a), pt(b)] for a, b in mt_edges]
    d = {'observable_type': 'particle_spectra', 'inspire_id': '661505', 'version': VERSION, 'hepdata_file': hf,
         'dv_index': i, 'hepdata_table_label': f'phi invariant yield, {cls} ({hf})',
         'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> PHI X'},
         'centrality': {'classes': [cls], **BBC}, 'particle_id': {'species': 'phi'},
         'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
         'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
         'notes': {'value': f'{CITE}. 1/(2 pi mT) d2N/(dmT dy) = 1/(2 pi pT) d2N/(dpT dy) of phi (K+ K-), {cls}, '
                            'mid-rapidity (|y| < 0.35 used); mT bins converted to pT = sqrt(mT^2 - m_phi^2). Errors: '
                            'stat and sys.'}}
    if extra:
        d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, staged = [], [], []
    t7 = load('ppg016fig7.yaml')
    e7 = [(float(v['low']), float(v['high'])) for v in t7['independent_variables'][0]['values']]
    for i, dv in enumerate(t7['dependent_variables']):
        spectra.append(spectrum('ppg016fig7.yaml', i, dv['header']['name'], e7))
    t11 = load('ppg016fig11.yaml')
    e11 = [(float(v['low']) + 1.0, float(v['high']) + 1.0) for v in t11['independent_variables'][0]['values']]
    assert all(abs(a - b) < 1e-9 and abs(c - d) < 1e-9 for (a, c), (b, d) in zip(e11, e7)), (e11, e7)
    for i, dv in enumerate(t11['dependent_variables']):
        if dv['header']['name'] == '0-92%':
            spectra.append(spectrum('ppg016fig11.yaml', i, '0-92%', e11, {'hepdata_corrections': {'pT': FIG11_FIX}}))
    t5 = load('ppg016fig5.yaml')
    classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in t5['independent_variables'][0]['values']]
    i = [dv['header']['name'] for dv in t5['dependent_variables']].index('dN/dy')
    for rows in ([j for j, c in enumerate(classes) if c != '0-92%'], [classes.index('0-92%')]):
        yields.append({'observable_type': 'particle_yields', 'inspire_id': '661505', 'version': VERSION,
                       'hepdata_file': 'ppg016fig5.yaml', 'dv_index': i,
                       'hepdata_table_label': 'phi dN/dy vs centrality (ppg016fig5.yaml)',
                       'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> PHI X'},
                       'centrality': {'classes': [classes[j] for j in rows], **BBC}, 'particle_id': {'species': 'phi'},
                       'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                       'observable': 'dN_dy', 'value_type': 'dN_dy', 'table_rows': rows,
                       'notes': {'value': f'{CITE}, Table V. phi dN/dy at mid-rapidity (|y| < 0.35 used) from '
                                          'exponential fits in mT. Errors: stat and sys.'}})
    for hf in ('ppg016fig14b.yaml', 'ppg016fig16.yaml'):
        t = load(hf)
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            what = 'phi/pi' if 'pi' in h else 'phi/K-'
            if hf == 'ppg016fig16.yaml':
                classes = ['0-92%']
            else:
                classes = [f"{float(v['low']):g}-{float(v['high']):g}%" for v in iv['values']]
            staged.append({'observable_type': 'particle_ratios', 'inspire_id': '661505', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{what} ({hf})',
                           'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> PHI X'},
                           'centrality': {'classes': classes, **BBC}, 'particle_id': {'species': 'phi'},
                           'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
                           'observable': what, 'value_type': h, 'numerator': 'phi',
                           'denominator': 'pi' if what == 'phi/pi' else 'K-',
                           'notes': {'value': f'{CITE}. {what} dN/dy ratio at mid-rapidity, the pion and kaon yields '
                                              'from INSPIRE 624474. Errors: stat and sys.'},
                           'staging_reason': 'particle ratio (dN/dy of phi over pi or K- of another paper): not computed'})
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '661505' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    for out, docs in ((OUT_SPEC, spectra), (OUT_YIELD, yields)):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
