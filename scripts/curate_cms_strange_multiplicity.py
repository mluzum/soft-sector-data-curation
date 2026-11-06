#!/usr/bin/env python3
"""Curate CMS K0_S, Lambda, Xi production vs multiplicity in pp, pPb, PbPb, INSPIRE 1464834.

CMS Collaboration, "Multiplicity and rapidity dependence of strange hadron
production in pp, pPb, and PbPb collisions at the LHC", PLB 768 (2017)
103, arXiv:1605.06699 (CMS-HIN-15-006).

From the paper:
- pp at 7 TeV, pPb at 5.02 TeV, PbPb at 2.76 TeV (peripheral, the pPb
  track reconstruction);
- classes in N_trk^offline: primary tracks with |eta| < 2.4 and pT > 0.4
  GeV, not corrected for efficiency (pp: 0-35, 35-60, 60-90, 90-110,
  110-130, >= 130; pPb, PbPb: 0-35, 35-60, 60-90, 90-120, 120-150,
  150-185, 185-220, 220-260); lower bounds inclusive;
- the efficiency of N_trk^offline: <N_trk^offline> / <N_trk^corrected>
  per class, flat in multiplicity -- pp 0.86 (Table 1: 12/14 ... 137/158),
  pPb 0.83 and PbPb 0.80 (CMS PLB 724 (2013) 213, Table 1: e.g. 132/159,
  232/280; 134/168, 239/299), fake tracks (1-2%) included;
- pPb: "the proton beam rapidity is taken to be positive", y_lab = y_cm +
  0.465; N_trk^offline counted in the laboratory frame;
- K0_S, Lambda + Lambdabar, Xi- + Xibar+ (sums) invariant yields at
  |y_cm| < 1; pPb K0_S and Lambda also in y_cm slices for three classes
  (Fig. 7: 0-35, 120-150, 220-260 top to bottom; HEPData tables 35-40 in
  that order); Lambda/2K0_S and Xi/Lambda vs pT;
- errors: stat and sys.

Records (public): the spectra per class (Fig. 2) and per pPb y_cm slice
(Fig. 7), classes in track count (centrality.multiplicity_bins with the
efficiencies above; the pPb estimator window |eta_lab| < 2.4 is -2.865 <
eta_cm < 1.935); Lambda/2K0_S and Xi/Lambda (Fig. 3) and the pPb
Lambda/2K0_S per y_cm slice (Fig. 8), linked to the spectra.  Staged
(--staged OUT): blast-wave fit parameters, <KE_T> (no computation for
<m_T> - m), and the classes' <N_trk^offline> (no uncertainties given).

Run from the repository root:  python scripts/curate_cms_strange_multiplicity.py --staged OUT.yaml
"""
import argparse
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1464834/HEPData-ins1464834-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'CMS Collaboration, PLB 768 (2017) 103, arXiv:1605.06699'
OUT_S = 'data/particle_spectra/cms_particle_spectra_strange_multiplicity_pp_pPb_PbPb_ins1464834.yaml'
OUT_R = 'data/observable_ratios/cms_observable_ratios_strange_multiplicity_pp_pPb_PbPb_ins1464834.yaml'
SHIFT = 0.465
SYSTEMS = {'P P': ('pp', 7000.0, 0.86, 'Table 1 of this paper: 12/14, 43/50, ..., 137/158'),
           'P PB': ('pPb', 5020.0, 0.83, 'CMS PLB 724 (2013) 213 Table 1: e.g. 45/54, 132/159, 232/280'),
           'PB PB': ('PbPb', 2760.0, 0.80, 'CMS PLB 724 (2013) 213 Table 1: e.g. 44/55, 134/168, 239/299')}
SPECIES = {'K0S': ('K0_S', 'K0S'), 'Lambda + anti-Lambda': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR'),
           'Lambda+anti-Lambda': ('Lambda + Lambdabar', 'LAMBDA LAMBDABAR'),
           'Xi+ + Xi-': ('Xi- + Xibar+', 'XI- XIBAR+')}
FIG7 = {'data35.yaml': '0-35', 'data36.yaml': '0-35', 'data37.yaml': '120-150', 'data38.yaml': '120-150',
        'data39.yaml': '220-260', 'data40.yaml': '220-260'}


def cls_label(s):
    lo, hi = re.fullmatch(r'\s*(\d+)\s+TO\s+(\d+|inf)\s*', s).groups()
    return f"{lo}-{1000 if hi == 'inf' else hi}"


def centrality(system, classes):
    key, energy, eff, src = SYSTEMS[system]
    eta = ([{'min': round(-2.4 - SHIFT, 3), 'max': round(2.4 - SHIFT, 3)}] if key == 'pPb'
           else [{'min': -2.4, 'max': 2.4}])
    return {'classes': list(classes),
            'estimator': {'name': 'Ntrk_offline', 'observable': 'Nch', 'detector': 'Tracker'},
            'eta': eta, 'pT': {'min': 0.4, 'max': 100.0},
            'multiplicity_bins': {'efficiency': eff,
                                  'notes': f'<N_trk^offline>/<N_trk^corrected> per class, flat in multiplicity ({src}).'},
            'notes': ('N_trk^offline: primary tracks |eta| < 2.4, pT > 0.4 GeV, not efficiency corrected; lower '
                      'bounds inclusive' + (' (the top class open, here to 1000)' if key == 'pp' else '')
                      + ('; counted in the laboratory frame, here -2.865 < eta_cm < 1.935.' if key == 'pPb' else '.'))}


def collision(system, prod):
    key, energy, *_ = SYSTEMS[system]
    cs = {'energy_gev': energy, 'reaction': f'{system} --> {prod} X'}
    if key == 'pPb':
        cs.update(frame='nn_cms', rapidity_shift=SHIFT, forward_beam='p')
    return cs


def errs(dv):
    labels = {tuple(e['label'] for e in v.get('errors', [])) for v in dv['values']}
    assert labels == {('stat', 'sys')}, labels
    return 'stat and sys'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    spectra, index, staged = [], {}, []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        n = int(re.fullmatch(r'data(\d+)\.yaml', hf).group(1))
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if not (1 <= n <= 18 or 35 <= n <= 40):
            continue
        bins = [[v['low'], v['high']] for v in t['independent_variables'][0]['values']]
        assert t['independent_variables'][0]['header']['name'] == 'PT', hf
        for i, dv in enumerate(t['dependent_variables']):
            assert dv['header']['name'] == '(1/Nev)*(1/(2*PI*PT))*D2(N)/DPT/DYRAP', (hf, dv['header'])
            q = {x['name']: x['value'] for x in dv['qualifiers']}
            system, prod = re.fullmatch(r'(P P|P PB|PB PB) ---?> (.+) X', q['RE']).groups()
            species, rprod = SPECIES[prod]
            if n <= 18:
                cls, y = cls_label(q['class']), (-1.0, 1.0)
            else:
                cls = FIG7[hf]
                y = tuple(float(x) for x in re.fullmatch(r'(-?[\d.]+) TO (-?[\d.]+)', q['class']).groups())
            key, energy, *_ = SYSTEMS[system]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1464834', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f"{'Fig. 2' if n <= 18 else 'Fig. 7'}: {species} invariant yield, {key} "
                                        f"N_trk^offline {cls}, {y[0]:g} < y_cm < {y[1]:g} ({hf} column {i})",
                 'collision_system': collision(system, rprod), 'centrality': centrality(system, [cls]),
                 'particle_id': {'species': species},
                 'kinematic_range': {'y': {'min': y[0], 'max': y[1]},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                 'notes': {'value': f'{CITE}. 1/(2 pi pT) d2N/(dpT dy) per event of {species} (sum of particle and '
                                    f'antiparticle), {y[0]:g} < y_cm < {y[1]:g}, {key} {energy / 1000:g} TeV, '
                                    f'N_trk^offline {cls}. Errors: {errs(dv)}.'}}
            if n >= 35:
                d['hepdata_corrections'] = {'centrality': "HEPData's 'class' qualifier holds the y_cm slice; the "
                                                          "N_trk^offline class is Fig. 7's (0-35, 120-150, 220-260 "
                                                          "for tables 35-36, 37-38, 39-40)"}
            index[(key, species, cls, y)] = len(spectra)
            spectra.append(d)
    # Fig. 7 order (0-35, 120-150, 220-260): yields rise with the class
    for sp in ('K0_S', 'Lambda + Lambdabar'):
        for y in ((-0.8, 0.0), (0.0, 0.8)):
            first = [float(yaml.safe_load(open(os.path.join(RAW, spectra[index[('pPb', sp, c, y)]]['hepdata_file'])))
                           ['dependent_variables'][spectra[index[('pPb', sp, c, y)]]['dv_index']]['values'][3]['value'])
                     for c in ('0-35', '120-150', '220-260')]
            assert first == sorted(first), (sp, y, first)

    ratios = []
    for s in sub:
        hf = s['data_file']
        n = int(re.fullmatch(r'data(\d+)\.yaml', hf).group(1))
        if not (19 <= n <= 24 or n == 41):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        bins = [[v['low'], v['high']] for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            q = {x['name']: x['value'] for x in dv['qualifiers']}
            system = re.match(r'(P P|P PB|PB PB) -->', q.get('RE(Q=LAM)')).group(1)
            key, energy, *_ = SYSTEMS[system]
            if n == 41:
                c, yy = q['NTRKOFFLINE, YRAP(C=CM)'].split(',')
                cls = cls_label(c)
                y = tuple(float(x) for x in re.fullmatch(r'\s*(-?[\d.]+) TO (-?[\d.]+)', yy).groups())
            else:
                cls, y = cls_label(q['NTRKOFFLINE']), (-1.0, 1.0)
            if name == 'SIG(Q=LAM)/2*SIG(Q=K0S)':
                num, den, scale, label = 'Lambda + Lambdabar', 'K0_S', 0.5, 'Lambda/2K0_S'
            else:
                assert name == 'SIG(Q=XI)/SIG(Q=LAM)', name
                num, den, scale, label = 'Xi- + Xibar+', 'Lambda + Lambdabar', 1.0, 'Xi/Lambda'
            sn, sd = spectra[index[(key, num, cls, y)]], spectra[index[(key, den, cls, y)]]
            lo = max(sn['kinematic_range']['pT']['min'], sd['kinematic_range']['pT']['min'])
            hi = min(sn['kinematic_range']['pT']['max'], sd['kinematic_range']['pT']['max'])
            assert bins[0][0] >= lo - 1e-9 and bins[-1][1] <= hi + 1e-9, (hf, i, bins[0], bins[-1], lo, hi)
            assert {tuple(e['label'] for e in v['errors']) for v in dv['values']} == {('stat', 'sys')}, (hf, i)
            d = {'observable_type': 'observable_ratios', 'inspire_id': '1464834', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f"{'Fig. 8' if n == 41 else 'Fig. 3'}: {label} vs pT, {key} N_trk^offline "
                                        f"{cls}, {y[0]:g} < y_cm < {y[1]:g} ({hf} column {i})",
                 'collision_system': dict(collision(system, 'X'), reaction=f'{system} --> X X'),
                 'centrality': centrality(system, [cls]),
                 'kinematic_range': {'y': {'min': y[0], 'max': y[1]},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': f'{label}, {key} {energy / 1000:g} TeV',
                 'ratio_type': 'particle_ratio', 'numerator': num, 'denominator': den,
                 'numerator_particle': num, 'denominator_particle': den,
                 'numerator_system': key, 'denominator_system': key,
                 'numerator_record': f'{OUT_S}#{index[(key, num, cls, y)]}',
                 'denominator_record': f'{OUT_S}#{index[(key, den, cls, y)]}',
                 'notes': {'value': f'{CITE}. {label} = ({num}) / ({"2 " if scale == 0.5 else ""}{den}) vs pT, '
                                    f'{y[0]:g} < y_cm < {y[1]:g}, {key}, N_trk^offline {cls}; linked to the spectra. '
                                    'Errors: stat and sys.'}}
            if scale != 1.0:
                d['scale'] = scale
            ratios.append(d)

    for s in sub:
        hf = s['data_file']
        n = int(re.fullmatch(r'data(\d+)\.yaml', hf).group(1))
        if not (25 <= n <= 34 or n in (42, 43)):
            continue
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            reason = ('blast-wave fit parameter: not computed' if n <= 27 else
                      '<N_trk^offline> per class, given without uncertainties' if name == 'MEAN(MULTIPLICITY)' else
                      '<KE_T> = <m_T> - m: no computation for it')
            staged.append({'observable_type': 'particle_yields', 'inspire_id': '1464834', 'version': VERSION,
                           'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{name} ({hf} column {i})", 'value_type': name,
                           'notes': {'value': f'{CITE}. {s["description"][:150]}'}, 'staging_reason': reason})
    for out, docs in ((OUT_S, spectra), (OUT_R, ratios)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
