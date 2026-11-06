#!/usr/bin/env python3
"""Curate STAR strange and multi-strange hadron production in Au+Au at 62.4 GeV, INSPIRE 871561.

M. M. Aggarwal et al. (STAR), "Strange and multi-strange particle
production in Au+Au collisions at sqrt(sNN) = 62.4 GeV", PRC 83 (2011)
024901, arXiv:1010.0142.

From the paper:
- K0_S, Lambda, Lambdabar, Xi-, Xibar+, Omega-, Omegabar+ from their weak
  decay topology, all at mid-rapidity |y| < 1.0;
- centrality from the uncorrected TPC charged multiplicity (as STAR PRC 79
  034909: |eta| < 0.5), classes 0-5, 5-10, 10-20, 20-30, 30-40, 40-60,
  60-80% for K0_S and Lambda; Xi in 0-5, 5-10, 10-20, 20-40, 40-60,
  60-80%; Omega in 0-20, 20-40, 40-60% (Fig. 3);
- Lambda and Lambdabar corrected for the feed-down of charged and neutral
  Xi decays (Xi0 taken equal to Xi+-), not of Omega; Sigma0 decays count
  as prompt;
- spectra 1/(2 pi pT) d2N/(dpT dy) per event at the mean pT of each bin
  (no edges), errors statistical only (Fig. 3).

Records (public): pT spectra of the seven species per class (Fig. 3;
HEPData stacks the Xi and Omega classes in row blocks: table_rows), at the
tabulated points with 0.1 GeV/c windows; Lambda/K0_S vs pT in 0-5, 40-60,
60-80% (Fig. 16, linked to the Lambda and K0_S spectra, total errors; its
4.5 GeV/c point, beyond the Lambda spectra, left out).
Staged (--staged OUT): <pT> vs dNch/dy (Fig. 4: HEPData's K0_S, Lambda,
Xi values differ from the paper's Table IV, e.g. K0_S 0-5% 0.701 +-
0.008 vs 0.684 +- 0.030, Lambda 0.942 vs 0.901, Xi- 1.04 vs 1.097, and
its dNch/dy from Table I; pi, K-, pbar are STAR PRC 79 034909's);
correction factors (Fig. 2); K0_S vs K+- dN/dpT (Fig. 5, the K+- from PRC
79 034909); energy dependences and antibaryon/baryon ratios (Figs. 6-8,
13); thermal fits (Figs. 9-11); ratios to pi- relative to peripheral
(Fig. 12); R_CP (Figs. 14, 15); the maximum of Lambda/K0_S (Fig. 17).
The integrated dN/dy and <pT> per class (Tables IV-VI) are not in HEPData.

Run from the repository root:
  python scripts/curate_star_strange_auau62.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins871561/HEPData-ins871561-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'M. M. Aggarwal et al. (STAR), PRC 83 (2011) 024901, arXiv:1010.0142'
OUT_S = 'data/particle_spectra/star_particle_spectra_strange_AuAu_62.4gev_ins871561.yaml'
OUT_R = 'data/observable_ratios/star_observable_ratios_lambda_k0s_AuAu_62.4gev_ins871561.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}],
           'notes': 'uncorrected TPC charged multiplicity (STAR PRC 79 034909: |eta| < 0.5).'}
WINDOW = 0.1
FEED = {'Lambda': ' Corrected for the Xi- and Xi0 weak-decay feed-down (Xi0 taken equal to Xi+-; not Omega); '
                  'Sigma0 decays count as prompt.',
        'Lambdabar': ' Corrected for the Xibar weak-decay feed-down (Table V); Sigma0bar decays count as prompt.'}
SPECTRA = {'figure_3_a_Kshort.yaml': ('K0_S', 'K0S'), 'figure_3_b_lambda.yaml': ('Lambda', 'LAMBDA'),
           'figure_3_c_anti_lambda.yaml': ('Lambdabar', 'LAMBDABAR'), 'figure_3_d_Xi_minus.yaml': ('Xi-', 'XI-'),
           'figure_3_e_anti-Xi_plus.yaml': ('Xibar+', 'XIBAR+'), 'figure_3_f_Omega_minus.yaml': ('Omega-', 'OMEGA-'),
           'figure_3_g_Anti-Omega_plus.yaml': ('Omegabar+', 'OMEGABAR+')}
STAGED = {
    'figure_2.yaml': 'detector correction factors, not a measurement',
    'figure_4.yaml': ('<pT> vs dNch/dy: HEPData\'s K0_S, Lambda, Xi values differ from the paper\'s Table IV (K0_S '
                      '0-5% 0.701 +- 0.008 vs 0.684 +- 0.030, Lambda 0.942 vs 0.901, Xi- 1.04 vs 1.097; dNch/dy '
                      '588 vs Table I 582); pi-, K-, pbar from STAR PRC 79 034909: unresolved'),
    'figure_5_left.yaml': 'K0_S dN/dpT compared with K+- of STAR PRC 79 034909 (the K0_S spectra are curated from Fig. 3)',
    'figure_5_right.yaml': 'K0_S dN/dpT compared with K+- of STAR PRC 79 034909 (the K0_S spectra are curated from Fig. 3)',
    'figure_6_up.yaml': 'central yields vs beam energy (compilation; 62.4 GeV 0-5% only here, not Table V\'s values)',
    'figure_6_down.yaml': 'central yields vs beam energy (compilation; 62.4 GeV 0-5% only here)',
    'figure_7.yaml': 'antibaryon/baryon ratios vs beam energy (compilation)',
    'figure_8.yaml': 'antibaryon/baryon ratios vs dNch/dy (ratios of integrated yields not in HEPData)',
    'figure_9.yaml': 'particle ratios with statistical-model fits, 0-5%',
    'figure_10_a.yaml': 'statistical-model T_ch vs N_part (a fit parameter)',
    'figure_10_b.yaml': 'statistical-model gamma_s vs N_part (a fit parameter)',
    'figure_11_up.yaml': 'statistical-model T vs beam energy (fit parameters, compilation)',
    'figure_11_down.yaml': 'statistical-model mu_B vs beam energy (fit parameters, compilation)',
    'figure_12_left_up.yaml': 'baryon/pi- yield ratios relative to peripheral vs dNch/dy (pi- from PRC 79 034909)',
    'figure_12_left_down.yaml': 'antibaryon/pi- yield ratios relative to peripheral vs dNch/dy (pi- from PRC 79 034909)',
    'figure_12_right_up.yaml': 'baryon/pi- yield ratios relative to peripheral, 200 GeV (another paper)',
    'figure_12_right_down.yaml': 'antibaryon/pi- yield ratios relative to peripheral, 200 GeV (another paper)',
    'figure_13.yaml': 'baryon/pi- ratios vs beam energy (compilation)',
    'figure_14.yaml': 'R_CP (0-10%/40-80%, N_bin-scaled) of pi, K0_S, Lambda, Xi: not curated (high-pT emphasis)',
    'figure_15.yaml': 'R_CP (0-5%/40-60%, N_bin-scaled) of Lambda, Xi: not curated (high-pT emphasis)',
    'figure_17.yaml': 'maximum of Lambda/K0_S vs N_part (a derived quantity)'}


def rows_of(dv):
    return [i for i, v in enumerate(dv['values']) if v['value'] not in ('-', '', None)]


def cls_of(dv):
    q = {x['name']: x['value'] for x in dv['qualifiers']}
    return q['Centrality'].replace(' ', '')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, index = [], {}
    for hf, (species, prod) in SPECTRA.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        assert t['independent_variables'][0]['header']['name'] == 'Transverse Momentum', hf
        x = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            assert dv['header']['name'].startswith('(1/(2*Pi*pT))*d^2N/dydpT'), (hf, dv['header'])
            rows = rows_of(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat',)}, (hf, i)
            pts = [x[r] for r in rows]
            assert pts == sorted(pts) and min(b - a for a, b in zip(pts, pts[1:])) > WINDOW, (hf, i, pts)
            cls = cls_of(dv)
            kr = {'y': {'min': -1.0, 'max': 1.0},
                  'pT': {'min': round(pts[0] - WINDOW / 2, 6), 'max': round(pts[-1] + WINDOW / 2, 6),
                         'points': pts, 'point_window': WINDOW}}
            d = {'observable_type': 'particle_spectra', 'inspire_id': '871561', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'Fig. 3: invariant yield of {species}, Au+Au 62.4 GeV, {cls} ({hf})',
                 'collision_system': {'energy_gev': 62.4, 'reaction': f'AU AU --> {prod} X'},
                 'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': species},
                 'kinematic_range': kr, 'observable': 'invariant_yield', 'value_type': 'invariant_yield'}
            if rows != list(range(len(x))):
                d['table_rows'] = rows
            d['notes'] = {'value': f'{CITE}. 1/(2 pi pT) d2N/(dpT dy) of {species} per event, |y| < 1, Au+Au 62.4 '
                                   f'GeV {cls}; weak-decay topology.{FEED.get(species, "")} Points at the mean pT '
                                   f'of each bin (no edges), computed in {WINDOW} GeV/c windows. Errors: '
                                   'statistical only (Fig. 3).'}
            index[(species, cls)] = len(spectra)
            spectra.append(d)
    ratios = []
    hf = 'figure_16.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    x = [float(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        assert dv['header']['name'] == 'Lambda/K0 short', dv['header']
        rows = rows_of(dv)
        assert rows == list(range(len(rows))), rows
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('Total Uncertainty',)}
        cls = cls_of(dv)
        # points whose window lies inside both spectra (the ratio's last point, 4.5 GeV/c in 0-5 and
        # 40-60%, combines bins beyond the Lambda spectrum's 4.29 GeV/c)
        sides = [spectra[index[(sp, cls)]]['kinematic_range']['pT'] for sp in ('Lambda', 'K0_S')]
        lo, hi = max(a['min'] for a in sides), min(a['max'] for a in sides)
        kept = [r for r in rows if x[r] - WINDOW / 2 >= lo - 1e-9 and x[r] + WINDOW / 2 <= hi + 1e-9]
        dropped = [x[r] for r in rows if r not in kept]
        assert kept == rows[:len(kept)] and len(kept) >= len(rows) - 1, (cls, dropped)
        rows = kept
        pts = [x[r] for r in rows]
        d = {'observable_type': 'observable_ratios', 'inspire_id': '871561', 'version': VERSION, 'hepdata_file': hf,
             'dv_index': i, 'hepdata_table_label': f'Fig. 16: Lambda/K0_S vs pT, Au+Au 62.4 GeV, {cls} ({hf})',
             'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> X X'},
             'centrality': {'classes': [cls], **REFMULT},
             'kinematic_range': {'y': {'min': -1.0, 'max': 1.0},
                                 'pT': {'min': round(pts[0] - WINDOW / 2, 6), 'max': round(pts[-1] + WINDOW / 2, 6),
                                        'points': pts, 'point_window': WINDOW}},
             'observable': 'invariant_yield', 'value_type': 'Lambda/K0S, Au+Au 62.4 GeV',
             'ratio_type': 'particle_ratio', 'numerator': 'Lambda', 'denominator': 'K0_S',
             'numerator_particle': 'Lambda', 'denominator_particle': 'K0_S',
             'numerator_system': 'Au+Au', 'denominator_system': 'Au+Au',
             'numerator_record': f"{OUT_S}#{index[('Lambda', cls)]}",
             'denominator_record': f"{OUT_S}#{index[('K0_S', cls)]}",
             'notes': {'value': f'{CITE}. Lambda/K0_S vs pT, |y| < 1, {cls}; ratio of the Fig. 3 spectra (Lambda '
                                'feed-down corrected), points at the tabulated pT in 0.1 GeV/c windows'
                                + (f'; the point at {dropped[0]:g} GeV/c (beyond the Lambda spectrum) not included'
                                   if dropped else '') + '. Errors: total.'}}
        if len(rows) < len(x):
            d['table_rows'] = rows
        ratios.append(d)
    staged = []
    for hf, reason in STAGED.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        for i, dv in enumerate(t['dependent_variables']):
            q = {k['name']: k['value'] for k in dv.get('qualifiers', [])}
            staged.append({'observable_type': 'mean_pT' if hf == 'figure_4.yaml' else 'particle_yields',
                           'inspire_id': '871561', 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                           'hepdata_table_label': f"{dv['header']['name']} {q} ({hf} column {i})",
                           'collision_system': {'energy_gev': 62.4, 'reaction': 'AU AU --> X'},
                           'value_type': dv['header']['name'],
                           'notes': {'value': f'{CITE}. {hf}: {dv["header"]["name"]} {q}.'},
                           'staging_reason': reason})
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
