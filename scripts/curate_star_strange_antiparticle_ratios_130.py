#!/usr/bin/env python3
"""Curate STAR strange antiparticle/particle ratios in central Au+Au at 130 GeV, INSPIRE 602867.

J. Adams et al. (STAR), "Strange anti-particle to particle ratios at
mid-rapidity in sqrt(sNN) = 130 GeV Au+Au collisions", PLB 567 (2003)
167, arXiv:nucl-ex/0211024.

From the paper:
- the most central 11% of the hadronic cross section (a CTB trigger and a
  cut on the observed TPC track multiplicity);
- Lambdabar/Lambda at |y| < 0.5, pT > 0.4 GeV/c: 0.74 observed, 0.71
  after the feed-down correction (Xi and Omega decays); Xibar+/Xi- at |y| <
  0.5, pT > 0.5 GeV/c: 0.83; Omegabar+/Omega- at |y| < 1: 0.95; K+/K-
  from dE/dx (|y| < 0.4, 0.15 < pT < 0.6 GeV/c) and from kinks (to 2
  GeV/c), corrected for phi decays and secondary interactions;
- Fig. 2: the ratios vs pT; Fig. 3: the integrated anti-baryon/baryon
  ratios with STAR's pbar/p of another paper; errors stat and sys.

Records (public): Lambdabar/Lambda vs pT (|y| < 0.5, 0.2 GeV/c bins
around the points, 0.4-2.6), K+/K- vs pT from dE/dx (|y| < 0.4, 0.05 bins,
0.15-0.6) and from kinks (|y| < 0.5 assumed, 0.2 bins, 0.2-2.0), and the
integrated Lambdabar/Lambda, Xibar+/Xi-, Omegabar+/Omega- (Fig. 3; the
Omega pT acceptance is not stated: all pT); particle_ratio records linked
to side records without values (experimental_values: false).  Staged
(--staged OUT): Xibar+/Xi- vs pT (tabulated ~0.40-0.45 against the quoted
0.83: shifted for display?), the pbar/p point of another STAR paper, and
the invariant-mass distributions.

Run from the repository root:  python scripts/curate_star_strange_antiparticle_ratios_130.py --staged OUT.yaml
"""
import argparse
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins602867/HEPData-ins602867-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PLB 567 (2003) 167, arXiv:nucl-ex/0211024'
OUT_R = 'data/observable_ratios/star_observable_ratios_strange_antiparticle_AuAu_130gev_ins602867.yaml'
OUT_S = 'data/particle_spectra/star_particle_spectra_strange_ratio_sides_AuAu_130gev_ins602867.yaml'
OUT_Y = 'data/particle_yields/star_particle_yields_strange_ratio_sides_AuAu_130gev_ins602867.yaml'
NCH = {'estimator': {'name': 'Nch', 'observable': 'Nch', 'detector': 'TPC'},
       'eta': [{'min': -0.75, 'max': 0.75}],
       'notes': "the most central 11%: a CTB trigger and a cut on the observed TPC track multiplicity (STAR's "
                '130 GeV classes, |eta| < 0.75).'}
CLS = ['0-11%']
PROD = {'Lambdabar': 'LAMBDABAR', 'Lambda': 'LAMBDA', 'K+': 'K+', 'K-': 'K-', 'Xibar+': 'XIBAR+', 'Xi-': 'XI-',
        'Omegabar+': 'OMEGABAR+', 'Omega-': 'OMEGA-'}
# Fig. 2 columns: (numerator, denominator, |y| max, bin width, note)
FIG2 = {'Anti-$\\Lambda/\\Lambda$': ('Lambdabar', 'Lambda', 0.5, 0.2, 'TPC V0 reconstruction'),
        'dE/dx K+/K-': ('K+', 'K-', 0.4, 0.05, 'TPC dE/dx'),
        'Kink K+/K-': ('K+', 'K-', 0.5, 0.2, 'kink topology; the rapidity window is not stated (|y| < 0.5 here)')}
# Fig. 3 rows: (numerator, denominator, |y| max, pT min, feed-down note)
FIG3 = {'Anti-$\\Lambda/\\Lambda$': ('Lambdabar', 'Lambda', 0.5, 0.4, ' Corrected for the Xi and Omega feed-down.'),
        'Anti-$\\Xi+/\\Xi-$': ('Xibar+', 'Xi-', 0.5, 0.5, ''),
        'Anti-$\\Omega/\\Omega$': ('Omegabar+', 'Omega-', 1.0, 0.0, ' The pT acceptance is not stated (all pT here).')}


def stage(hf, i, label, reason):
    return {'observable_type': 'observable_ratios', 'inspire_id': '602867', 'version': VERSION, 'hepdata_file': hf,
            'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})', 'value_type': label,
            'notes': {'value': f'{CITE}.'}, 'staging_reason': reason}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    spectra, yields, ratios, staged = [], [], [], []

    def side(species, kr, integrated, label):
        docs, out = (yields, OUT_Y) if integrated else (spectra, OUT_S)
        docs.append({'observable_type': 'particle_yields' if integrated else 'particle_spectra',
                     'inspire_id': '602867', 'version': VERSION, 'experimental_values': False,
                     'hepdata_table_label': f'{species} {label} (a ratio side; no values published)',
                     'collision_system': {'energy_gev': 130.0, 'reaction': f'AU AU --> {PROD[species]} X'},
                     'centrality': {'classes': list(CLS), **NCH}, 'particle_id': {'species': species},
                     'kinematic_range': kr, 'observable': 'dN_dy' if integrated else 'invariant_yield',
                     'value_type': 'dN_dy' if integrated else 'invariant_yield',
                     'notes': {'value': f'{CITE}. {species} {label}: the side of a published antiparticle/particle '
                                        'ratio (the ratio record holds the values).'}})
        return f'{out}#{len(docs) - 1}'

    def ratio(hf, i, num, den, kr, label, note, dim=None):
        n_ref = side(num, kr, dim is None, label)
        d_ref = side(den, kr, dim is None, label)
        d = {'observable_type': 'observable_ratios', 'inspire_id': '602867', 'version': VERSION,
             'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{num}/{den} {label} ({hf} column {i})',
             'collision_system': {'energy_gev': 130.0, 'reaction': 'AU AU --> X X'},
             'centrality': {'classes': list(CLS), **NCH}, 'kinematic_range': kr,
             'observable': 'dN_dy' if dim is None else 'invariant_yield', 'value_type': f'{num}/{den}',
             'ratio_type': 'particle_ratio', 'numerator': num, 'denominator': den, 'numerator_particle': num,
             'denominator_particle': den, 'numerator_system': 'Au+Au', 'denominator_system': 'Au+Au',
             'numerator_record': n_ref, 'denominator_record': d_ref, 'notes': {'value': note}}
        return d

    hf = 'StrangeAntiP2PFig2.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    xs = [float(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
        if name not in FIG2:
            staged.append(stage(hf, i, name, 'Xibar+/Xi- vs pT tabulated at ~0.40-0.45 against the quoted 0.83 '
                                             '(shifted for display?)'))
            continue
        num, den, ymax, w, how = FIG2[name]
        pts = [xs[r] for r in rows]
        assert all(abs(b - a - w) < 1e-6 for a, b in zip(pts, pts[1:])), (name, pts)
        assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('Statistical', 'Systematic')}
        bins = [[round(p - w / 2, 4), round(p + w / 2, 4)] for p in pts]
        kr = {'y': {'min': -ymax, 'max': ymax}, 'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}}
        d = ratio(hf, i, num, den, kr, f'vs pT, |y| < {ymax:g}, Au+Au 130 GeV 0-11%',
                  f'{CITE}, Fig. 2. {num}/{den} vs pT ({how}), |y| < {ymax:g}, central (0-11%) Au+Au 130 GeV; pT '
                  f'bins of {w:g} GeV/c around the points. Errors: stat and sys.', dim='differential_pT')
        d['table_rows'] = rows
        ratios.append(d)
    hf = 'StrangeAntiP2PFig3.yaml'
    t = yaml.safe_load(open(os.path.join(RAW, hf)))
    for i, dv in enumerate(t['dependent_variables']):
        name = dv['header']['name']
        rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
        if name not in FIG3:
            staged.append(stage(hf, i, name, "the pbar/p point of another STAR paper (Fig. 3's comparison)"))
            continue
        num, den, ymax, ptmin, feed = FIG3[name]
        assert len(rows) == 1
        kr = {'y': {'min': -ymax, 'max': ymax}, 'pT': {'min': ptmin, 'max': 100.0}}
        cut = f', pT > {ptmin:g} GeV/c' if ptmin else ''
        d = ratio(hf, i, num, den, kr, f'|y| < {ymax:g}{cut}, Au+Au 130 GeV 0-11%',
                  f'{CITE}, Fig. 3. {num}/{den} at |y| < {ymax:g}{cut}, central (0-11%) Au+Au 130 GeV.{feed} '
                  'Errors: stat and sys.')
        d['table_rows'] = rows
        d['hepdata_corrections'] = {'centrality': "the table's rows are the four ratios; the class is the paper's "
                                                  'central 11%'}
        ratios.append(d)
    hf = 'StrangeAntiP2PFig1'
    for f in sorted(x for x in os.listdir(RAW) if x.startswith(hf)):
        t = yaml.safe_load(open(os.path.join(RAW, f)))
        for i, dv in enumerate(t['dependent_variables']):
            staged.append(stage(f, i, dv['header']['name'], 'invariant-mass distribution (detector level)'))
    for out, docs in ((OUT_S, spectra), (OUT_Y, yields), (OUT_R, ratios)):
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
