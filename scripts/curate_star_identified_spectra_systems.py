#!/usr/bin/env python3
"""Curate STAR identified spectra in d+Au and Au+Au, INSPIRE 793126.

B. I. Abelev et al. (STAR), "Systematic measurements of identified particle
spectra in pp, d+Au and Au+Au collisions from STAR", PRC 79, 034909 (2009),
arXiv:0808.2041.

From the paper (Sec. II C, Table II, figure captions):
- pi+-, K+-, p, pbar at |y| < 0.1 (TPC dE/dx); p and pbar inclusive (weak
  decay products included);
- Au+Au centrality from the uncorrected charged multiplicity in the TPC,
  |eta| < 0.5 (primary tracks with >= 10 fit points); classes 0-5 ... 70-80%
  (62.4, 200 GeV), 0-6 ... 58-85% (130 GeV);
- d+Au centrality from the charged multiplicity in the East (Au-going) FTPC,
  -3.8 < eta < -2.8, pT < 3 GeV/c: 0-20, 20-40, 40-100% and minimum bias;
- spectra errors: statistical and point-to-point systematic in quadrature
  (labelled 'stat' in HEPData); <pT> (Table VII, fits extrapolated to all
  pT) errors: statistical and systematic in quadrature.

Records (public): <pT> of pi+, pi-, K+, K-, p, pbar vs centrality in Au+Au
200 GeV (Table VII: only the 200 GeV rows are in HEPData); invariant yields
of pi+-, K+-, p, pbar in d+Au 200 GeV (Fig. 18, three classes) and Au+Au
62.4 GeV (Fig. 19, nine classes), of pi+- in Au+Au 130 GeV (Fig. 20, eight
classes); pT points with edges at the midpoints.  d+Au: lab frame
(equal beam energies per nucleon), the deuteron toward +eta.  Staged
(--staged OUT): the d+Au minimum-bias spectra (no class).  Not curated: the
detector figures (energy loss, efficiencies, vertexing, dca, backgrounds,
TOF comparisons; Figs. 10-17), the uncorrected multiplicity distributions
(Figs. 1-3), Glauber quantities (overlap area, dsigma/db; Figs. 4, 42), the
Bjorken energy density, K/pi ratios (Figs. 30, 32, 33: compilations and
yield ratios), freeze-out fit parameters (Figs. 38-41).  The 3 old public
records read the pi+ column for pi, K and p alike.

Run from the repository root:
    python scripts/curate_star_identified_spectra_systems.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins793126/HEPData-ins793126-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. I. Abelev et al. (STAR), PRC 79, 034909 (2009), arXiv:0808.2041'
TPC = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}],
       'notes': 'Uncorrected charged multiplicity in the TPC, |eta| < 0.5 (primary tracks, >= 10 fit points).'}
FTPC = {'estimator': {'name': 'FTPC-East', 'observable': 'Nch', 'detector': 'FTPC'},
        'eta': [{'min': -3.8, 'max': -2.8}], 'pT': {'min': 0.0, 'max': 3.0},
        'notes': 'Charged multiplicity in the East (Au-going) FTPC, -3.8 < eta < -2.8, pT < 3 GeV/c.'}
Y = {'min': -0.1, 'max': 0.1}
SPECIES = {'pi^-': 'pi-', 'pi^+': 'pi+', 'K^-': 'K-', 'K^+': 'K+', 'proton': 'p', 'pbar': 'pbar',
           'pi-': 'pi-', 'pi+': 'pi+', 'K-': 'K-', 'K+': 'K+', 'p': 'p'}
FIGS = {'Fig18a_dAu_spectra_pions.yaml': ('D AU', 200.0), 'Fig18b_dAu_spectra_kaons.yaml': ('D AU', 200.0),
        'Fig18c_dAu_spectra_protons.yaml': ('D AU', 200.0),
        'Fig19a_AuAu62_spectra_pions.yaml': ('AU AU', 62.4), 'Fig19b_AuAu62_spectra_kaons.yaml': ('AU AU', 62.4),
        'Fig19c_AuAu62_spectra_antiprotons.yaml': ('AU AU', 62.4),
        'Fig19d_AuAu62_spectra_protons.yaml': ('AU AU', 62.4), 'Fig20_AuAu130_spectra.yaml': ('AU AU', 130.0)}
OLD = 'data/mean_pT/star_mean_pT_systematic_measurements_of_identified_particle_200gev.yaml'


def system(beams, energy):
    cs = {'energy_gev': energy, 'reaction': f'{beams} --> X'}
    if beams == 'D AU':
        cs.update({'frame': 'lab', 'rapidity_shift': 0.0, 'forward_beam': 'd'})
    return cs


def edges(pts):
    mids = [round(0.5 * (a + b), 4) for a, b in zip(pts[:-1], pts[1:])]
    return [round(pts[0] - (mids[0] - pts[0]), 4)] + mids + [round(pts[-1] + (pts[-1] - mids[-1]), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out, staged = collections.defaultdict(list), []

    t = yaml.safe_load(open(os.path.join(RAW, 'TableVII.yaml')))
    classes = [str(v['value']) for v in t['independent_variables'][0]['values']]
    for i, dv in enumerate(t['dependent_variables']):
        sp = SPECIES[str(dv['qualifiers'][0]['value'])]
        out['mean_pT'].append({
            'observable_type': 'mean_pT', 'inspire_id': '793126', 'version': VERSION, 'hepdata_file': 'TableVII.yaml',
            'dv_index': i, 'hepdata_table_label': f'<pT> of {sp} vs centrality, Au+Au 200 GeV (TableVII.yaml)',
            'collision_system': system('AU AU', 200.0), 'centrality': {'classes': classes, **TPC},
            'particle_id': {'species': sp},
            'kinematic_range': {'y': dict(Y), 'pT': {'min': 0.0, 'max': 100.0}},
            'observable': 'mean_pT', 'value_type': 'mean_pT',
            'notes': {'value': f'{CITE}, Table VII. <pT> of {sp} at |y| < 0.1, Au+Au 200 GeV, from fits extrapolated '
                               'to all pT' + ('; inclusive (weak decay products included)' if sp in ('p', 'pbar') else '')
                               + '. Errors: statistical and systematic in quadrature.'}})

    for hf, (beams, energy) in FIGS.items():
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        pts = [float(v['value']) for v in t['independent_variables'][0]['values']]
        e = edges(pts)
        for i, dv in enumerate(t['dependent_variables']):
            lab = str(dv['qualifiers'][0]['value'])
            m = re.fullmatch(r'\$\\?(.+?)\$ \((.+)\)', lab)
            sp, cls = SPECIES[m.group(1)], m.group(2)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[e[j], e[j + 1]] for j in rows]
            mb = cls == 'min-bias'
            d = {'observable_type': 'particle_spectra', 'inspire_id': '793126', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f'invariant yield of {sp}, {cls}, {beams.title()} {energy:g} GeV ({hf})',
                 'collision_system': system(beams, energy),
                 'centrality': {'classes': ['MB' if mb else cls], **(FTPC if beams == 'D AU' else TPC)},
                 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f'{CITE}. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls}, {beams.title()} {energy:g} GeV, '
                                    '|y| < 0.1' + ('; inclusive (weak decay products included)' if sp in ('p', 'pbar') else '')
                                    + ('; lab frame, the deuteron toward +eta' if beams == 'D AU' else '')
                                    + '; pT points, edges at the midpoints. Errors: statistical and point-to-point '
                                      'systematic in quadrature (labelled stat).'}}
            if rows != list(range(len(pts))):
                d['table_rows'] = rows
            if mb:
                d['staging_reason'] = 'minimum bias: no centrality class of the model events'
                staged.append(d)
            else:
                out['particle_spectra'].append(d)
    if os.path.exists(OLD):
        assert all(str(x['inspire_id']) == '793126' for x in yaml.safe_load_all(open(OLD)) if x), OLD
        os.remove(OLD)
    for otype, docs in out.items():
        path = f'data/{otype}/star_{otype}_identified_dAu_AuAu_ins793126.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records;', dict(collections.Counter(
            (d['collision_system']['reaction'].split(' -')[0], d['collision_system']['energy_gev']) for d in docs)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
