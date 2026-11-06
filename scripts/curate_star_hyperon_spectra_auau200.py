#!/usr/bin/env python3
"""Curate STAR Lambda, Xi and Omega spectra in Au+Au at 200 GeV, INSPIRE 718755.

J. Adams et al. (STAR), "Scaling properties of hyperon production in Au+Au
collisions at sqrt(sNN) = 200 GeV", PRL 98 (2007) 062301,
arXiv:nucl-ex/0606014.

From the paper:
- Lambda (Lambdabar) at |y| < 1, Xi- (Xibar+) and Omega- + Omegabar+ at
  |y| < 0.75; centrality from the charged multiplicity in the TPC, |eta| <
  0.5: 0-5% (central trigger), 10-20, 20-40, 40-60, 60-80% (Omega: 0-5,
  20-40, 40-60%);
- Fig. 1: 1/(N_ev 2 pi) d2N/(pT dpT dy); the Lambda spectra corrected for
  the Xi, Xi0 and Omega feed-down (a ~15% contribution); the figure shows
  statistical errors (HEPData labels them 'total'); pT-dependent
  systematics not tabulated.

Records (public): the 23 spectra.  HEPData gives one pT per point, no
edges: the bins are contiguous with each point at its centre, the lowest
edge chosen so that the edges fall on a 0.1 (else 0.05, 0.025) GeV/c grid
and the widths vary smoothly -- Xi: 0.7, 1.0, 1.2, ..., 2.8, 3.1, 3.5, 4.0, 4.5, 5.0 (central classes)
and 0.7, 1.2, 1.6, 2.0, 2.4, 3.1, 4.0, 6.0 (60-80%), sharing their
edges; Omega: 1.0 + 0.625 k, last bin 3.5-4.5; Lambda: 0.6, 0.7, 0.8,
1.0, ..., 3.0, 3.35, 3.8, 4.4, 5.1.  Staged (--staged OUT): yields per
N_part relative to pp and gamma_s (Fig. 2), R_CP (Fig. 3).

Run from the repository root:  python scripts/curate_star_hyperon_spectra_auau200.py --staged OUT.yaml
"""
import argparse
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins718755/HEPData-ins718755-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 98 (2007) 062301, arXiv:nucl-ex/0606014'
OUT = 'data/particle_spectra/star_particle_spectra_hyperons_AuAu_200gev_ins718755.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'charged particle multiplicity in the TPC, |eta| < 0.5.'}
SPECIES = {'\\Lambda': ('Lambda', 'LAMBDA', 1.0), '\\overline{\\Lambda}': ('Lambdabar', 'LAMBDABAR', 1.0),
           '\\Xi^{-}': ('Xi-', 'XI-', 0.75), '\\Xi^{+}': ('Xibar+', 'XIBAR+', 0.75),
           'Omega': ('Omega- + Omegabar+', 'OMEGA- OMEGABAR+', 0.75)}


def centred_edges(pts):
    """Contiguous bins with each point at its centre; the lowest edge on the finest grid that fits."""
    def build(e0):
        edges = [e0]
        for c in pts:
            edges.append(round(2 * c - edges[-1], 6))
            if not edges[-2] < c < edges[-1]:
                return None
        return edges

    def score(edges):
        return sum(2 if abs(e * 10 - round(e * 10)) < 1e-6 else 1 if abs(e * 20 - round(e * 20)) < 1e-6 else
                   0.5 if abs(e * 40 - round(e * 40)) < 1e-6 else 0 for e in edges)

    def tv(edges):                               # total variation of the bin widths: smooth binning
        w = [b - a for a, b in zip(edges, edges[1:])]
        return sum(abs(b - a) for a, b in zip(w, w[1:]))
    width = pts[1] - pts[0]
    cands = [round(pts[0] - k * 0.0125, 6) for k in range(1, int(round(width / 0.0125)) + 1)]
    built = [(score(e) - 2 * tv(e), e) for e in map(build, cands) if e]
    best = max(built)
    assert all(abs(e * 40 - round(e * 40)) < 1e-6 for e in best[1]), (pts, best)
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    sub = list(yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))))[1:]
    public, staged = [], []
    for s in sub:
        hf, desc = s['data_file'], s['description']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        if not hf.startswith('figure_1_'):
            for i, dv in enumerate(t['dependent_variables']):
                staged.append({'observable_type': 'nuclear_modification' if hf == 'figure_3.yaml' else
                               'particle_yields', 'inspire_id': '718755', 'version': VERSION, 'hepdata_file': hf,
                               'dv_index': i, 'hepdata_table_label': f"{dv['header']['name']} ({hf} column {i})",
                               'value_type': dv['header']['name'], 'notes': {'value': f'{CITE}. {desc[:200]}'},
                               'staging_reason': ('R_CP: N_bin scaled' if hf == 'figure_3.yaml' else
                                                  'gamma_s: a fit parameter' if 'down' in hf else
                                                  'dN/dy per N_part relative to pp')})
            continue
        xs = [float(v['value']) for v in t['independent_variables'][0]['values']]
        runs, start = [], 0                      # the table stacks point sets, each ascending
        for r in range(1, len(xs) + 1):
            if r == len(xs) or xs[r] <= xs[r - 1]:
                runs.append(list(range(start, r)))
                start = r
        edges = {tuple(run): centred_edges([xs[r] for r in run]) for run in runs}
        for i, dv in enumerate(t['dependent_variables']):
            q = ' '.join(str(k['value']) for k in dv['qualifiers'])
            m = re.search(r'(\d+) - (\d+)%', q)
            cls = f'{m.group(1)}-{m.group(2)}%'
            key = 'Omega' if 'Omega' in q else re.match(r'\$(.+?)\$:', q).group(1)
            species, prod, ymax = SPECIES[key]
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            run = next(run for run in runs if rows[0] in run)
            assert set(rows) <= set(run) and rows == list(range(rows[0], rows[-1] + 1)), (hf, i, rows)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('total',)}
            e = edges[tuple(run)]
            bins = [[e[r - run[0]], e[r - run[0] + 1]] for r in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '718755', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f'Fig. 1: {species} invariant yield, {cls} ({hf} column {i})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': f'AU AU --> {prod} X'},
                 'centrality': {'classes': [cls], **REFMULT}, 'particle_id': {'species': species},
                 'kinematic_range': {'y': {'min': -ymax, 'max': ymax},
                                     'pT': {'min': bins[0][0], 'max': bins[-1][1], 'bin_ranges': bins}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield'}
            if rows != list(range(len(xs))):
                d['table_rows'] = rows
            feed = (' Corrected for the Xi, Xi0 and Omega feed-down.' if species.startswith('Lambda') else '')
            trig = ' (central-trigger sample)' if cls == '0-5%' else ''
            d['notes'] = {'value': f'{CITE}, Fig. 1. 1/(2 pi pT) d2N/(dpT dy) of {species}, |y| < {ymax:g}, Au+Au '
                                   f'200 GeV {cls}{trig}.{feed} pT bins contiguous with the tabulated points at '
                                   'their centres (edges on a 0.1, 0.05 or 0.025 GeV/c grid). Errors: statistical '
                                   "(HEPData's 'total'; pT-dependent systematics not tabulated)."}
            public.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records; {len(staged)} staged')
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
