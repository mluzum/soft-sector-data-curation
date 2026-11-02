#!/usr/bin/env python3
"""Curate STAR identified spectra in Au+Au at 200 GeV, INSPIRE 630160.

J. Adams et al. (STAR), "Identified particle distributions in pp and
Au+Au collisions at sqrt(sNN) = 200 GeV", PRL 92, 112301 (2004),
nucl-ex/0310004.

From the paper and tables:
- pi+-, K+-, p, pbar at |y| < 0.1 (TPC dE/dx); p and pbar inclusive (weak
  decay products included);
- Au+Au centrality from the charged multiplicity in the TPC, |eta| < 0.5;
  classes 0-5 ... 70-80%;
- invariant yields vs mT - m (points); errors: stat and sys in quadrature.

Records (public): invariant yields of pi+-, K+-, p, pbar in nine Au+Au
classes (Fig. 1), the mT - m points converted to pT = sqrt(x^2 + 2 x m),
the bin edges at the midpoints in mT - m (converted likewise).  Staged
(--staged OUT): the pp spectra (no centrality class); fiducial dN/dy and
<pT> vs y in 0-5% and 70-80% (Fig. 2: rapidity-differential yields of an
identified species are not computed; pi- tabulated / 5).  Not curated:
<pT> vs dNch/deta (Fig. 3a: the same values as INSPIRE 793126 Table VII),
K-/pi- and pbar/pi- vs dNch/deta (Fig. 3b: ratios of the curated yields),
the K-/pi- compilation (Fig. 3c: other experiments) and freeze-out fit
parameters (Fig. 4).  The old record read the 70-80% Fig. 2 column as dN/dy
of protons.

Run from the repository root:
    python scripts/curate_star_identified_spectra_200_prl.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import math
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins630160/HEPData-ins630160-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adams et al. (STAR), PRL 92, 112301 (2004), nucl-ex/0310004'
TPC = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'}, 'eta': [{'min': -0.5, 'max': 0.5}],
       'notes': 'Charged multiplicity in the TPC, |eta| < 0.5.'}
Y = {'min': -0.1, 'max': 0.1}
MASS = {'pi': 0.13957, 'K': 0.493677, 'p': 0.938272}
SPECIES = {'pi^{+}': 'pi+', 'pi^{-}': 'pi-', 'K^{+}': 'K+', 'K^{-}': 'K-', 'p': 'p', 'bar{p}': 'pbar',
           'pi^-': 'pi-', 'K^-': 'K-'}
OLD = 'data/particle_yields/star_particle_yields_identified_particle_distributions_200gev.yaml'
OUT = 'data/particle_spectra/star_particle_spectra_identified_AuAu_200gev_ins630160.yaml'
BLANK = ('', '-', '--', 'none')


def edges_x(x):
    mids = [0.5 * (a + b) for a, b in zip(x, x[1:])]
    return [max(0.0, x[0] - (mids[0] - x[0]))] + mids + [x[-1] + (x[-1] - mids[-1])]


def to_pt(x, m):
    return round(math.sqrt(x * x + 2.0 * x * m), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for hf, mkey in (('fig1-1.yaml', 'pi'), ('fig1-2.yaml', 'K'), ('fig1-3.yaml', 'p')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        x = [float(v['value']) for v in t['independent_variables'][0]['values']]
        ex = edges_x(x)
        for i, dv in enumerate(t['dependent_variables']):
            lab = str(dv['qualifiers'][0]['value']).replace('$', '').replace('\\', '')
            m = re.fullmatch(r'(.+?),\s*(.+)', lab)
            sp = SPECIES[m.group(1).strip()]
            rest = m.group(2).strip()
            pp = rest.lower().startswith('p') and 'au' not in rest.lower()
            cls = None if pp else re.search(r'(\d+)-(\d+)%', rest).group(0)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip().lower() not in BLANK]
            pairs = [[to_pt(ex[j], MASS[mkey]), to_pt(ex[j + 1], MASS[mkey])] for j in rows]
            d = {'observable_type': 'particle_spectra', 'inspire_id': '630160', 'version': VERSION, 'hepdata_file': hf,
                 'dv_index': i, 'hepdata_table_label': f"invariant yield of {sp}, {cls or 'pp'} ({hf})",
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'P P --> X' if pp else 'AU AU --> X'},
                 'particle_id': {'species': sp},
                 'kinematic_range': {'y': dict(Y), 'pT': {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs}},
                 'observable': 'invariant_yield', 'value_type': 'invariant_yield', 'dimensionality': 'differential_pT',
                 'notes': {'value': f"{CITE}, Fig. 1. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls or 'pp'}, |y| < 0.1"
                                    + ('; inclusive (weak decay products included)' if sp in ('p', 'pbar') else '')
                                    + '; tabulated vs mT - m, converted to pT, bin edges at the midpoints. Errors: '
                                      'stat and sys in quadrature.'}}
            if not pp:
                d['centrality'] = {'classes': [cls], **TPC}
            if rows != list(range(len(x))):
                d['table_rows'] = rows
            if pp:
                d['staging_reason'] = 'pp: no centrality class'
                staged.append(d)
            else:
                public.append(d)
    # Fig. 2: fiducial dN/dy and <pT> vs y (staged)
    for hf in ('fig2-1.yaml', 'fig2-2.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        yv = [float(v['value']) for v in t['independent_variables'][0]['values']]
        ybins = [[round(v - 0.05, 3), round(v + 0.05, 3)] for v in yv]
        for i, dv in enumerate(t['dependent_variables']):
            h = dv['header']['name']
            lab = str(dv['qualifiers'][0]['value']).replace('$', '').replace('\\', '')
            m = re.fullmatch(r'(\S+?)(/5)?\s+(\d+-\d+%)\s+Au\+Au', lab)
            sp, scaled, cls = SPECIES[m.group(1)], bool(m.group(2)), m.group(3)
            ptw = re.search(r'([\d.]+)<p_\{T\}<([\d.]+)', h)
            pt = {'min': float(ptw.group(1)), 'max': float(ptw.group(2))} if ptw else None
            yields = hf == 'fig2-1.yaml'
            d = {'observable_type': 'particle_yields' if yields else 'mean_pT', 'inspire_id': '630160',
                 'version': VERSION, 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f"{'fiducial dN/dy' if yields else '<pT>'} of {sp} vs y, {cls} ({hf})",
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> X'},
                 'centrality': {'classes': [cls], **TPC}, 'particle_id': {'species': sp},
                 'kinematic_range': {'y': {'min': ybins[0][0], 'max': ybins[-1][1], 'bin_ranges': ybins},
                                     'pT': pt or {'min': 0.0, 'max': 100.0}},
                 'observable': 'dN_dy' if yields else 'mean_pT', 'value_type': 'dN_dy' if yields else 'mean_pT',
                 'notes': {'value': f"{CITE}, Fig. 2. {'dN/dy in the fiducial pT range' if yields else '<pT>'} of {sp} "
                                    f"vs y, {cls}" + ('; tabulated / 5' if scaled else '') + '. Errors: stat and sys '
                                    'in quadrature.'},
                 'staging_reason': 'rapidity-differential yield or <pT> of an identified species: not computed'}
            if scaled:
                d['value_scale'] = 0.2
            staged.append(d)
    if os.path.exists(OLD):
        assert all(str(r['inspire_id']) == '630160' for r in yaml.safe_load_all(open(OLD)) if r), OLD
        os.remove(OLD)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(public)} records;', dict(collections.Counter(d['particle_id']['species'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
