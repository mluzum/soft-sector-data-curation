#!/usr/bin/env python3
"""Curate BRAHMS identified spectra in Cu+Cu at 200 GeV, INSPIRE 1419279.

I. Arsene et al. (BRAHMS), "Rapidity and centrality dependence of particle
production for identified hadrons in Cu+Cu collisions at sqrt(sNN) = 200
GeV", PRC 94, 014907 (2016), arXiv:1602.01183.

From the paper:
- invariant yields 1/(2 pi pT) d2N/(dpT dy) of pi+-, K+-, p, pbar at y = 0
  (Mid-Rapidity Spectrometer, |y| < 0.1) and y = 3 (Forward Spectrometer,
  2.95 < y < 3.15), in 0-10, 10-30, 30-50, 50-70%;
- centrality from the charged-particle multiplicity in the Multiplicity
  Array, |eta| < 2.2;
- no Lambda feed-down correction for p and pbar; pions corrected for
  weak-decay contamination as stated in Sec. III;
- the figure scales the spectra for display; the tables do not;
- errors: statistical and systematic.

Records: the 48 spectra (6 species x 4 classes x 2 rapidities) with their
rapidity windows (the records had a pseudorapidity window -0.1 < eta < 3.3
for every table).  Not curated: dN/dy/(N_part/2) (extrapolated yields
divided by N_part), R_AA (needs p+p), and the particle ratios vs N_part
and pT (no particle-ratio computation).

Run from the repository root:  python scripts/curate_brahms_cucu_spectra.py
"""
import collections
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1419279/HEPData-ins1419279-v1-yaml'
CITE = 'I. Arsene et al. (BRAHMS), PRC 94, 014907 (2016), arXiv:1602.01183'
MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -2.2, 'max': 2.2}]}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'K+': 'K+', 'K-': 'K-', 'P': 'p', 'PBAR': 'pbar'}
YWIN = {0.0: (-0.1, 0.1), 3.0: (2.95, 3.15)}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs = []
    for x in sub:
        hf = x['data_file']
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        dv = t['dependent_variables'][0]
        if dv['header']['name'] != '1/(2*PI*PT)*D2N/DPT/DYRAP':
            continue
        q = {qq['name']: qq.get('value') for qq in dv['qualifiers']}
        sp, y = SPECIES[q['PARTICLE']], float(q['YRAP'])
        lo, hi = (float(v) for v in re.fullmatch(r'\s*(\d+) TO (\d+)\s*', q['CENTRALITY']).groups())
        cls = f'{lo:g}-{hi:g}%'
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
        assert rows == list(range(rows[0], rows[-1] + 1)), hf
        vals = [t['independent_variables'][0]['values'][j] for j in rows]
        pairs = [(float(v['low']), float(v['high'])) for v in vals]
        assert all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])), hf
        edges = [pairs[0][0]] + [p[1] for p in pairs]
        ylo, yhi = YWIN[y]
        docs.append({
            'observable_type': 'particle_spectra', 'inspire_id': '1419279', 'version': 1, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': f'{sp} invariant yield, {cls} Cu+Cu, y = {y:g} ({x.get("location", "")})',
            'collision_system': {'energy_gev': 200.0, 'reaction': 'CU CU --> X'},
            'centrality': {'classes': [cls], **MA},
            'kinematic_range': {'y': {'min': ylo, 'max': yhi}, 'pT': {'min': edges[0], 'max': edges[-1], 'bins': edges}},
            'particle_id': {'species': sp},
            'observable': 'invariant_yield', 'value_type': 'invariant_yield',
            'notes': {'value': (f'{CITE}. 1/(2 pi pT) d2N/(dpT dy) of {sp}, {cls}, {ylo:g} < y < {yhi:g}'
                                + ('; p and pbar not corrected for Lambda feed-down' if sp in ('p', 'pbar') else '')
                                + '. Errors: statistical and systematic.')},
        })
    out = 'data/particle_spectra/brahms_particle_spectra_rapidity_and_centrality_dependence_of_CuCu_200gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(docs)} records;',
          dict(collections.Counter((d['particle_id']['species'], d['kinematic_range']['y']['min']) for d in docs)))


if __name__ == '__main__':
    main()
