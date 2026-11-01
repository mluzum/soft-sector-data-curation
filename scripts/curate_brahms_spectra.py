#!/usr/bin/env python3
"""Curate BRAHMS identified-hadron spectra from two HEPData records.

- INSPIRE 729167: I. Arsene et al. (BRAHMS), "Nuclear modification factor
  for charged pions and protons at forward rapidity in central Au+Au
  collisions at 200 GeV", PLB 650, 219 (2007), arXiv:nucl-ex/0610021.
  0-10% Au+Au: pi+, pi-, p, pbar at mid-rapidity, -0.05 < y < 0.05
  (invariant yield per unit y, tables 1a-1d), and pi-, pbar at forward
  pseudorapidity eta = 2.2 (per unit eta, tables 1h, 1i; the window is not
  stated -- 2.1 < eta < 2.3 is used).  Proton yields corrected for
  Lambda feed-down; pions not corrected for weak-decay contamination
  (< 5-7%, in the systematics).  The tables give statistical errors; the
  paper's systematic errors are < 15% (pions) and < 18% (protons) at
  mid-rapidity, 14% and 17% forward.  Tables 1e-1g, 1j, 1k are p+p
  references (no centrality classes) and are not curated.
- INSPIRE 836865: I. Arsene et al. (BRAHMS), "Kaon and pion production in
  central Au+Au collisions at sqrt(sNN) = 62.4 GeV", PLB 687, 36 (2010),
  arXiv:0911.2586.  0-10% Au+Au at 62.4 GeV (not 200): pi+-, K+- spectra in
  the rapidity windows of the tables (-0.2 < y < 3.6), statistical errors
  (point-to-point systematics <= 5%).  The weak-decay feed-down correction
  for pions was applied to the integrated dN/dy only, not to the spectra.

Centrality: the BRAHMS multiplicity array (silicon strips and scintillator
tiles, |eta| < 2.2).  pT bins from the tables' independent variable.

Run from the repository root:  python scripts/curate_brahms_spectra.py
"""
import glob
import os
import re

import yaml

MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -2.2, 'max': 2.2}]}
SPECIES = {'PI+': 'pi+', 'PI-': 'pi-', 'P': 'p', 'PBAR': 'pbar', 'K+': 'K+', 'K-': 'K-'}
ARSENE200 = 'I. Arsene et al. (BRAHMS), PLB 650, 219 (2007), arXiv:nucl-ex/0610021'
ARSENE62 = 'I. Arsene et al. (BRAHMS), PLB 687, 36 (2010), arXiv:0911.2586'


def table(iid, name):
    d = glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v1-yaml')[0]
    return yaml.safe_load(open(os.path.join(d, name)))


def pt_axis(t):
    v = t['independent_variables'][0]['values']
    pairs = [[float(x['low']), float(x['high'])] for x in v]
    contiguous = all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:]))
    ax = {'min': pairs[0][0], 'max': pairs[-1][1]}
    if contiguous:
        ax['bins'] = [pairs[0][0]] + [p[1] for p in pairs]
    else:
        ax['bin_ranges'] = pairs
    return ax


def record(iid, hf, energy, sp, kr, cite, note):
    t = table(iid, hf)
    kr = dict(kr, pT=pt_axis(t))
    return {
        'observable_type': 'particle_spectra', 'inspire_id': iid, 'version': 1, 'hepdata_file': hf,
        'dv_index': 0,
        'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
        'centrality': {'classes': ['0-10%'], **MA},
        'kinematic_range': kr, 'particle_id': {'species': sp},
        'observable': 'invariant_yield', 'value_type': 'invariant_yield',
        'notes': {'value': f'{cite}. {note}'},
    }


def main():
    docs200 = []
    for hf, part, where in (('tab1a.yaml', 'PBAR', 'mid'), ('tab1b.yaml', 'PI-', 'mid'), ('tab1c.yaml', 'P', 'mid'),
                            ('tab1d.yaml', 'PI+', 'mid'), ('tab1h.yaml', 'PBAR', 'fwd'), ('tab1i.yaml', 'PI-', 'fwd')):
        q = {x['name']: x.get('value') for x in table('729167', hf)['dependent_variables'][0]['qualifiers']}
        assert q['PARTICLE'] == part and str(q.get('CENTRALITY')) == '0-10', (hf, q)
        if where == 'mid':
            kr = {'y': {'min': -0.05, 'max': 0.05}}
            note = ('Mid-rapidity, -0.05 < y < 0.05, per unit rapidity. Statistical errors; systematic < 15% '
                    '(pions), < 18% (protons), not tabulated.')
        else:
            kr = {'eta': {'min': 2.1, 'max': 2.3}}
            note = ('Forward pseudorapidity eta = 2.2, per unit eta; the window is not stated (2.1 < eta < 2.3 '
                    'used). Statistical errors; systematic 14% (pions), 17% (antiprotons), not tabulated.')
        note += (' Protons corrected for Lambda feed-down; pions not corrected for weak decays (< 5-7%, in the '
                 'systematics).')
        docs200.append(record('729167', hf, 200.0, SPECIES[part], kr, ARSENE200, note))
    docs62 = []
    names = sorted(os.path.basename(p) for p in glob.glob('hepdata_raw/records/ins836865/HEPData-ins836865-v1-yaml/tab1*.yaml'))
    for hf in names:
        q = {x['name']: x.get('value') for x in table('836865', hf)['dependent_variables'][0]['qualifiers']}
        m = re.match(r'^(-?[0-9.]+)-(-?[0-9.]+)$', str(q['YRAP']))
        assert m and str(q['CENTRALITY']) == '0-10' and q['PARTICLE'] in SPECIES, (hf, q)
        lo, hi = float(m.group(1)), float(m.group(2))
        docs62.append(record('836865', hf, 62.4, SPECIES[q['PARTICLE']], {'y': {'min': lo, 'max': hi}}, ARSENE62,
                             f'{lo} < y < {hi}, per unit rapidity. Statistical errors (point-to-point systematic '
                             '<= 5%, not tabulated). Not corrected for weak-decay feed-down into pions (applied to '
                             'the integrated dN/dy only).'))
    assert len(docs62) == 28, len(docs62)
    for docs, out in ((docs200, 'data/particle_spectra/brahms_particle_spectra_identified_AuAu_200gev_ins729167.yaml'),
                      (docs62, 'data/particle_spectra/brahms_particle_spectra_identified_AuAu_62.4gev_ins836865.yaml')):
        with open(out, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{out}: {len(docs)} records')


if __name__ == '__main__':
    main()
