#!/usr/bin/env python3
"""Curate ATLAS v_n (n = 2-7) in Pb+Pb at 5.02 TeV, INSPIRE 1686834.

M. Aaboud et al. (ATLAS), "Measurement of the azimuthal anisotropy of
charged particles produced in sqrt(sNN) = 5.02 TeV Pb+Pb collisions with
the ATLAS detector", EPJC 78, 997 (2018), arXiv:1808.03951.

From the paper:
- tracks |eta| < 2.5; centrality from the FCal Sigma E_T (3.2 < |eta| <
  4.9), classes from 0-0.1% to 70-80%;
- scalar product (SP): the track flow vector correlated with the FCal
  flow vector Q_n^{N|P} of the side opposite the track (3.2 < |eta| < 4.9,
  E_T weighted), normalized by sqrt(<Q_n^N Q_n^P*>);
- event plane (EP): the same with the FCal event-plane angles and their
  resolution; v_n{EP} lies between <v_n> and sqrt(<v_n^2>);
- two-particle correlations (2PC): pairs with 2 < |Delta eta| < 5,
  reference 0.5 < pT_b < 5 GeV;
- v_n(eta) folded into |eta| (0-2.5) for three pT ranges;
- errors: statistical and systematic.

Records (one per table):
- SP and EP v_n(pT), |eta| < 2.5: scalar_product / event_plane with the
  FCal reference, both sides (a pT bin spans both halves of the tracker;
  ATLAS takes the opposite side per track, which in a model without
  non-flow differs only by longitudinal decorrelation);
- 2PC v_n(pT): cumulant order 2 with the 0.5-5 GeV reference
  (flow_reference.pT); the pair gap is recorded as eta_gap, not computed;
- SP and EP v_n(|eta|): the mirrored FCal sub-event (the side opposite the
  bin), computed at eta > 0 (v_n(eta) is symmetric).
The pipeline computes the event-plane records with the scalar-product
estimator (it has no resolution correction); ATLAS reports SP/EP within a
few per cent.  Not curated: v_n vs <N_part> (overlapping classes 0-0.1, 0-1,
0-5% in one table), the EP tables with Run-1 pT binning (duplicates), the
SP/EP and SP/2PC ratios, and the scaled v_n(pT) and its scale factors.

Run from the repository root:  python scripts/curate_atlas_vn_pbpb_5tev.py
"""
import os
import re

import yaml

RAW = 'hepdata_raw/records/ins1686834/HEPData-ins1686834-v1-yaml'
CITE = 'M. Aaboud et al. (ATLAS), EPJC 78, 997 (2018), arXiv:1808.03951'
FCAL = {'estimator': {'name': 'FCal', 'observable': 'ET', 'detector': 'FCal'},
        'eta': [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]}
FCAL_BOTH = [{'min': -4.9, 'max': -3.2}, {'min': 3.2, 'max': 4.9}]
PT_RE = re.compile(r"The V(\d) harmonic measured with the (scalar product|event plane|two particle correlation) method "
                   r"as a fun?c?tion of transverse momentum in centrality bin ([0-9.]+-[0-9.]+)%(\. PT binning matched to RUN1\.)?$")
ETA_RE = re.compile(r"The V(\d) harmonic measured with the (scalar product|event plane) method as a fun?c?tion of "
                    r"pseudorapidity for transverse momentum range ([0-9.]+) < PT < ([0-9.]+) GeV in centrality bin "
                    r"([0-9.]+-[0-9.]+)%$")
METHOD = {'scalar product': 'scalar_product', 'event plane': 'event_plane', 'two particle correlation': 'cumulant'}


def axis(t):
    pairs = [[float(v['low']), float(v['high'])] for v in t['independent_variables'][0]['values']]
    ax = {'min': pairs[0][0], 'max': pairs[-1][1]}
    if all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs, pairs[1:])):
        ax['bins'] = [pairs[0][0]] + [p[1] for p in pairs]
    else:
        ax['bin_ranges'] = pairs
    return ax


def base(hf, n, cls, label):
    return {'observable_type': 'flow_vn', 'inspire_id': '1686834', 'version': 1, 'hepdata_file': hf,
            'dv_index': 0, 'hepdata_table_label': label,
            'collision_system': {'energy_gev': 5020.0, 'reaction': 'PB PB --> CHARGED X'},
            'centrality': {'classes': [f'{cls}%'], **FCAL},
            'particle_id': {'species': 'charged'}, 'harmonic': n, 'value_type': f'V{n}'}


def main():
    sub = [x for x in yaml.safe_load_all(open(os.path.join(RAW, 'submission.yaml'))) if x and 'data_file' in x]
    docs, skipped = [], 0
    for x in sub:
        desc = ' '.join(x['description'].split())
        hf = x['data_file']
        m, e = PT_RE.match(desc), ETA_RE.match(desc)
        if m and not m.group(4):
            n, meth, cls = int(m.group(1)), METHOD[m.group(2)], m.group(3)
            t = yaml.safe_load(open(os.path.join(RAW, hf)))
            q = {qq['name']: qq.get('value') for qq in t['dependent_variables'][0].get('qualifiers', [])}
            assert q.get('CENTRALITY') == cls and q.get('ETARAP') == '-2.5-2.5', (hf, q)
            d = base(hf, n, cls, desc)
            d.update({'kinematic_range': {'eta': {'min': -2.5, 'max': 2.5}, 'pT': axis(t)},
                      'method': meth, 'dimensionality': 'differential_pT'})
            if meth == 'cumulant':
                d.update({'cumulant_order': 2, 'eta_gap': 2.0,
                          'flow_reference': {'pT': {'min': 0.5, 'max': 5.0},
                                             'notes': 'Two-particle correlations, 2 < |Delta eta| < 5, reference '
                                                      '0.5 < pT_b < 5 GeV (no pair gap in the computation).'}})
                how = 'two-particle correlations (2 < |Delta eta| < 5, reference 0.5 < pT_b < 5 GeV)'
            else:
                d['flow_reference'] = {'detector': 'FCal', 'eta': FCAL_BOTH,
                                       'notes': 'FCal Q_n, 3.2 < |eta| < 4.9, E_T weighted; ATLAS uses the side '
                                                'opposite each track (both sides here: the pT bins span |eta| < 2.5).'}
                how = (f'{m.group(2)} method with the FCal reference'
                       + ('; computed with the scalar-product estimator (EP/SP within a few per cent)'
                          if meth == 'event_plane' else ''))
            d['notes'] = {'value': f'{CITE}. v_{n}(pT), |eta| < 2.5, {how}. Errors: statistical and systematic.'}
            docs.append(d)
        elif e:
            n, meth, plo, phi_, cls = int(e.group(1)), METHOD[e.group(2)], float(e.group(3)), float(e.group(4)), e.group(5)
            t = yaml.safe_load(open(os.path.join(RAW, hf)))
            q = {qq['name']: qq.get('value') for qq in t['dependent_variables'][0].get('qualifiers', [])}
            assert q.get('CENTRALITY') == cls, (hf, q)
            d = base(hf, n, cls, desc)
            d.update({'kinematic_range': {'eta': axis(t), 'pT': {'min': plo, 'max': phi_}},
                      'method': meth, 'dimensionality': 'differential_eta',
                      'flow_reference': {'detector': 'FCal', 'mirror': True, 'eta': [{'min': 3.2, 'max': 4.9}],
                                         'notes': 'FCal Q_n of the side opposite the track, 3.2 < |eta| < 4.9.'},
                      'notes': {'value': (f'{CITE}. v_{n}(|eta|) folded into 0 < |eta| < 2.5 for {plo:g} < pT < '
                                          f'{phi_:g} GeV, {e.group(2)} method, FCal on the opposite side; computed '
                                          'at eta > 0.' + (' Computed with the scalar-product estimator (EP/SP '
                                                           'within a few per cent).' if meth == 'event_plane' else '')
                                          + ' Errors: statistical and systematic.')}})
            docs.append(d)
        else:
            skipped += 1
    out = 'data/flow_vn/atlas_flow_vn_azimuthal_anisotropy_of_charged_particles_PbPb_5020gev.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    import collections
    print(f'{out}: {len(docs)} records, {skipped} tables not curated;',
          dict(collections.Counter((d['method'], d['dimensionality']) for d in docs)))


if __name__ == '__main__':
    main()
