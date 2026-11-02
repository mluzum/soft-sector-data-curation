#!/usr/bin/env python3
"""Curate BRAHMS proton and antiproton spectra and dN/dy vs rapidity (three papers).

- INSPIRE 636579: I. G. Bearden et al. (BRAHMS), "Nuclear stopping in
  Au+Au collisions at sqrt(sNN) = 200 GeV", PRL 93, 102301 (2004),
  arXiv:nucl-ex/0312023.  0-5% Au+Au: invariant yields of p, pbar, p - pbar
  in four rapidity windows (Table 1: -0.1 < y < 0.1, 0.75-0.95, 1.7-2.4,
  2.7-3.1), statistical errors; dN/dy vs y (Table 2, fits extrapolated to
  all pT: 85% of dN/dy measured near y = 0, 45% at y ~ 3), errors stat
  and sys.  No weak-decay correction.
- INSPIRE 810481: I. C. Arsene et al. (BRAHMS), "Nuclear stopping and
  rapidity loss in Au+Au collisions at sqrt(sNN) = 62.4 GeV", PLB 677, 267
  (2009), arXiv:0901.0872.  0-10% Au+Au at 62.4 GeV (not 200): p, pbar
  invariant yields in four rapidity windows (Tables 1a-1h), statistical
  errors (Fig. 1); dN/dy vs y (Table 2), statistical errors.  No
  weak-decay correction (the paper estimates it for the net baryons only).
  HEPData gives the spectra units as (GeV/c)^-1; the paper's Fig. 1,
  GeV^-2 c^3.
- INSPIRE 856692: I. Arsene et al. (BRAHMS), "Rapidity dependence of
  deuteron production in central Au+Au collisions at sqrt(sNN) = 200
  GeV", PRC 83, 044906 (2011), arXiv:1005.5427.  0-20% Au+Au: p, d in four
  rapidity bins (-0.1, 0.1), (0.5, 1.2), (1.5, 2.5), (2.8, 3.2) (HEPData
  'YRAP' 0, 1, 2, 3), pbar, dbar in the first two (Tables 3a-3l);
  protons corrected for hyperon feed-down (Sec. II E); statistical errors.

Centrality: the BRAHMS multiplicity array (|eta| < 2.2).  The old records
had the rapidity windows as centrality classes, species 'charged' for
deuterons and 200 GeV for the 62.4 GeV paper.

Records (public): the p and pbar spectra (8 + 8 + 6) and dN/dy (one
record per rapidity bin and species: 18 + 8).  Staged (--staged OUT):
p - pbar (636579) and the deuterons (856692): species not provided.  Not
curated: C_Lambda (856692 Table 2), B2 and <f> (Tables 4, 5).

Run from the repository root:  python scripts/curate_brahms_stopping.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os
import re

import yaml

MA = {'estimator': {'name': 'MA', 'observable': 'Nch', 'detector': 'Multiplicity array'},
      'eta': [{'min': -2.2, 'max': 2.2}]}
SPECIES = {'P': 'p', 'PBAR': 'pbar', 'P - PBAR': 'p - pbar', 'DEUT': 'd', 'DEUTBAR': 'dbar'}
PAPERS = {
    '636579': {'energy': 200.0, 'cls': '0-5%', 'out': 'stopping_AuAu_200gev_ins636579',
               'cite': 'I. G. Bearden et al. (BRAHMS), PRL 93, 102301 (2004), arXiv:nucl-ex/0312023',
               'note': 'No weak-decay correction.'},
    '810481': {'energy': 62.4, 'cls': '0-10%', 'out': 'stopping_AuAu_62gev_ins810481',
               'cite': 'I. C. Arsene et al. (BRAHMS), PLB 677, 267 (2009), arXiv:0901.0872',
               'note': 'No weak-decay correction.'},
    '856692': {'energy': 200.0, 'cls': '0-20%', 'out': 'p_d_AuAu_200gev_ins856692',
               'cite': 'I. Arsene et al. (BRAHMS), PRC 83, 044906 (2011), arXiv:1005.5427',
               'note': 'Protons corrected for hyperon feed-down (Sec. II E).'},
}
Y856692 = {'0': (-0.1, 0.1), '1': (0.5, 1.2), '2': (1.5, 2.5), '3': (2.8, 3.2)}
OLD = ['data/particle_spectra/brahms_particle_spectra_nuclear_stopping_AuAu_200gev.yaml',
       'data/particle_yields/brahms_particle_yields_nuclear_stopping_AuAu_200gev.yaml',
       'data/particle_spectra/brahms_particle_spectra_nuclear_stopping_and_rapidity_loss_AuAu_200gev.yaml',
       'data/particle_yields/brahms_particle_yields_nuclear_stopping_and_rapidity_loss_AuAu_200gev.yaml',
       'data/particle_spectra/brahms_particle_spectra_rapidity_dependence_of_deuteron_production_AuAu_200gev.yaml']


def raw(iid):
    return glob.glob(f'hepdata_raw/records/ins{iid}/HEPData-ins{iid}-v1-yaml')[0]


def y_window(iid, q):
    if iid == '856692':
        return Y856692[str(q['YRAP'])]
    lo, hi = re.fullmatch(r'\s*(-?[\d.]+)\s*-\s*(-?[\d.]+)\s*', str(q['YRAP'])).groups()
    return float(lo), float(hi)


def errs(dv):
    labels = sorted({e['label'] for v in dv['values'] for e in v.get('errors', [])})
    return 'statistical' if labels == ['stat'] else 'statistical and systematic'


def base(iid, otype, hf, i, sp, label):
    p = PAPERS[iid]
    return {'observable_type': otype, 'inspire_id': iid, 'version': 1, 'hepdata_file': hf, 'dv_index': i,
            'hepdata_table_label': label,
            'collision_system': {'energy_gev': p['energy'], 'reaction': f'AU AU --> {sp.upper()} X'},
            'centrality': {'classes': [p['cls']], **MA}, 'particle_id': {'species': sp}}


def spectra(iid, hf):
    p = PAPERS[iid]
    t = yaml.safe_load(open(os.path.join(raw(iid), hf)))
    iv = t['independent_variables'][0]
    for i, dv in enumerate(t['dependent_variables']):
        assert dv['header']['name'] == '1/(2*PI*PT)*D2N/DPT/DYRAP', (iid, hf, dv['header'])
        q = {qq['name']: qq.get('value') for qq in dv['qualifiers']}
        assert str(q.get('CENTRALITY', p['cls'][:-1])) == p['cls'][:-1], (iid, hf, q)
        sp = SPECIES[q['PARTICLE']]
        y0, y1 = y_window(iid, q)
        rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-', 'nan')]
        pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
        contiguous = all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:]))
        pt = ({'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [x[0] for x in pairs] + [pairs[-1][1]]}
              if contiguous else {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs})
        d = base(iid, 'particle_spectra', hf, i, sp, f'1/(2 pi pT) d2N/(dpT dy) of {sp}, {y0:g} < y < {y1:g} ({hf})')
        d.update({'kinematic_range': {'y': {'min': y0, 'max': y1}, 'pT': pt},
                  'observable': 'invariant_yield', 'value_type': 'invariant_yield',
                  'notes': {'value': f"{p['cite']}. Invariant yield of {sp}, {p['cls']}, {y0:g} < y < {y1:g}. "
                                     f"{p['note']} Errors: {errs(dv)}."}})
        if rows != list(range(len(iv['values']))):
            d['table_rows'] = rows
        yield d


def dndy(iid, hf):
    p = PAPERS[iid]
    t = yaml.safe_load(open(os.path.join(raw(iid), hf)))
    iv = t['independent_variables'][0]
    assert iv['header']['name'] == 'YRAP', (iid, hf)
    for i, dv in enumerate(t['dependent_variables']):
        assert dv['header']['name'] == 'DN/DYRAP', (iid, hf, dv['header'])
        q = {qq['name']: qq.get('value') for qq in dv['qualifiers']}
        assert str(q['CENTRALITY']) == p['cls'][:-1], (iid, hf, q)
        sp = SPECIES[q['PARTICLE']]
        for j, v in enumerate(dv['values']):
            if str(v['value']).strip() in ('', '-', 'nan'):
                continue
            y0, y1 = float(iv['values'][j]['low']), float(iv['values'][j]['high'])
            d = base(iid, 'particle_yields', hf, i, sp, f'dN/dy of {sp}, {y0:g} < y < {y1:g} ({hf}, row {j})')
            d.update({'kinematic_range': {'y': {'min': y0, 'max': y1}, 'pT': {'min': 0.0, 'max': 100.0}},
                      'observable': 'dN_dy', 'value_type': 'dN_dy', 'table_rows': [j],
                      'notes': {'value': f"{p['cite']}. dN/dy of {sp}, {p['cls']}, {y0:g} < y < {y1:g}, all pT "
                                         f"(fits extrapolated outside the measured pT). {p['note']} Errors: "
                                         f"{errs(dv)}."}})
            yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the staged records here (a staging collection)')
    staged_out = ap.parse_args().staged
    out = collections.defaultdict(list)
    staged = []
    tables = {'636579': ['tab1.yaml'], '810481': [f'tab1{c}.yaml' for c in 'abcdefgh'],
              '856692': [f'tab3{c}.yaml' for c in 'abcdefghijkl']}
    for iid, files in tables.items():
        docs = [d for hf in files for d in spectra(iid, hf)]
        if iid != '856692':
            docs += list(dndy(iid, 'tab2.yaml'))
        for d in docs:
            sp = d['particle_id']['species']
            if sp in ('p', 'pbar'):
                out[(d['observable_type'], iid)].append(d)
            else:
                d['staging_reason'] = f'species {sp!r} is not provided by the pipeline'
                staged.append(d)
    for f in OLD:
        if os.path.exists(f):
            os.remove(f)
    for (otype, iid), docs in out.items():
        path = f'data/{otype}/brahms_{otype}_{PAPERS[iid]["out"]}.yaml'
        with open(path, 'w') as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{path}: {len(docs)} records;', dict(collections.Counter(d['particle_id']['species'] for d in docs)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
