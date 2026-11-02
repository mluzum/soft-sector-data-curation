#!/usr/bin/env python3
"""Curate ALICE pi, K, p spectra in Pb-Pb at 5.02 TeV, INSPIRE 1759506.

S. Acharya et al. (ALICE), "Production of charged pions, kaons and
(anti-)protons in Pb-Pb and inelastic pp collisions at sqrt(sNN) = 5.02
TeV", PRC 101, 044907 (2020), arXiv:1910.07678.

From the paper and tables:
- primary pi+ + pi-, K+ + K-, p + pbar (summed charges), |y| < 0.5,
  1/N_ev d2N/(dpT dy) (not the invariant yield) in ten V0M classes 0-5 ...
  80-90% (Tables 1, 3, 5) and in inelastic pp (Tables 2, 4, 6);
- errors: stat., syst. (total) and syst. uncorr., "the contribution
  uncorrelated across centrality bins" (Fig. 6 caption), a part of the
  total: errors_excluded (Pb-Pb; the pp tables give stat. and syst.).

Records (public): the 30 Pb-Pb spectra, value_type dN_dpT_dy.  Staged
(--staged OUT): the pp spectra (no centrality classes; the old public
records had the pT bins as centrality classes and species 'charged').  Not
curated: the pT-differential K/pi, p/pi (Tables 7-10; no ratio records:
the pion spectrum's bins differ), the integrated ratios (Tables 11-14:
full-pT extrapolations without tabulated yields) and R_AA (Tables 15-17).

Run from the repository root:
    python scripts/curate_alice_pikp_pbpb_5020.py [--staged OUT.yaml]
"""
import argparse
import collections
import glob
import os

import yaml

RAW = glob.glob('hepdata_raw/records/ins1759506/HEPData-ins1759506-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'S. Acharya et al. (ALICE), PRC 101, 044907 (2020), arXiv:1910.07678'
V0M = {'estimator': {'name': 'V0M', 'observable': 'amplitude', 'detector': 'V0'},
       'eta': [{'min': -3.7, 'max': -1.7}, {'min': 2.8, 'max': 5.1}]}
TABLES = {1: ('pi+ + pi-', 'PB PB'), 3: ('K+ + K-', 'PB PB'), 5: ('p + pbar', 'PB PB'),
          2: ('pi+ + pi-', 'P P'), 4: ('K+ + K-', 'P P'), 6: ('p + pbar', 'P P')}
ERR_NOTE = ("Errors: stat. and syst. (total); 'syst. uncorr.', the part uncorrelated across centrality, is "
            "contained in syst. (errors_excluded).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--staged', help='write the pp records here (a staging collection)')
    staged_out = ap.parse_args().staged
    public, staged = [], []
    for n, (sp, beams) in TABLES.items():
        hf = f'table_{n}.yaml'
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        iv = t['independent_variables'][0]
        for i, dv in enumerate(t['dependent_variables']):
            q = {qq['name']: str(qq.get('value')) for qq in dv.get('qualifiers', [])}
            assert dv['header']['name'] == '(1/Nev)*D2(N)/DPT/DYRAP' and q['YRAP'] == '-0.5-0.5', (hf, dv['header'])
            assert q['RE'].startswith(beams), (hf, q['RE'])
            labels = {e['label'] for v in dv['values'] for e in v.get('errors', [])}
            assert labels in ({'stat.', 'syst.', 'syst. uncorr.'}, {'stat.', 'syst.'}), (hf, labels)
            rows = [j for j, v in enumerate(dv['values']) if str(v['value']).strip() not in ('', '-')]
            pairs = [[float(iv['values'][j]['low']), float(iv['values'][j]['high'])] for j in rows]
            contiguous = all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(pairs[:-1], pairs[1:]))
            pt = ({'min': pairs[0][0], 'max': pairs[-1][1], 'bins': [p[0] for p in pairs] + [pairs[-1][1]]}
                  if contiguous else {'min': pairs[0][0], 'max': pairs[-1][1], 'bin_ranges': pairs})
            if 'CENTRALITY' in q:
                lo, hi = (float(x) for x in q['CENTRALITY'].split('-'))
                cls = f'{lo:g}-{hi:g}%'
            else:
                cls = None
            d = {'observable_type': 'particle_spectra', 'inspire_id': '1759506', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i,
                 'hepdata_table_label': f"1/N_ev d2N/(dpT dy) of {sp}, {cls or 'inelastic pp'} ({hf})",
                 'collision_system': {'energy_gev': 5020.0, 'reaction': q['RE']},
                 'centrality': ({'classes': [cls], **V0M} if cls else {'classes': ['INEL']}),
                 'particle_id': {'species': sp},
                 'kinematic_range': {'y': {'min': -0.5, 'max': 0.5}, 'pT': pt},
                 'observable': 'dN_dpT_dy', 'value_type': 'dN_dpT_dy',
                 'notes': {'value': f"{CITE}. 1/N_ev d2N/(dpT dy) of primary {sp} (summed charges), |y| < 0.5, "
                                    f"{cls + ' (V0M)' if cls else 'inelastic pp'}. "
                                    + (ERR_NOTE if 'syst. uncorr.' in labels else 'Errors: stat. and syst.')}}
            if 'syst. uncorr.' in labels:
                d['errors_excluded'] = ['syst. uncorr.']
            if rows != list(range(len(iv['values']))):
                d['table_rows'] = rows
            if cls:
                public.append(d)
            else:
                d['staging_reason'] = 'inelastic pp: no centrality classes (the pipeline needs percentile classes)'
                staged.append(d)
    old = 'data/particle_spectra/alice_particle_spectra_production_of_charged_pions_kaons_PbPb_pp_5020gev.yaml'
    if os.path.exists(old):
        os.remove(old)
    out = 'data/particle_spectra/alice_particle_spectra_pi_K_p_PbPb_5020gev_ins1759506.yaml'
    with open(out, 'w') as fh:
        yaml.safe_dump_all(public, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{out}: {len(public)} records;', dict(collections.Counter(d['particle_id']['species'] for d in public)))
    if staged_out:
        with open(staged_out, 'w') as fh:
            yaml.safe_dump_all(staged, fh, sort_keys=False, default_flow_style=False, width=120)
        print(f'{staged_out}: {len(staged)} records')


if __name__ == '__main__':
    main()
