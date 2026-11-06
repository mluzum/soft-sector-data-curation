#!/usr/bin/env python3
"""Curate STAR linear and mode-coupled v4, v5 and NSC in Au+Au at 54.4, 39 and 27 GeV, INSPIRE 2634985.

B. Aboona et al. (STAR), "Beam energy dependence of the linear and
mode-coupled flow harmonics in Au+Au collisions", PLB 839 (2023) 137755,
arXiv:2211.11637.

From the paper:
- charged tracks, |eta| < 1, 0.2 < pT < 4 GeV/c; centrality from the
  charged multiplicity in |eta| < 0.5 (Glauber);
- two-subevent cumulants (A: eta > 0.35, B: eta < -0.35), Eqs. (6)-(12):
  C_{k,nm}, inclusive, mode-coupled and linear v_k, chi_{k,nm},
  rho_{k,nm} (as STAR 1802752 at 200 GeV);
- NSC(n, m) = SC(n, m) / ((v_n^inc)^2 (v_m^inc)^2), SC from the standard
  (one-window) cumulants, the inclusive v_n from the two subevents (Eqs.
  13, 14);
- errors stat and syst; C_{k,nm} plotted x 10^4.

The tables give each class by its centre, the classes merged differently
per energy and quantity (e.g. 40 -> 30-50%, 30 -> 20-40%, 5 -> 0-10%):
the classes tile 0% upwards, each centred on its point, edges on 5%.  The
54.4 GeV NSC table puts its most central point at 2.1%: 0-5% (the next
class is 5-10%), a plotting offset (hepdata_corrections).

Records (public): C_{4,22}, C_{5,23} (value_scale 10^4), v4 (inclusive,
mode-coupled, linear) at the three energies and v5 at 54.4 GeV,
chi_{4,22} and rho_{4,22} at the three energies and chi_{5,23},
rho_{5,23} at 54.4 GeV (correlation_type mode_coupling,
analysis/mode_coupling.py), NSC(2,3) and NSC(2,4) at 54.4 and 27 GeV
(symmetric_cumulant, normalized, normalization_eta_gap 0.7).

Run from the repository root:  python scripts/curate_star_mode_coupling_bes.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins2634985/HEPData-ins2634985-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'B. Aboona et al. (STAR), PLB 839 (2023) 137755, arXiv:2211.11637'
OUT = 'data/flow_harmonic_correlations/star_flow_harmonic_correlations_mode_coupling_AuAu_bes_ins2634985.yaml'
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': 'charged multiplicity in |eta| < 0.5 (Glauber).'}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': 0.35, 'max': 1.0}}},
             'B': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.35}}}}
ENERGY = {'54': 54.4, '39': 39.0, '27': 27.0}
QUANT = {'Inclusive': 'v_inclusive', 'Mode-coupled': 'v_mode_coupled', 'Linear': 'v_linear'}


def classes_from_centres(centres):
    """Classes tiling 0% upwards, each centred on its point (edges on 5%);
    a first point below 2.5 followed by 7.5 is 0-5% plotted off-centre."""
    asc = sorted(centres)
    edges, offset = [0.0], False
    for i, c in enumerate(asc):
        hi = 2 * c - edges[-1]
        if i == 0 and c < 2.5 and len(asc) > 1 and asc[1] == 7.5:
            hi, offset = 5.0, True
        assert hi > edges[-1] and abs(hi / 5 - round(hi / 5)) < 1e-9, (centres, edges, hi)
        edges.append(hi)
    cls = {c: f'{edges[i]:g}-{edges[i + 1]:g}%' for i, c in enumerate(asc)}
    return [cls[c] for c in centres], offset


def harmonics_of(name):
    if re.search(r'224|4,22|v_\{4\}', name):
        return [4, 2, 2]
    if re.search(r'235|5,23|v_\{5\}', name):
        return [5, 2, 3]
    raise ValueError(name)


def main():
    docs = []
    for hf in sorted(f for f in os.listdir(RAW) if f.startswith('Fig_')):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        energy = ENERGY[re.search(r'_(54|39|27)', hf).group(1)]
        centres = [float(v['value']) for v in t['independent_variables'][0]['values']]
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name'].strip()
            rows = [r for r, v in enumerate(dv['values']) if str(v['value']).strip() not in ('-', '', '--')]
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'syst')}, hf
            classes, offset = classes_from_centres([centres[r] for r in rows])
            label = re.sub(r'\s+', ' ', name.replace('$', '').replace('\\', '')).strip()
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': '2634985', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{label}, {energy:g} GeV ({hf} column {i})',
                 'collision_system': {'energy_gev': energy, 'reaction': 'AU AU --> CHARGED X'},
                 'centrality': {'classes': classes, **REFMULT}, 'particle_id': {'species': 'charged'},
                 'kinematic_range': {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 4.0}},
                 'observable': label, 'value_type': name}
            where = (f'charged particles, |eta| < 1, 0.2 < pT < 4 GeV/c, Au+Au {energy:g} GeV vs centrality (classes '
                     'tiling 0% upwards around the tabulated centres)')
            if name.startswith('$NSC'):
                m, n = (int(x) for x in re.search(r'NSC\((\d),(\d)\)', name).groups())
                d.update(correlation_type='symmetric_cumulant', harmonics=[m, n], normalized=True,
                         normalization_eta_gap=0.7)
                d['notes'] = {'value': f'{CITE}, Fig. 4. NSC({m},{n}) of {where}: SC from the standard cumulants, '
                                       'normalized by the two-subevent v^2 (|Delta eta| > 0.7). Errors: stat and syst.'}
            else:
                harms = harmonics_of(name)
                q = ('C' if name.startswith('$C_') else 'chi' if name.startswith('$\\chi') else
                     'rho' if name.startswith('$\\rho') else
                     QUANT[re.search(r'\((Inclusive|Mode-coupled|Linear)\)', name).group(1)])
                d.update(correlation_type='mode_coupling', harmonics=harms, quantity=q, particles='AB',
                         subevents=SUBEVENTS)
                if q == 'C':
                    d['value_scale'] = 10000.0
                what = {'C': f'C_{{{harms[0]},{harms[1]}{harms[2]}}} (tabulated x 10^4)',
                        'v_inclusive': f'inclusive v{harms[0]}', 'v_mode_coupled': f'mode-coupled v{harms[0]}',
                        'v_linear': f'linear v{harms[0]}', 'chi': f'chi_{{{harms[0]},{harms[1]}{harms[2]}}}',
                        'rho': f'rho_{{{harms[0]},{harms[1]}{harms[2]}}}'}[q]
                fig = {'C': 1, 'v_inclusive': 2, 'v_mode_coupled': 2, 'v_linear': 2, 'chi': 3, 'rho': 3}[q]
                d['notes'] = {'value': f'{CITE}, Fig. {fig}. {what} of {where}, two-subevent cumulants (A: 0.35 < '
                                       'eta < 1, B: -1 < eta < -0.35). Errors: stat and syst.'}
            if rows != list(range(len(centres))):
                d['table_rows'] = rows
            if offset:
                d['hepdata_corrections'] = {'centrality': 'the most central point is plotted at 2.1%: the 0-5% '
                                                          'class (the next is 5-10%)'}
            docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
