#!/usr/bin/env python3
"""Curate STAR linear and mode-coupled v4, v5 in Au+Au at 200 GeV, INSPIRE 1802752.

J. Adam et al. (STAR), "Investigation of the linear and mode-coupled flow
harmonics in Au+Au collisions at sqrt(sNN) = 200 GeV", PLB 809 (2020)
135728, arXiv:2006.13537.

From the paper:
- charged tracks with |eta| < 1, 0.2 < pT < 4 GeV/c (Sec. 2.1 and Figs.
  1-3; HEPData's qualifier says pT = 0.2-2 GeV/c);
- centrality from the event multiplicity (Glauber), classes 0-5, 5-10,
  10-20, ..., 60-70% (HEPData gives the class centres);
- two-subevent cumulants, Delta eta > 0.7 between A (eta > 0.35) and B
  (eta < -0.35), Eqs. (6)-(8): v_k^inclusive, C_{k,nm}, <v_n^2 v_m^2>;
  v^mc = C / sqrt(<v_n^2 v_m^2>), v^L = sqrt(v_inc^2 - v_mc^2), rho =
  v_mc / v_inc, chi = v_mc / sqrt(<v_n^2 v_m^2>), with <v_2^2 v_3^2> ~
  <v_2^2><v_3^2> for the differential chi_{5,23} (Eq. 15);
- C_{4,22}, C_{5,23} also with the standard (one-window) method (Fig. 1);
  errors stat and syst (non-flow not included).

Records (public), as correlation_type mode_coupling
(analysis/mode_coupling.py): C (one window and two subevents, Table 1;
tabulated x 10^4: value_scale), v_inclusive, v_mode_coupled, v_linear
(Table 2), chi, rho (Table 3) vs class, and their pT dependence in 10-50%
(Table 4, at the tabulated mean pT, 0.1 GeV/c windows; the normalizations
over 0.2-4 GeV/c).  Rows HEPData fills with 0 +- 0.001 (no measurement:
v5 and chi_{5,23} in peripheral classes) are left out.

Run from the repository root:  python scripts/curate_star_mode_coupling_200.py
"""
import glob
import os
import re

import yaml

RAW = glob.glob('hepdata_raw/records/ins1802752/HEPData-ins1802752-v*-yaml')[0]
VERSION = int(RAW.rsplit('-v', 1)[1].split('-')[0])
CITE = 'J. Adam et al. (STAR), PLB 809 (2020) 135728, arXiv:2006.13537'
OUT = 'data/flow_harmonic_correlations/star_flow_harmonic_correlations_mode_coupling_AuAu_200gev_ins1802752.yaml'
WINDOW = 0.1
REFMULT = {'estimator': {'name': 'RefMult', 'observable': 'Nch', 'detector': 'TPC'},
           'eta': [{'min': -0.5, 'max': 0.5}], 'notes': "STAR's charged multiplicity in |eta| < 0.5 (Glauber)."}
CENTRES = {65.0: '60-70%', 55.0: '50-60%', 45.0: '40-50%', 35.0: '30-40%', 25.0: '20-30%', 15.0: '10-20%',
           7.5: '5-10%', 2.5: '0-5%'}
SUBEVENTS = {'A': {'kinematic_range': {'eta': {'min': 0.35, 'max': 1.0}}},
             'B': {'kinematic_range': {'eta': {'min': -1.0, 'max': -0.35}}}}
PT_FIX = {'pT': "HEPData's qualifier says pT = 0.2-2 GeV/c; the paper (Sec. 2.1, Figs. 1-3) uses 0.2 < pT < 4 GeV/c"}
QUANT = {'Inclusive': 'v_inclusive', 'Mode-coupled': 'v_mode_coupled', 'Linear': 'v_linear'}


def harmonics_of(text):
    if re.search(r'4,22|22,4|v_\{4\}', text):
        return [4, 2, 2]
    if re.search(r'5,23|23,5|v_\{5\}', text):
        return [5, 2, 3]
    raise ValueError(text)


def quantity_of(name):
    if name.startswith('$C_'):
        return 'C'
    if name.startswith('$\\chi'):
        return 'chi'
    if name.startswith('$\\rho'):
        return 'rho'
    return QUANT[re.search(r'\^\{(Inclusive|Mode-coupled|Linear)\}', name).group(1)]


def filled(dv):
    """Rows with a measurement: HEPData's 0 +- 0.001 marks none."""
    return [r for r, v in enumerate(dv['values'])
            if str(v['value']).strip() not in ('-', '', '--')
            and not (float(v['value']) == 0.0 and all(e.get('symerror') == 0.001 for e in v['errors']))]


def main():
    docs = []
    for hf in ('table1.yaml', 'table2.yaml', 'table3.yaml', 'table4_a.yaml', 'table4_b.yaml'):
        t = yaml.safe_load(open(os.path.join(RAW, hf)))
        xs = [float(v['value']) for v in t['independent_variables'][0]['values']]
        differential = hf.startswith('table4')
        for i, dv in enumerate(t['dependent_variables']):
            name = dv['header']['name']
            q = quantity_of(name)
            harms = harmonics_of(name)
            rows = filled(dv)
            assert {tuple(e['label'] for e in dv['values'][r]['errors']) for r in rows} == {('stat', 'syst')}
            one_window = '1-Subevent' in name
            label = name.replace('$', '').replace('\\', '').strip()
            d = {'observable_type': 'flow_harmonic_correlations', 'inspire_id': '1802752', 'version': VERSION,
                 'hepdata_file': hf, 'dv_index': i, 'hepdata_table_label': f'{label} ({hf} column {i})',
                 'collision_system': {'energy_gev': 200.0, 'reaction': 'AU AU --> CHARGED X'}}
            if differential:
                cls = dv['qualifiers'][0]['value'].replace('\\', '')
                assert cls == '10-50%', cls
                pts = [xs[r] for r in rows]
                d['centrality'] = {'classes': [cls], **REFMULT}
                kr = {'eta': {'min': -1.0, 'max': 1.0},
                      'pT': {'min': 0.2, 'max': 4.0, 'points': pts, 'point_window': WINDOW}}
            else:
                d['centrality'] = {'classes': [CENTRES[xs[r]] for r in rows], **REFMULT}
                kr = {'eta': {'min': -1.0, 'max': 1.0}, 'pT': {'min': 0.2, 'max': 4.0}}
            d.update(particle_id={'species': 'charged'}, kinematic_range=kr, observable=label,
                     correlation_type='mode_coupling', harmonics=harms, quantity=q, value_type=name)
            if not one_window:
                d.update(particles='AB', subevents=SUBEVENTS)
            if differential:
                d['dimensionality'] = 'differential_pT'
                if q == 'chi' and harms[0] == 5:
                    d['chi_denominator'] = 'factorized'
            if q == 'C':
                d['value_scale'] = 10000.0
            if rows != list(range(len(xs))):
                d['table_rows'] = rows
            d['hepdata_corrections'] = dict(PT_FIX)
            how = ('the standard (one-window) cumulant method' if one_window else
                   'two-subevent cumulants (A: 0.35 < eta < 1, B: -1 < eta < -0.35)')
            what = {'C': f'C_{{{harms[0]},{harms[1]}{harms[2]}}} (tabulated x 10^4)',
                    'v_inclusive': f'inclusive v{harms[0]}', 'v_mode_coupled': f'mode-coupled v{harms[0]}',
                    'v_linear': f'linear v{harms[0]}', 'chi': f'chi_{{{harms[0]},{harms[1]}{harms[2]}}}',
                    'rho': f'rho_{{{harms[0]},{harms[1]}{harms[2]}}}'}[q]
            extra = (' chi with <v2^2 v3^2> ~ <v2^2><v3^2> (Eq. 15).' if d.get('chi_denominator') else '')
            where = (f'vs pT at the tabulated mean pT ({WINDOW:g} GeV/c windows), 10-50%' if differential
                     else 'vs centrality')
            d['notes'] = {'value': f'{CITE}, {hf.split(".")[0].replace("table", "Table ")}. {what} of charged '
                                   f'particles, |eta| < 1, 0.2 < pT < 4 GeV/c, Au+Au 200 GeV, {how}, {where}.'
                                   f'{extra} Errors: stat and syst (non-flow not included).'}
            docs.append(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        yaml.safe_dump_all(docs, fh, sort_keys=False, default_flow_style=False, width=120)
    print(f'{OUT}: {len(docs)} records')


if __name__ == '__main__':
    main()
