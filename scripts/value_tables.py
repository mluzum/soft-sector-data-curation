"""Write HEPData-format value tables for records without a HEPData table.

A record's `value_table` (observables/_common.yaml, record_status) is read by
analysis.dispatcher.load_experimental exactly like a HEPData table: the
record's `dv_index` selects a dependent variable, whose rows must follow the
record's bins (or centrality classes).  Each point's first error is taken as
the statistical one, a second as the systematic one; a single asymmetric
error enters as the mean of its two sides.

    from scripts.value_tables import write_value_table
    write_value_table(path, description, x={'name': 'ETA', 'bins': [(-5.4, -5.2), ...]},
                      series=[{'name': 'DN/DETA', 'qualifiers': {'CENTRALITY': '0-3%'},
                               'values': [...], 'errors': [[{'label': 'sys', 'plus': .., 'minus': ..}], ...]}])
"""
import os

import yaml


def _error(e):
    if 'sym' in e:
        return {'symerror': float(e['sym']), 'label': e['label']}
    return {'asymerror': {'plus': float(e['plus']), 'minus': -abs(float(e['minus']))}, 'label': e['label']}


def write_value_table(path, description, x, series):
    """x: {'name', 'units'?, 'bins': [(lo, hi)] or 'values': [v]};
    series: [{'name', 'units'?, 'qualifiers': {name: value}, 'values': [v],
    'errors': [[{'label', 'sym'} or {'label', 'plus', 'minus'}] per point]}]."""
    n = len(x.get('bins') or x.get('values'))
    xv = ([{'low': float(lo), 'high': float(hi)} for lo, hi in x['bins']] if 'bins' in x
          else [{'value': float(v)} for v in x['values']])
    dvs = []
    for s in series:
        if len(s['values']) != n or len(s['errors']) != n:
            raise ValueError(f"{s['name']}: {len(s['values'])} values, {len(s['errors'])} errors, {n} bins")
        dvs.append({
            'header': {'name': s['name'], 'units': s.get('units', '')},
            'qualifiers': [{'name': k, 'value': v} for k, v in s.get('qualifiers', {}).items()],
            'values': [{'value': float(v), 'errors': [_error(e) for e in errs]}
                       for v, errs in zip(s['values'], s['errors'])],
        })
    table = {'description': description,
             'independent_variables': [{'header': {'name': x['name'], 'units': x.get('units', '')},
                                        'values': xv}],
             'dependent_variables': dvs}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        yaml.safe_dump(table, fh, sort_keys=False, default_flow_style=None, width=120)
    return table
