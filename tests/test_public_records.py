"""Every record in data/ can be computed and compared end to end.

The public database holds published measurements that the tools here can
compute from a simulation and compare with the data -- the test of a fully
vetted record.  For each record: resolve() dispatches it, its centrality
classes are percentile bins, and its experimental values load.  Ratio
records qualify when both linked sides do (scripts/compute_ratios.py); the
records a public ratio links to belong here too.  A result published only
as a ratio (BRAHMS p/pi) has side records with experimental_values: false:
they compute and load no values, and are public only as the target of a
public ratio's link, and every ratio record's own values must load.
Records that do not qualify yet are curated elsewhere and used through
--extra-records.

Run: python3 tests/test_public_records.py
"""
import contextlib
import io
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from analysis.dispatcher import extract_measurement_params, load_experimental, resolve  # noqa: E402
from analysis.ratios import resolve_link, side_record  # noqa: E402


def problem(doc, linked=False):
    """None if computable end to end, else what is missing.

    linked: the doc is a side of a public ratio, so a value-less side
    (experimental_values: false, the values only in the ratio) qualifies.
    """
    try:
        energy = float((doc.get('collision_system') or {}).get('energy_gev') or 0)
    except (TypeError, ValueError):
        energy = 0.0
    if energy <= 0:
        return 'no collision energy (collision_system.energy_gev): no simulation group matches it'
    try:
        resolve(doc)
    except ValueError as e:
        return f'not dispatchable: {e}'
    p = extract_measurement_params(doc)
    if not p.get('cent_edges') or p.get('cent_classes_suspicious'):
        return 'no percentile centrality classes'
    if doc.get('experimental_values') is False:
        return None if linked else 'experimental_values: false, but no public ratio links this record'
    with contextlib.redirect_stdout(io.StringIO()):
        v, _ = load_experimental(p)
    return None if v is not None else 'experimental values do not load'


def main():
    bad, n, link_targets, docs = [], 0, set(), {}
    for t in sorted(os.listdir('data')):
        if not os.path.isdir(f'data/{t}'):
            continue
        for f in sorted(os.listdir(f'data/{t}')):
            path = f'data/{t}/{f}'
            for i, d in enumerate(yaml.safe_load_all(open(path))):
                if d is not None:
                    d['_record_file'] = os.path.abspath(path)    # resolves a relative value_table
                    docs[(os.path.abspath(path), i)] = d
    for (path, i), d in docs.items():
        if d.get('observable_type') != 'observable_ratios':
            continue
        with contextlib.redirect_stdout(io.StringIO()):          # the ratio's own values
            v, _ = load_experimental(extract_measurement_params(d))
        if v is None:
            bad.append(f'{os.path.relpath(path, ROOT)}#{i}: the ratio\'s experimental values do not load')
        for link in (d.get('numerator_record'), d.get('denominator_record')):
            try:
                p, j, side = resolve_link(link, ROOT)
                why = problem(side_record(side, d), linked=True)
            except ValueError as e:
                why = str(e)
            if why:
                bad.append(f'{os.path.relpath(path, ROOT)}#{i}: ratio side {link}: {why}')
            else:
                link_targets.add((os.path.abspath(p), j))
    for key, d in docs.items():
        n += 1
        if d.get('observable_type') == 'observable_ratios' or key in link_targets:
            continue
        why = problem(d)
        if why:
            bad.append(f'{os.path.relpath(key[0], ROOT)}#{key[1]}: {why}')
    for b in bad[:20]:
        print('  FAIL', b)
    print(f'  {n} records checked, {len(bad)} not computable end to end')
    ok = not bad
    print(f'\n{int(ok)}/1 passed')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
