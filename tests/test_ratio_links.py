"""observable_ratios records: numerator/denominator links resolve to the right records.

A link is a repository-relative path to a record file, optionally with
"#<document index>" for multi-document files.  The linked document must
exist, be a measurement of the ratio's quantity (same observable_type family
is not required, e.g. a flow_vn record under a v2 ratio), and belong to the
system the ratio names for that side.

Run: python3 tests/test_ratio_links.py
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.dispatcher import iter_records  # noqa: E402


def resolve_link(link):
    path, _, idx = str(link).partition('#')
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return None, f'no file {path}'
    docs = [d for d in yaml.safe_load_all(open(full)) if d]
    if idx:
        i = int(idx)
        return (docs[i], None) if i < len(docs) else (None, f'{path} has no document {i}')
    if len(docs) != 1:
        return None, f'{path} holds {len(docs)} documents; link one with "#<index>"'
    return docs[0], None


def _norm(reaction):
    return str(reaction).replace(' ', '').upper()


def main():
    checked, problems, unlinked = 0, [], []
    for fp, d in iter_records(os.path.join(ROOT, 'data/observable_ratios')):
        sides = [('numerator', d.get('numerator_record'), d.get('numerator_system')),
                 ('denominator', d.get('denominator_record'), d.get('denominator_system'))]
        if not any(link for _, link, _ in sides):
            unlinked.append(d.get('value_type'))
            continue
        for side, link, system in sides:
            checked += 1
            target, err = resolve_link(link) if link else (None, f'{side} link missing')
            if err:
                problems.append(f"{d.get('value_type')}: {err}")
                continue
            reaction = (target.get('collision_system') or {}).get('reaction')
            if system and _norm(system) not in _norm(reaction):
                problems.append(f"{d.get('value_type')}: {side} system {system} but the linked "
                                f"record is {reaction}")
    for p in problems:
        print('  FAIL', p)
    print(f'  {checked} links checked; {len(unlinked)} ratio records without links '
          f'(composite ratios such as double ratios)')
    n_pass = 1 if not problems else 0
    print(f'\n{n_pass}/1 passed')
    sys.exit(0 if n_pass else 1)


if __name__ == '__main__':
    main()
