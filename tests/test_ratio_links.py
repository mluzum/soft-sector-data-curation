"""observable_ratios records: numerator/denominator links resolve to the right records.

A link is a repository-relative path to a record file, optionally with
"#<document index>" for multi-document files.  The linked document must
exist, be a measurement of the ratio's quantity (same observable_type family
is not required, e.g. a flow_vn record under a v2 ratio), and belong to the
system the ratio names for that side (nuclei, and the energy when the side
names one, as energy ratios do: 'Pb+Pb 5.02 TeV').  A ratio within one
system (centrality_ratio, rapidity_ratio) names the same system on both
sides, and a denominator_class is a class of the linked denominator record.

Run: python3 tests/test_ratio_links.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.dispatcher import iter_records  # noqa: E402
from analysis import ratios  # noqa: E402


def resolve_link(link):
    """Links of this repository's ratio records must resolve within it."""
    try:
        return ratios.resolve_link(link, ROOT)[2], None
    except ValueError as e:
        return None, str(e)


def _norm(reaction):
    return str(reaction).replace(' ', '').replace('+', '').upper()


_ENERGY = re.compile(r'\s+([0-9.]+)\s*(GeV|TeV)\s*$', re.IGNORECASE)


def side_matches(system, target):
    """A side's system string ('Ru+Ru', 'Pb+Pb 5.02 TeV') against the linked
    record: the nuclei in its reaction, and the energy when the string states
    one (energy ratios link two records of the same reaction)."""
    cs = target.get('collision_system') or {}
    m = _ENERGY.search(system)
    nuclei = system[:m.start()] if m else system
    if _norm(nuclei) not in _norm(cs.get('reaction')):
        return False
    if m:
        gev = float(m.group(1)) * (1000.0 if m.group(2).lower() == 'tev' else 1.0)
        e = cs.get('energy_gev')
        return e is not None and abs(float(e) - gev) <= 1e-3 * gev
    return True


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
            if side == 'denominator' and d.get('denominator_class') and \
                    d['denominator_class'] not in ((target.get('centrality') or {}).get('classes') or []):
                problems.append(f"{d.get('value_type')}: denominator_class {d['denominator_class']} is not a "
                                f"class of the linked denominator record")
            if system and not side_matches(system, target):
                cs = target.get('collision_system') or {}
                problems.append(f"{d.get('value_type')}: {side} system {system} but the linked "
                                f"record is {cs.get('reaction')} at {cs.get('energy_gev')} GeV")
        if d.get('ratio_type') in ('centrality_ratio', 'rapidity_ratio') and \
                d.get('numerator_system') != d.get('denominator_system'):
            problems.append(f"{d.get('value_type')}: {d.get('ratio_type')} across systems "
                            f"({d.get('numerator_system')} / {d.get('denominator_system')})")
    for p in problems:
        print('  FAIL', p)
    print(f'  {checked} links checked; {len(unlinked)} ratio records without links '
          f'(composite ratios such as double ratios)')
    n_pass = 1 if not problems else 0
    print(f'\n{n_pass}/1 passed')
    sys.exit(0 if n_pass else 1)


if __name__ == '__main__':
    main()
