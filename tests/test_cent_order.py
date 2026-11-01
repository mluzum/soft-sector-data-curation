"""Every public record's centrality classes follow its table (scripts/check_cent_order.py).

Classes are matched to table rows by position: a class list in the wrong
order, or one invented for a table that gives point centralities, mislabels
every value.  Run: python3 tests/test_cent_order.py
"""
import contextlib
import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from check_cent_order import main  # noqa: E402


def test_no_mismatch():
    with contextlib.redirect_stdout(io.StringIO()) as out:
        bad = main()
    print('  ' + out.getvalue().splitlines()[0])
    for m in bad:
        print('  MISMATCH', m)
    assert not bad


if __name__ == '__main__':
    try:
        test_no_mismatch()
        print('  PASS no mismatch\n\n1/1 passed')
    except AssertionError:
        print('\n0/1 passed')
        sys.exit(1)
