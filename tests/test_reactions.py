"""Collision-system keys of reaction strings (analysis.reactions) and their use
by run_pipeline: sim paths without the Au+Au proxy unless asked.
Run: python3 tests/test_reactions.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analysis.reactions import system_key


def test_spellings():
    cases = {'AU AU --> CHARGED X': 'Au Au', 'AuAu': 'Au Au', 'Au+Au\\implies CHARGED X': 'Au Au',
             'AU + AU': 'Au Au', 'Pb-Pb': 'Pb Pb', 'P PB --> X': 'p Pb', 'D AU --> CHARGED X': 'd Au',
             'DEUT AU': 'd Au', 'd Au': 'd Au', '^3He Au --> CHARGED X': '3He Au', 'HE3 AU --> X': '3He Au',
             'CU AU': 'Cu Au', 'NE NE': 'Ne-Ne', 'Zr+Zr': 'Zr Zr', 'RURU': 'Ru Ru', 'p+p': 'p p',
             'unknown': 'unknown', 'Ru+Ru and Zr+Zr': 'Ru+Ru and Zr+Zr'}
    for r, want in cases.items():
        assert system_key(r) == want, (r, system_key(r), want)


def test_sim_path_no_proxy():
    """A system without a configured simulation gets none (it got the Au+Au
    200 GeV proxy, and 'D AU' records missed the configured d+Au one)."""
    import run_pipeline as rp
    cfg = {'3d': {'Au Au': {'200': 'au'}, 'd Au': {'200': 'dau'}}}
    assert rp._get_sim_path(cfg, '3d', 'D AU --> CHARGED X', '200.0') == 'dau'
    assert rp._get_sim_path(cfg, '3d', 'DEUT AU', '200') == 'dau'
    assert rp._get_sim_path(cfg, '3d', 'CU CU --> X', '200.0') is None
    assert rp._get_sim_path(cfg, '3d', 'CU CU --> X', '200.0', fallback=True) == 'au'


def test_beam_names():
    from analysis.frames import _beam
    assert _beam('3He') == _beam('He3') == _beam('^3He') == _beam('HE')
    assert _beam('d') == _beam('DEUT') and _beam('Pb') == _beam('PB') != _beam('p')


if __name__ == '__main__':
    tests = [n for n in sorted(dir()) if n.startswith('test_')]
    n_pass = 0
    for name in tests:
        try:
            globals()[name]()
            print(f'  PASS  {name}')
            n_pass += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e!r}')
    print(f'{n_pass}/{len(tests)} passed')
    sys.exit(0 if n_pass == len(tests) else 1)
