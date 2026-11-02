"""Collision-system keys of curated reaction strings.

HEPData reactions spell the initial state many ways ('AU AU', 'Au+Au',
'AuAu', 'Pb-Pb', 'D AU', 'DEUT AU', '^3He Au', 'Au+Au\\implies CHARGED X').
`system_key` maps them to the keys of config/sim_paths.yaml, sim_frames and
centrality anchors ('Au Au', 'd Au', '3He Au', 'p Pb', ...).
"""
import re

_NUCLEI = {'AU': 'Au', 'PB': 'Pb', 'CU': 'Cu', 'XE': 'Xe', 'U': 'U', 'O': 'O', 'NE': 'Ne', 'RU': 'Ru', 'ZR': 'Zr',
           'P': 'p', 'D': 'd', 'DEUT': 'd', 'HE3': '3He', '3HE': '3He', 'HE': '3He', 'AL': 'Al', 'SI': 'Si',
           'C': 'C', 'AR': 'Ar', 'KR': 'Kr', 'IN': 'In', 'CA': 'Ca'}
# keys kept as they were written in configs and sim packages
_KEYS = {('Ne', 'Ne'): 'Ne-Ne'}


def system_key(reaction):
    """'Au Au', 'd Au', '3He Au', 'p Pb', ... for a reaction string, from the
    text before '-->' (or \\implies, \\rightarrow).  Unparsed: unchanged."""
    s = str(reaction)
    init = re.split(r'-->|\\implies|\\rightarrow|=>', s)[0].replace('^', '').replace('$', '').strip()
    toks = [t for t in re.split(r'[\s+\-]+', init) if t]
    if len(toks) == 1:                                   # 'AuAu', 'RURU'
        m = re.fullmatch(r'([A-Z][a-z]?)([A-Z][a-z]?)', toks[0])
        if m:
            toks = [m.group(1), m.group(2)]
        else:
            m = re.fullmatch(r'([A-Z]{1,2})\1', toks[0].upper())
            if m:
                toks = [m.group(1), m.group(1)]
    if len(toks) != 2 or any(t.upper() not in _NUCLEI for t in toks):
        return reaction
    beams = tuple(_NUCLEI[t.upper()] for t in toks)
    return _KEYS.get(beams, ' '.join(beams))
