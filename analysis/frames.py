"""Frames of asymmetric collision systems (p+Pb, d+Au, 3He+Au, Cu+Au).

A record of an asymmetric system states the frame of its eta / y windows
(collision_system.frame: lab | nn_cms), the centre-of-mass rapidity in the
lab (rapidity_shift: y_lab = y_cms + shift) and the beam moving toward +eta
(forward_beam).  A 3D container states its own orientation: the
nucleon-nucleon centre of mass and the beam toward +eta (sim_frames in
config/sim_paths.yaml, or forward_beam on the package's system entry).  The
JETSCAPE 3D p+Pb and d+Au simulations put the light projectile (p, d) at
+eta: dN/deta is larger at negative eta, the nucleus side.

to_container_frame maps every window of a record into the container frame:
eta_c = s (eta - shift), with s = -1 when the +eta beams differ.  For
pseudorapidity the shift is exact for massless particles only (the
rapidity shift of a massive particle at small pT is smaller); windows a
few tenths wide at mid-rapidity are insensitive to it.  Differential bins
keep their row order (bins become bin_ranges).  References declared as
'mirror' or by |eta| segments describe hemispheres of the record frame and
are refused when the mapping shifts or flips.  2D containers are
boost-invariant and need no mapping.
"""
import copy


def record_frame(rec):
    """(shift, forward_beam) of a record that states a frame, else None."""
    cs = rec.get('collision_system') or {}
    if not any(cs.get(k) is not None for k in ('frame', 'rapidity_shift', 'forward_beam')):
        return None
    shift = float(cs.get('rapidity_shift') or 0.0) if cs.get('frame', 'lab') == 'lab' else 0.0
    return shift, cs.get('forward_beam')


def _beam(b):
    """Nucleus name, case- and spelling-insensitive ('3He', 'He3', '^3He';
    'd', 'DEUT'): analysis.reactions."""
    from analysis.reactions import _NUCLEI
    k = str(b or '').strip().replace('^', '')
    return _NUCLEI.get(k.upper(), k).lower()


def to_container_frame(rec, container):
    """(record in the container frame, None), or (None, reason).

    container: {'frame': 'nn_cms', 'forward_beam': 'p'} or None.  A record
    that states no frame is returned unchanged."""
    rf = record_frame(rec)
    if rf is None:
        return rec, None
    shift, beam = rf
    if not container or not container.get('forward_beam') or container.get('frame', 'nn_cms') != 'nn_cms':
        return None, 'the container does not declare its frame (sim_frames / forward_beam)'
    if not beam:
        return None, 'the record does not state its +eta beam (collision_system.forward_beam)'
    sign = 1.0 if _beam(beam) == _beam(container['forward_beam']) else -1.0
    if shift == 0.0 and sign == 1.0:
        return rec, None
    fr = rec.get('flow_reference') or {}
    if isinstance(fr, dict) and (fr.get('mirror') or fr.get('segments') or fr.get('odd')):
        return None, 'a mirrored, segmented or eta-odd reference is defined in the record frame'

    def m(x):
        return round(sign * (float(x) - shift), 6)

    def window(w):
        """{'min', 'max', bins/points/bin_ranges, ...} mapped in place."""
        if not isinstance(w, dict):
            return
        lo, hi = w.get('min'), w.get('max')
        if lo is not None and hi is not None:
            w['min'], w['max'] = sorted((m(lo), m(hi)))
        elif lo is not None or hi is not None:
            raise ValueError('one-sided window')
        if w.get('bins'):
            e = w.pop('bins')
            w['bin_ranges'] = [sorted((m(a), m(b))) for a, b in zip(e[:-1], e[1:])]
        elif w.get('bin_ranges'):
            w['bin_ranges'] = [sorted((m(a), m(b))) for a, b in w['bin_ranges']]
        if w.get('points'):
            w['points'] = [m(p) for p in w['points']]

    def window_list(ws):
        out = []
        for w in ws or []:
            w = dict(w)
            w['min'], w['max'] = sorted((m(w['min']), m(w['max'])))
            out.append(w)
        return out

    r = copy.deepcopy(rec)
    try:
        kr = r.get('kinematic_range') or {}
        window(kr.get('eta'))
        window(kr.get('y'))
        cent = r.get('centrality') or {}
        if cent.get('eta'):
            cent['eta'] = window_list(cent['eta'])
        for sub in (r.get('subevents') or {}).values():
            skr = (sub or {}).get('kinematic_range') or {}
            window(skr.get('eta'))
            window(skr.get('y'))
        if isinstance(fr, dict) and fr.get('eta'):
            r['flow_reference']['eta'] = window_list(fr['eta'])
    except (ValueError, KeyError, TypeError) as e:
        return None, f'window not mappable ({e})'
    r['collision_system'] = dict(r['collision_system'], frame='nn_cms', rapidity_shift=0.0,
                                 forward_beam=container['forward_beam'])
    return r, None
