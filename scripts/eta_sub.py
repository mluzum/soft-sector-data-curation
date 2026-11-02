"""STAR eta-sub event plane as two subevents, for the curation scripts.

The eta-sub method correlates a particle in one TPC half with the event
plane of the other half (a gap of 2 x gap_half between them) and averages
the two halves.  A record states it as subevents (particles: AB,
subevents A and B), which the dispatcher computes as the two-subevent v_n
of the particle against the charged reference in the opposite subevent
(analysis/dispatcher.py: event_plane + particles AB).  A mirrored
flow_reference served eta-differential records (each bin on one side) but
took both halves for a particle spanning eta = 0 (pT-differential and
integrated records).

    from scripts.eta_sub import as_eta_sub
    as_eta_sub(record, gap_half=0.05, eta_max=1.0, ref_pt={'min': 0.15, 'max': 2.0})
"""

OLD_NOTE = 'computed with the scalar-product estimator'
NEW_NOTE = 'computed as the two-subevent v_n (the particle in one half, the charged reference in the other)'


def subevents(gap_half, eta_max, ref_pt):
    return {'A': {'kinematic_range': {'eta': {'min': -eta_max, 'max': -gap_half}, 'pT': dict(ref_pt)}},
            'B': {'kinematic_range': {'eta': {'min': gap_half, 'max': eta_max}, 'pT': dict(ref_pt)}}}


def as_eta_sub(d, gap_half, eta_max, ref_pt):
    """Replace record d's eta-sub flow_reference by the two subevents (in place);
    eta-differential records keep the mirrored reference.  Returns d."""
    if d.get('dimensionality') == 'differential_eta':
        return d
    d.pop('flow_reference', None)
    d['particles'] = 'AB'
    d['eta_gap'] = round(2 * gap_half, 4)
    d['subevents'] = subevents(gap_half, eta_max, ref_pt)
    notes = d.setdefault('notes', {})
    if isinstance(notes.get('value'), str):
        notes['value'] = notes['value'].replace(OLD_NOTE, NEW_NOTE).replace(
            OLD_NOTE.capitalize(), NEW_NOTE.capitalize())
    notes['subevents'] = (f'eta-sub event plane: the particle in one subevent, the plane of the other '
                          f'({gap_half} < |eta| < {eta_max}; reference pT {ref_pt["min"]}-{ref_pt["max"]} GeV/c).')
    return d
