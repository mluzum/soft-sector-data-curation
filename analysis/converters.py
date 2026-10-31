"""Converters: arbitrary simulation output → pipeline Q-vector containers.

The analysis pipeline consumes two container types (see
``analysis/qvector.py``):

QVectorData  ("2D", boost-invariant)
    Per-event, per-species Q_n accumulated on a pT grid.  Particles are
    assumed distributed flat in RAPIDITY over |y| < rap_max; η acceptance
    is applied later through the analytic Jacobian dy/dη.  Harmonic slot 0
    stores counts (Q₀), slots 1..n store Σ e^{inφ}.

QGridData  ("3D", explicit pseudorapidity)
    Per-event Q_n on a rectangular (η, pT) grid; no Jacobian needed and no
    boost-invariance assumption.  Unit-weighted particles ⇒ dQ0 = Q0 and
    disjoint-η subevents have zero self-correlation.

The universal input format is a particle list per event — any simulation
code can dump one:

    events = [dict(pt=..., phi=..., species_idx=..., y=... or eta=...), ...]

`particles_to_qvector_data` / `particles_to_qgrid_data` bin these into the
containers with optional ΣpT first-moment storage (needed for mean-pT).
"""
from __future__ import annotations

import numpy as np

from analysis.qvector import QVectorData, QGridData


# ── kinematic helpers ───────────────────────────────────────────────────────

def y_to_eta(y, pt, mass):
    """Exact rapidity → pseudorapidity.

    E = mT cosh y, pz = mT sinh y, |p| = sqrt(pT² + pz²)  ⇒  tanhη = pz/|p|.
    """
    y = np.asarray(y, dtype=float)
    pt = np.asarray(pt, dtype=float)
    mT = np.sqrt(mass ** 2 + pt ** 2)
    pz = mT * np.sinh(y)
    p_tot = np.sqrt(pt ** 2 + pz ** 2)
    return np.arctanh(np.clip(pz / p_tot, -1 + 1e-12, 1 - 1e-12))


def eta_to_y(eta, pt, mass):
    """Exact pseudorapidity → rapidity.

    pz = pT sinhη, E = sqrt(pT² cosh²η + m²)  ⇒  tanh y = pz/E.
    """
    eta = np.asarray(eta, dtype=float)
    pt = np.asarray(pt, dtype=float)
    pz = pt * np.sinh(eta)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)
    return np.arctanh(np.clip(pz / E, -1 + 1e-12, 1 - 1e-12))


# ── particle-list normalisation ─────────────────────────────────────────────

def _normalize_events(events, masses):
    """Yield list of dicts with keys pt, phi, species_idx, y, eta, mass.

    Accepts per-event dicts of ragged arrays.  If only 'y' given, η is
    derived exactly; if only 'eta', y is derived.
    """
    out = []
    for ev in events:
        pt = np.asarray(ev['pt'], dtype=float)
        s_idx = np.asarray(ev['species_idx'])
        if 'mass' in ev and ev['mass'] is not None:
            mass = np.asarray(ev['mass'], dtype=float)
        elif masses is not None:
            mass = np.asarray(masses)[s_idx]
        else:
            raise ValueError("provide masses= or per-event 'mass'")
        if ev.get('y') is not None:
            y = np.asarray(ev['y'], dtype=float)
            eta = y_to_eta(y, pt, mass)
        elif ev.get('eta') is not None:
            eta = np.asarray(ev['eta'], dtype=float)
            y = eta_to_y(eta, pt, mass)
        else:
            raise ValueError("each event needs 'y' or 'eta'")
        out.append(dict(y=y, eta=eta, pt=pt,
                        phi=np.asarray(ev['phi'], dtype=float),
                        species_idx=s_idx, mass=mass))
    return out


# ── converters ──────────────────────────────────────────────────────────────

def particles_to_qvector_data(events, pt_edges, rap_max, species_names,
                               masses=None, n_harm=6, nsamples=None,
                               event_weights=None,
                               with_sum_pt=True, with_sum_pt2=False,
                               with_sum_mT=False):
    """Bin particle lists into a 2D-style QVectorData (flat-y semantics)."""
    events = _normalize_events(
        events, np.asarray(masses, dtype=float) if masses is not None
        else np.full(len(species_names), np.nan))
    pt_edges = np.asarray(pt_edges, dtype=float)
    n_pt = len(pt_edges) - 1
    n_sp = len(species_names)
    Qn = np.zeros((len(events), n_sp, n_harm, n_pt), dtype=complex)
    sum_pt = np.zeros((len(events), n_sp, n_pt)) if with_sum_pt else None
    sum_pt2 = np.zeros((len(events), n_sp, n_pt)) if with_sum_pt2 else None
    sum_mT = np.zeros((len(events), n_sp, n_pt)) if with_sum_mT else None

    for e, ev in enumerate(events):
        pi = np.clip(np.searchsorted(pt_edges, ev['pt'], side='right') - 1,
                     0, n_pt - 1)
        s = ev['species_idx']
        np.add.at(Qn[e][:, 0, :], (s, pi), 1.0)
        for h in range(1, n_harm):
            np.add.at(Qn[e][:, h, :], (s, pi), np.exp(1j * h * ev['phi']))
        if with_sum_pt:
            np.add.at(sum_pt[e], (s, pi), ev['pt'])
        if with_sum_pt2:
            np.add.at(sum_pt2[e], (s, pi), ev['pt'] ** 2)
        if with_sum_mT:
            mT = np.sqrt(ev['mass'] ** 2 + ev['pt'] ** 2)
            np.add.at(sum_mT[e], (s, pi), mT)

    return QVectorData(
        Qn=Qn, pt_bin_edges=pt_edges,
        masses=(np.asarray(masses, dtype=float) if masses is not None
                else np.ones(n_sp)),
        species_names=list(species_names),
        nsamples=(np.ones(len(events)) if nsamples is None
                  else np.asarray(nsamples, dtype=float)),
        event_weights=(np.asarray(event_weights, dtype=float)
                       if event_weights is not None else None),
        sim_type="2d", rap_max=float(rap_max), sum_pt=sum_pt,
        sum_pt2=sum_pt2, sum_mT=sum_mT)


def particles_to_qgrid_data(events, eta_edges, pt_edges, n_harm=6,
                            nsamples=None, cent_mult=None, with_sum_pt=True,
                            masses=None, with_sum_pt2=False, with_sum_ET=False,
                            event_weights=None):
    """Bin particle lists into a 3D-style QGridData (explicit η grid).

    masses is only needed when events provide 'y' instead of 'eta'.
    sum_ET accumulates the exact transverse energy E_T = E/coshη
    (= sqrt(m² + pT² cosh²η)/coshη), required for dET/dη.
    """
    events = _normalize_events(events, masses)
    eta_edges = np.asarray(eta_edges, dtype=float)
    pt_edges = np.asarray(pt_edges, dtype=float)
    n_eta, n_pt = len(eta_edges) - 1, len(pt_edges) - 1
    Qn = np.zeros((len(events), n_eta, n_pt, n_harm), dtype=complex)
    sum_pt = np.zeros((len(events), n_eta, n_pt)) if with_sum_pt else None
    sum_pt2 = np.zeros((len(events), n_eta, n_pt)) if with_sum_pt2 else None
    sum_ET = np.zeros((len(events), n_eta, n_pt)) if with_sum_ET else None

    for e, ev in enumerate(events):
        ei = np.clip(np.searchsorted(eta_edges, ev['eta'], side='right') - 1,
                     0, n_eta - 1)
        pi = np.clip(np.searchsorted(pt_edges, ev['pt'], side='right') - 1,
                     0, n_pt - 1)
        cells = Qn[e].reshape(-1, n_harm)
        flat = ei * n_pt + pi
        np.add.at(cells[:, 0], flat, 1.0)
        for h in range(1, n_harm):
            np.add.at(cells[:, h], flat, np.exp(1j * h * ev['phi']))
        if with_sum_pt:
            np.add.at(sum_pt[e].reshape(-1), flat, ev['pt'])
        if with_sum_pt2:
            np.add.at(sum_pt2[e].reshape(-1), flat, ev['pt'] ** 2)
        if with_sum_ET:
            ch = np.cosh(ev['eta'])
            ET = np.sqrt(ev['mass'] ** 2
                         + (ev['pt'] * ch) ** 2) / ch
            np.add.at(sum_ET[e].reshape(-1), flat, ET)

    if cent_mult is None:
        cent_mult = np.array([len(ev['pt']) for ev in events], dtype=float)
    return QGridData(Qn=Qn, eta_edges=eta_edges, pt_edges=pt_edges,
                      cent_mult=np.asarray(cent_mult, dtype=float),
                      nsamples=(np.ones(len(events)) if nsamples is None
                                else np.asarray(nsamples, dtype=float)),
                      event_weights=(np.asarray(event_weights, dtype=float)
                                     if event_weights is not None else None),
                      sim_type="3d", sum_pt=sum_pt, sum_pt2=sum_pt2,
                      sum_ET=sum_ET)
