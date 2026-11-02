"""Unified species / charged-hadron resolution shared by 2D and 3D pipelines.

Both the boost-invariant (QVectorData, species-axis indices) and explicit-eta
(QGridData / MultiSpeciesGrid, PDG codes) sim formats must resolve a curated
record's ``particle_id.species`` string to the SAME logical selection and the
SAME charge-symmetry factor, so event/charged/identified treatment is identical
across simulation formats (the only legitimate difference is the analytic
eta-cut in 2D vs the explicit eta grid in 3D).

``resolve_species`` returns ``(sim_indices, pdgs, charge_factor)`` where:
  * sim_indices : list of QVectorData species-axis indices (or None = all)
  * pdgs        : list of PDG codes to sum for a 3D MultiSpeciesGrid
  * charge_factor : multiplier for charge-symmetry (sims store positives only).
                   Applied to Q-vectors; self-correlation terms use charge_factor**2
                   (see pipeline 2D prepares and 3D select).
"""

# Charged-stable PDG set used to estimate "charged hadrons" from identified
# species when no explicit combined charged Q-vector is supplied.  Weak decays
# (Sigma, Xi) are excluded, matching the historical 2D convention.
CHARGED_PDGS = [211, 321, 2212]

# species string -> (2D sim species-axis indices, 3D PDG codes, charge factor)
_SPECIES_TABLE = {
    'charged':            ([0, 1, 2], CHARGED_PDGS, 2.0),
    'pi+ + pi-':          ([0], [211, -211], 2.0),
    # Neutral pions and kaons by isospin: pi0 ~ (pi+ + pi-)/2 ~ pi+ and
    # K0_S ~ (K+ + K-)/2 ~ K+ (charge-symmetric matter; K0_S = (K0 + K0bar)/2).
    # An explicit pi0 (111) or K0_S (310) axis wins (resolve_species_on_axis).
    'pi0':                ([0], [211], 1.0),
    'pi+ + pi- + pi0':    ([0], [211], 3.0),
    'pi+':                ([0], [211], 1.0),
    'pi-':                ([0], [-211], 1.0),
    'K+ + K-':            ([1], [321, -321], 2.0),
    'K+':                 ([1], [321], 1.0),
    'K-':                 ([1], [-321], 1.0),
    'K0_S':               ([1], [321], 1.0),
    'K0s':                ([1], [321], 1.0),
    'p + pbar':           ([2], [2212, -2212], 2.0),
    'p':                  ([2], [2212], 1.0),
    'pbar':               ([2], [-2212], 1.0),
    'Sigma+ + Sigmabar-': ([3], [3212, -3212], 2.0),
    'Xi- + Xibar+':       ([4], [3312, -3312], 2.0),
    # (Until November 2026 'Lambda' read the proton axis x 2; Lambda comes
    # from a container with a PDG axis or grid, _PDG_SETS.)
    'PHI < K- K+ >':      ([1], [321, -321], 2.0),
    'K*(892)0 < PI- K+ > + AK*(892)0 < PI+ K- >': ([1], [321, -321], 2.0),
}


def resolve_species(species_str):
    """Resolve a record species string to (sim_indices, pdgs, charge_factor).

    Returns (None, None, None) if the string is unknown (caller should warn
    and fall back to zero/combined).
    """
    if species_str is None:
        return None, None, None
    s = str(species_str)
    entry = _SPECIES_TABLE.get(s)
    if entry is not None:
        return entry
    for key, val in _SPECIES_TABLE.items():
        if key.upper() == s.upper():
            return val
    return None, None, None


# Exact-charge species sets for containers that store both charges on
# separate species axes (e.g. the A=96 isobar campaign with 13 PDGs).
# Values are PDG codes; charge symmetry is NOT applied when both charges
# are present.
_PDG_SETS = {
    'charged':              [211, -211, 321, -321, 2212, -2212],
    'pi+ + pi-':            [211, -211],
    'pi+':                  [211],
    'pi-':                  [-211],
    'k+ + k-':              [321, -321],
    'k+':                   [321],
    'k-':                   [-321],
    'p + pbar':             [2212, -2212],
    'p':                    [2212],
    'pbar':                 [-2212],
    'lambda':               [3122],
    'lambdabar':            [-3122],
    'lambda + lambdabar':   [3122, -3122],
    'xi-':                  [3312],
    'xibar+':               [-3312],
    'xi- + xibar+':         [3312, -3312],
    'omega-':               [3334],
    'omegabar+':            [-3334],
    'omega- + omegabar+':   [3334, -3334],
    'phi':                  [333],
    'phi < k- k+ >':        [333],
}

# Isospin estimates on a PDG axis: species -> (explicit PDG, charged PDGs to
# sum, factor on their sum).  The explicit particle is used when the axis has
# it; otherwise the charged partners, (pi+ + pi-)/2 etc.
_ISOSPIN = {
    'pi0':             (111, [211, -211], 0.5),
    'pi+ + pi- + pi0': (None, [211, -211], 1.5),
    'k0_s':            (310, [321, -321], 0.5),
    'k0s':             (310, [321, -321], 0.5),
}


def resolve_species_on_axis(species_str, pdg_ids):
    """Resolve a species string against a container's PDG-labelled axis.

    Returns ``(indices, charge_factor, found)``.  Exact charges are used
    when both signs are present; the legacy charge-symmetry factor is only
    applied when the axis stores one charge (positives-only).

    Falls back to :func:`resolve_species` when ``pdg_ids`` is None.
    """
    if pdg_ids is None:
        idx, _pdgs, cf = resolve_species(species_str)
        return idx, cf, idx is not None
    if species_str is None:
        return None, 1.0, False

    pdg_ids = [int(p) for p in pdg_ids]
    pos = {p: i for i, p in enumerate(pdg_ids)}

    key = str(species_str).strip().lower()
    if key in _ISOSPIN:
        explicit, partners, factor = _ISOSPIN[key]
        if explicit is not None and explicit in pos:
            return [pos[explicit]], 1.0, True
        have = [p for p in partners if p in pos]
        if not have:
            return [], 1.0, False
        # positives (or negatives) only: one partner stands for both
        return [pos[p] for p in have], factor * len(partners) / len(have), True
    wanted = _PDG_SETS.get(key)
    if wanted is None:
        # Try the legacy table to accept aliases, then map its PDGs.
        _idx, pdgs, _cf = resolve_species(species_str)
        wanted = list(pdgs) if pdgs is not None else None
    if not wanted:
        return None, 1.0, False

    indices = [pos[p] for p in wanted if p in pos]
    if not indices and not any(p < 0 for p in pdg_ids):
        # A positives-only axis (no antiparticle anywhere): a negative
        # species by charge symmetry from its conjugate (pbar from p).
        conj = [-p for p in wanted if -p in pos]
        if conj and len(conj) == len(wanted):
            return [pos[p] for p in conj], 1.0, True
    if not indices:
        return [], 1.0, False

    # Exact match: every requested charge is present -> no scaling.
    if len(indices) == len(wanted):
        return indices, 1.0, True

    # Some charge is missing (positives-only axis): fall back to the
    # legacy charge-symmetry factor.
    _idx, _pdgs, cf = resolve_species(species_str)
    cf = 1.0 if cf is None else float(cf)
    return indices, cf, True


def charged_species_indices(pdg_ids):
    """Indices of charged stable species (pi, K, p) on a PDG axis.

    Returns ``(indices, charge_factor)``.  With both charges present the
    factor is 1; with positives-only it is 2 (legacy convention).
    """
    if pdg_ids is None:
        return [0, 1, 2], 2.0
    pdg_ids = [int(p) for p in pdg_ids]
    pos = {p: i for i, p in enumerate(pdg_ids)}
    pos_pdgs = [211, 321, 2212]
    neg_pdgs = [-211, -321, -2212]
    have_pos = [p for p in pos_pdgs if p in pos]
    have_neg = [p for p in neg_pdgs if p in pos]
    indices = [pos[p] for p in have_pos + have_neg]
    if not indices:
        return [], 1.0
    cf = 1.0 if have_neg else 2.0
    return indices, cf


def species_known(species_str):
    """True if the pipeline can select this species: from the species table
    (charge-symmetric sims) or the exact-charge PDG sets."""
    key = str(species_str).strip().lower()
    return (resolve_species(species_str)[0] is not None
            or key in _PDG_SETS or key in _ISOSPIN)


def is_charged(species_str):
    """True for the 'charged' aggregate (explicit charged Q-vector OR sum)."""
    return str(species_str).lower() == 'charged'


def species_unavailable(particle_id, container):
    """Why ``container`` cannot provide the record's species, or None.

    The record is then skipped (run_pipeline, the toy harness) instead of
    computing zero (a species absent from the axis) or another species (a
    3D grid without species falls back to its combined charged grid).
    'charged' and no species are always available.
    """
    species = (particle_id or {}).get('species') if isinstance(particle_id, dict) else particle_id
    if species is None or is_charged(species):
        return None
    grids = getattr(container, 'grids', None)
    if grids is not None:                                  # MultiSpeciesGrid
        idx, _cf, found = resolve_species_on_axis(species, list(grids))
        return None if found and idx else f"species {species!r} not among the grids' PDGs {sorted(grids)}"
    if hasattr(container, 'eta_edges'):                    # plain QGridData: one combined grid
        return f"species {species!r}: the 3D container has no species grids"
    pdg_ids = getattr(container, 'pdg_ids', None)
    idx, _cf, found = resolve_species_on_axis(species, pdg_ids)
    n = getattr(container, 'n_species', None)
    if not found or not idx or (n is not None and max(idx) >= n):
        return f"species {species!r} not on the container's species axis"
    return None
