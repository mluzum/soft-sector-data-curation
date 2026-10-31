from dataclasses import dataclass, field
import numpy as np

from analysis.species import resolve_species, is_charged, CHARGED_PDGS


def _snap_searchsorted(edges, lo, hi, tol=1e-9):
    """searchsorted with float-noise tolerance on query edges.

    Returns [i_lo, i_hi] for the half-open slice edges[i_lo:i_hi].
    A query within ``tol`` ABOVE edges[k] snaps down to k (otherwise a
    +1e-15 repr-noise query would wrongly include one extra cell).
    Queries below an edge, or exactly on one, keep standard left semantics.
    """
    ei = np.searchsorted(edges, [lo, hi])
    for j, q in enumerate((lo, hi)):
        if ei[j] > 0 and 0.0 <= q - edges[ei[j] - 1] <= tol:
            ei[j] -= 1
    return ei


@dataclass
class QVectorData:
    Qn: np.ndarray
    pt_bin_edges: np.ndarray
    masses: np.ndarray
    species_names: list
    nsamples: np.ndarray
    event_weights: np.ndarray | None = None
    sim_type: str = "2d"
    rap_max: float = 1.0
    # Optional PDG id per species-axis entry.  When present, species
    # selection resolves exact charges (e.g. pi- vs pi+) instead of the
    # legacy positives-only charge-symmetry convention.
    pdg_ids: list | None = None
    # Optional per-event multiplicity for centrality sorting.  Needed when
    # the container includes events with zero particles (drawn-sample
    # denominator) or when the centrality estimator differs from the
    # charged-hadron sum.
    cent_mult: np.ndarray | None = None
    # Optional first pT moment: Σ 1·pT per (event, species, pT bin).
    # Shape must match Qn[:, :, 0, :] (counts slot).  Used by mean-pT.
    sum_pt: np.ndarray | None = None
    # Optional second pT moment Σ 1·pT² (same shape).  Enables pT-fluctuation
    # observables via (ΣpT)² − ΣpT² pair identities.
    sum_pt2: np.ndarray | None = None
    # Optional Σ 1·mT with mT=sqrt(m²+pT²) (same shape).  η-independent, so
    # exactly storable in flat-y containers; dET/dη = <ΣmT·(1/coshη)> with
    # the analytic window factor applied by the prepare step.
    sum_mT: np.ndarray | None = None

    def __post_init__(self):
        assert self.Qn.ndim == 4, f"Qn must be 4D [event, species, harmonic, pt], got shape {self.Qn.shape}"
        n_pt = self.Qn.shape[3]
        if len(self.pt_bin_edges) == n_pt:
            self.pt_bin_edges = np.concatenate([[0.0], self.pt_bin_edges])
        assert len(self.pt_bin_edges) == n_pt + 1, \
            f"Qn pt bins ({n_pt}) != pt_bin_edges-1 ({len(self.pt_bin_edges) - 1})"
        assert self.Qn.shape[1] == len(self.masses) == len(self.species_names), \
            f"species mismatch: Qn={self.Qn.shape[1]}, masses={len(self.masses)}, names={len(self.species_names)}"
        assert self.Qn.shape[0] == len(self.nsamples)
        if self.pdg_ids is not None:
            assert len(self.pdg_ids) == self.Qn.shape[1], \
                f"pdg_ids length {len(self.pdg_ids)} != species {self.Qn.shape[1]}"
        if self.cent_mult is not None:
            assert len(self.cent_mult) == self.Qn.shape[0], \
                f"cent_mult length {len(self.cent_mult)} != events {self.Qn.shape[0]}"
        if self.sum_pt is not None:
            expected = (self.Qn.shape[0], self.Qn.shape[1], self.Qn.shape[3])
            assert self.sum_pt.shape == expected, \
                f"sum_pt shape {self.sum_pt.shape} != (event, species, pt) {expected}"
        if self.sum_pt2 is not None:
            expected = (self.Qn.shape[0], self.Qn.shape[1], self.Qn.shape[3])
            assert self.sum_pt2.shape == expected, \
                f"sum_pt2 shape {self.sum_pt2.shape} != (event, species, pt) {expected}"
        if self.sum_mT is not None:
            expected = (self.Qn.shape[0], self.Qn.shape[1], self.Qn.shape[3])
            assert self.sum_mT.shape == expected, \
                f"sum_mT shape {self.sum_mT.shape} != (event, species, pt) {expected}"

    @property
    def n_events(self) -> int:
        return self.Qn.shape[0]

    @property
    def n_species(self) -> int:
        return self.Qn.shape[1]

    @property
    def n_harmonics(self) -> int:
        return self.Qn.shape[2]

    @property
    def n_pt(self) -> int:
        return self.Qn.shape[3]

    @property
    def pt_centers(self) -> np.ndarray:
        return (self.pt_bin_edges[:-1] + self.pt_bin_edges[1:]) / 2

    @classmethod
    def from_2d_npy(cls, qn_path, nsamples_path, pt_bin_edges,
                    masses=None, species_names=None, delta_f_index=0,
                    rap_max=1.0):
        Qn_full = np.load(qn_path)
        Ns = np.load(nsamples_path)
        if Qn_full.ndim == 5:
            Qn = Qn_full[delta_f_index].copy()
            ns = Ns[delta_f_index].copy()
        elif Qn_full.ndim == 4:
            Qn = Qn_full.copy()
            ns = Ns.copy()
        else:
            raise ValueError(f"Unexpected Qn shape: {Qn_full.shape}")
        if masses is None:
            masses = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
        if species_names is None:
            species_names = ['pion', 'kaon', 'proton', 'Sigma', 'Xi']
        pt_bin_edges = np.asarray(pt_bin_edges, dtype=np.float64)
        return cls(Qn=Qn, pt_bin_edges=pt_bin_edges,
                   masses=np.asarray(masses), species_names=list(species_names),
                   nsamples=ns.astype(np.float64), sim_type="2d",
                   rap_max=rap_max)

    def save(self, path):
        kw = dict(Qn=self.Qn, pt_bin_edges=self.pt_bin_edges,
                  masses=self.masses, nsamples=self.nsamples,
                  sim_type=self.sim_type, rap_max=self.rap_max)
        if self.pdg_ids is not None:
            kw['pdg_ids'] = np.asarray(self.pdg_ids, dtype=int)
        if self.cent_mult is not None:
            kw['cent_mult'] = self.cent_mult
        if self.event_weights is not None:
            kw['event_weights'] = self.event_weights
        if self.sum_pt is not None:
            kw['sum_pt'] = self.sum_pt
        if self.sum_pt2 is not None:
            kw['sum_pt2'] = self.sum_pt2
        if self.sum_mT is not None:
            kw['sum_mT'] = self.sum_mT
        np.savez_compressed(path, **kw)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        kw = dict(Qn=data['Qn'], pt_bin_edges=data['pt_bin_edges'],
                  masses=data['masses'], nsamples=data['nsamples'],
                  sim_type=str(data['sim_type']),
                  rap_max=float(data.get('rap_max', 1.0)))
        if 'species_names' in data:
            kw['species_names'] = list(data['species_names'])
        if 'pdg_ids' in data:
            kw['pdg_ids'] = [int(x) for x in data['pdg_ids']]
        cm = data.get('cent_mult')
        if cm is not None:
            kw['cent_mult'] = cm
        ew = data.get('event_weights')
        if ew is not None:
            kw['event_weights'] = ew
        sp = data.get('sum_pt')
        if sp is not None:
            kw['sum_pt'] = sp
        sp2 = data.get('sum_pt2')
        if sp2 is not None:
            kw['sum_pt2'] = sp2
        smt = data.get('sum_mT')
        if smt is not None:
            kw['sum_mT'] = smt
        obj = cls(**kw)
        if 'species_names' not in data:
            obj.species_names = [f'species_{i}' for i in range(obj.n_species)]
        return obj


@dataclass
class QGridData:
    """Finely-binned 3D Q-vectors on a rectangular (η, pT) grid.

    The grid preserves per-cell resolution so that η/pT differential
    observables can be computed by slicing.  Each cell stores the
    first-power Q-vector summed over all particles in that cell
    (or all charged hadrons in that cell for particle_9999 data).

    Fields
    ------
    Qn : (n_events, n_eta, n_pt, n_harm) complex
        First-power Q-vectors per grid cell.
    eta_edges : (n_eta + 1,) float
        Pseudorapidity bin edges.
    pt_edges : (n_pt + 1,) float
        Transverse momentum bin edges.
    cent_mult : (n_events,) float
        Per-hydro-event centrality multiplicity (per-oversample).
    nsamples : (n_events,) float
        Oversample count per hydro event.
    event_weights : (n_events,) float or None
        Optional per-(hydro-)event importance/reweighting vector for
        weighted event averages.  Multiplied with the centrality-bin weight
        by the pipeline (combine_weights).  None ⇒ equal weighting (ones).
    sum_pt : (n_events, n_eta, n_pt) float or None
        Optional first pT moment Σ 1·pT per cell (for mean-pT).
    """
    Qn: np.ndarray
    eta_edges: np.ndarray
    pt_edges: np.ndarray
    cent_mult: np.ndarray
    nsamples: np.ndarray
    sim_type: str = "3d"
    event_weights: np.ndarray | None = None
    sum_pt: np.ndarray | None = None
    sum_pt2: np.ndarray | None = None
    sum_ET: np.ndarray | None = None
    positives_only: bool = True
    """Whether the grid stores positive charges only (charge symmetry x2
    needed for 'charged' aggregates).  True for toy/converted positives-only
    grids (default, preserves historical behaviour).  False for JETSCAPE3D
    particle_9999 grids, which already contain all charged hadrons
    (verified Aug 2026: unscaled 9999 yields match the JETSCAPE Bayesian
    model totals to <1% median; applying x2 overestimates by exactly 2x)."""

    def __post_init__(self):
        assert self.Qn.ndim == 4, \
            f"Qn must be 4D [event, eta, pt, harmonic], got shape {self.Qn.shape}"
        assert self.Qn.shape[1] == len(self.eta_edges) - 1, \
            f"eta dim mismatch"
        assert self.Qn.shape[2] == len(self.pt_edges) - 1, \
            f"pt dim mismatch"
        assert self.Qn.shape[0] == len(self.cent_mult) == len(self.nsamples), \
            f"event dim mismatch"
        if self.event_weights is not None:
            assert self.event_weights.shape == (self.Qn.shape[0],), \
                f"event_weights shape {self.event_weights.shape} != (n_events,) {self.Qn.shape[0]}"
        cell_shape = self.Qn.shape[:3]
        for nm in ('sum_pt', 'sum_pt2', 'sum_ET'):
            v = getattr(self, nm)
            if v is not None:
                assert v.shape == cell_shape, \
                    f"{nm} shape {v.shape} != Qn event/eta/pt dims {cell_shape}"

    @property
    def n_events(self) -> int:
        return self.Qn.shape[0]

    @property
    def n_harmonics(self) -> int:
        return self.Qn.shape[3]

    def __add__(self, other):
        """Additive closure: Q-vectors and moments are additive across
        species on identical grids.  cent_mult/nsamples taken from self."""
        if not isinstance(other, QGridData):
            return NotImplemented
        assert np.allclose(self.eta_edges, other.eta_edges) and \
            np.allclose(self.pt_edges, other.pt_edges), "grid mismatch"

        def _sum(a, b):
            if a is None or b is None:
                return None
            return a + b

        out = QGridData(Qn=self.Qn + other.Qn,
                        eta_edges=self.eta_edges,
                        pt_edges=self.pt_edges,
                        cent_mult=self.cent_mult.copy(),
                        nsamples=self.nsamples.copy(),
                        sim_type=self.sim_type,
                        event_weights=(self.event_weights.copy()
                                       if self.event_weights is not None
                                       else None))
        for f in ('sum_pt', 'sum_pt2', 'sum_ET'):
            v = _sum(getattr(self, f), getattr(other, f))
            if v is not None:
                setattr(out, f, v)
        out.mass = getattr(self, 'mass', None)
        return out

    def acceptance(self, eta_min, eta_max, pt_min=0.0, pt_max=1e9):
        """Sum Q-vectors over an η-pT acceptance window.

        Returns (n_events, n_harm) complex — the integrated Q-vectors.
        For unit-weighted particles, dQ0 = Q0 = col 0's real part.
        Grids flagged ``pt_integrated`` ignore the pT bounds (single bin).

        Query edges within 1e-9 of a grid edge snap to it: curated bin
        edges sometimes carry float repr noise (e.g. -4.800000000000001
        for -4.8), and without snapping searchsorted includes/excludes a
        whole 0.1 cell (found Aug 2026: alternating ±50% dN/deta bins).
        """
        ei = _snap_searchsorted(self.eta_edges, eta_min, eta_max)
        if getattr(self, 'pt_integrated', False):
            pi = (0, self.Qn.shape[2])
        else:
            pi = _snap_searchsorted(self.pt_edges, pt_min, pt_max)
        return self.Qn[:, ei[0]:ei[1], pi[0]:pi[1], :].sum(axis=(1, 2))

    def select(self, particle_id):
        """Species selection for a plain (combined) QGridData.

        A positives-only non-species-resolved grid (e.g. toy grids, 2D-style
        inputs) represents total charged hadrons via charge symmetry, so
        'charged' applies x2 (cf=2.0).  Grids already holding all charged
        hadrons (positives_only=False, e.g. JETSCAPE3D particle_9999) are
        returned unscaled (cf=1.0).  Any identified request falls back to
        the combined grid (cf=1.0) — the same limitation as 2D when no
        species axis is available.  Per-species resolution requires a
        MultiSpeciesGrid (load_3d_species_grids).
        """
        species = (particle_id or {}).get('species') if isinstance(
            particle_id, dict) else particle_id
        if is_charged(species) and getattr(self, 'positives_only', True):
            out = _scale_grid(self, 2.0)
            out.charge_factor = 2.0
            return out
        self.charge_factor = 1.0
        return self


# ── species-resolved grid collections ────────────────────────────────────────

PDG_NAMES = {211: 'pi+', -211: 'pi-', 321: 'K+', -321: 'K-',
             2212: 'p', -2212: 'pbar', 3122: 'Lambda', -3122: 'Lambdabar',
             3312: 'Xi-', -3312: 'Xibar+', 3334: 'Omega-', 333: 'phi'}


def _scale_grid(g, cf):
    """Return a copy of ``g`` (QGridData) with Q-vectors and moments × cf.

    Used to apply charge-symmetry (sims store positives only): a record
    wanting 'p + pbar' from a 'p' grid scales by 2.  The returned grid
    carries ``charge_factor = cf`` for the downstream v_n{2} self-correlation
    (which must use cf**2, matching the 2D prepares' inline CF**2).
    """
    if cf == 1.0:
        return g
    out = QGridData(Qn=g.Qn * cf,
                    eta_edges=g.eta_edges, pt_edges=g.pt_edges,
                    cent_mult=g.cent_mult.copy(),
                    nsamples=g.nsamples.copy(),
                    event_weights=(g.event_weights.copy()
                                   if g.event_weights is not None else None),
                    sim_type=g.sim_type,
                    positives_only=False)
    out.charge_factor = cf
    for nm in ('sum_pt', 'sum_pt2', 'sum_ET'):
        v = getattr(g, nm, None)
        if v is not None:
            setattr(out, nm, v * cf)
    return out


class MultiSpeciesGrid:
    """Collection of per-PDG QGridData objects sharing one grid geometry.

    Observables that need species selection call ``select(species_string)``
    which returns a plain (possibly summed) QGridData; everything downstream
    is unchanged.  The combined charged grid (particle_9999 equivalent) is
    exposed via ``combined`` and used when selection is None/unknown.
    """

    def __init__(self, grids, combined=None):
        # Normalize PDG keys to ints so selection matches resolve_species()
        # (which returns integer PDG codes) regardless of how callers built
        # the dict (string keys are common from YAML-derived specs).
        self.grids = {int(k): v for k, v in grids.items()}
        self.combined = combined          # QGridData or None

    @property
    def n_events(self):
        g = next(iter(self.grids.values())) if self.grids else self.combined
        return g.n_events

    @property
    def nsamples(self):
        return (self.combined or next(iter(self.grids.values()))).nsamples

    @property
    def cent_mult(self):
        return (self.combined or next(iter(self.grids.values()))).cent_mult

    @property
    def event_weights(self):
        g = self.combined if self.combined is not None \
            else next(iter(self.grids.values()), None)
        return getattr(g, 'event_weights', None) if g is not None else None

    def acceptance(self, *a, **k):        # delegated for pipeline centrality
        return (self.combined or next(iter(self.grids.values()))).acceptance(*a, **k)

    def select(self, particle_id):
        """Return a QGridData for a record's ``particle_id``.

        Resolution mirrors the 2D (species-axis) selection via the shared
        ``analysis.species.resolve_species`` table, so charged/identified
        treatment is identical between sim formats.  Charge-symmetry (sims
        store positives only) is applied by scaling the returned Q-vectors by
        the charge factor and tagging ``out.charge_factor``; the 3D v_n{2}
        prepares multiply ``dQ0 = Q0`` by that factor to keep the
        self-correlation subtraction consistent (cf**2), exactly as the 2D
        prepares do inline.

        Rule (user-specified): explicit combined charged Q-vector wins for
        'charged'; otherwise sum the available charged-stable PDG grids
        (pi+, K+, p) and apply charge-symmetry x2.
        """
        import warnings
        species = (particle_id or {}).get('species') if isinstance(
            particle_id, dict) else particle_id
        sim_indices, pdgs, cf = resolve_species(species)
        base = (self.combined if self.combined is not None
                else (next(iter(self.grids.values())) if self.grids else None))
        if base is None:
            raise ValueError("MultiSpeciesGrid has no grids and no combined")

        # No species requested -> combined if present, else estimate charged
        if species is None:
            if self.combined is not None:
                out = self.combined
                out.charge_factor = 1.0
                return out
            desired = [p for p in CHARGED_PDGS if p in self.grids]
            if not desired:
                return base
            out = self._sum_pdgs(desired)
            out = _scale_grid(out, 2.0)
            out.charge_factor = 2.0
            out.mass = None
            return out

        # 'charged' aggregate: explicit combined first, else pi/K/p x2 fallback.
        # The combined (particle_9999-equivalent) grid stores positives only
        # for positives-only inputs (mirroring CHARGE_FACTOR['charged']); for
        # JETSCAPE3D particle_9999 grids (positives_only=False, already all
        # charged) no scaling is applied.
        if is_charged(species):
            if self.combined is not None:
                if getattr(self.combined, 'positives_only', True):
                    out = _scale_grid(self.combined, 2.0)
                    out.charge_factor = 2.0
                else:
                    out = self.combined
                    out.charge_factor = 1.0
                out.mass = None
                return out
            desired = [p for p in CHARGED_PDGS if p in self.grids]
            if not desired:
                warnings.warn("no charged species grids available; "
                              "returning base grid")
                return base
            out = self._sum_pdgs(desired)
            out = _scale_grid(out, 2.0)
            out.charge_factor = 2.0
            out.mass = None
            return out

        # Specific identified species
        if pdgs is None:
            warnings.warn(f"Species '{species}' unknown; using combined")
            out = base
            out.charge_factor = 1.0
            return out
        available_pdgs = [p for p in pdgs if p in self.grids]
        if not available_pdgs:
            warnings.warn(f"Species '{species}' not in sim grids; using combined")
            out = base
            out.charge_factor = 1.0
            return out
        out = self._sum_pdgs(available_pdgs)
        out = _scale_grid(out, cf)
        out.charge_factor = cf
        out.mass = SPECIES_MASS.get(pdgs[0], None)
        return out

    def _sum_pdgs(self, pdgs):
        """Sum the listed PDG grids into one QGridData."""
        out = self.grids[pdgs[0]]
        for p in pdgs[1:]:
            out = out + self.grids[p]
        return out


SPECIES_MASS = {211: 0.13957, -211: 0.13957, 321: 0.49368, -321: 0.49368,
                2212: 0.93827, -2212: 0.93827, 3122: 1.11568,
                -3122: 1.11568, 3312: 1.32171, -3312: 1.32171,
                3334: 1.67245, 333: 1.01946}
