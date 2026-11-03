from dataclasses import dataclass, field
import numpy as np

from analysis.species import resolve_species, resolve_species_on_axis, is_charged, CHARGED_PDGS


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


def _summed_area(Q):
    """S[e, a, p, h] = sum of Q[e, :a, :p, h], complex128, shape (n, A+1, P+1, H)."""
    n, A, P, H = Q.shape
    S = np.zeros((n, A + 1, P + 1, H), dtype=np.complex128)
    np.cumsum(Q, axis=1, dtype=np.complex128, out=S[:, 1:, 1:, :])
    np.cumsum(S[:, 1:, 1:, :], axis=2, out=S[:, 1:, 1:, :])
    return S


def _overlap_fractions(edges, lo, hi, measure=None):
    """Fraction of each cell [edges[i], edges[i+1]] inside [lo, hi], in the
    variable itself, or in measure(variable) -- e.g. pT**2 for a spectrum,
    whose invariant yield (dN/dpT / pT) is closer to constant within a cell
    than dN/dpT (which vanishes like pT at low pT)."""
    f = measure or (lambda x: x)
    lo_c, hi_c = edges[:-1], edges[1:]
    a, b = np.maximum(lo_c, lo), np.minimum(hi_c, hi)
    num = np.where(b > a, f(np.maximum(b, a)) - f(a), 0.0)
    return np.clip(num / (f(hi_c) - f(lo_c)), 0.0, 1.0)


def _pt_squared(x):
    return np.asarray(x, dtype=float) ** 2


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
        Pseudorapidity bin edges -- or None for a rapidity-binned grid,
        which gives ``y_edges`` (and ``y_mass``) instead.
    y_edges : (n_y + 1,) float or None
        Rapidity bin edges of an identified-species grid binned in y (the
        second axis of Qn is then y).  Exactly one of eta_edges / y_edges.
    y_mass : float or None
        The species mass of a rapidity-binned grid (eta <-> y per pT cell).
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
    y_edges: np.ndarray | None = None
    y_mass: float | None = None
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
        assert (self.eta_edges is None) != (self.y_edges is None), \
            "a 3D grid has exactly one of eta_edges / y_edges"
        if self.y_edges is not None:
            assert self.y_mass is not None and self.y_mass >= 0, \
                "a rapidity-binned grid needs its species mass (y_mass)"
        assert self.Qn.shape[1] == len(self.axis_edges) - 1, \
            f"eta/y dim mismatch"
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

    @property
    def rapidity_binned(self) -> bool:
        return self.y_edges is not None

    @property
    def axis_edges(self) -> np.ndarray:
        """Edges of the second Qn axis: eta, or y for a rapidity-binned grid."""
        return self.y_edges if self.y_edges is not None else self.eta_edges

    def __add__(self, other):
        """Additive closure: Q-vectors and moments are additive across
        species on identical grids.  cent_mult/nsamples taken from self."""
        if not isinstance(other, QGridData):
            return NotImplemented
        if self.rapidity_binned or other.rapidity_binned:
            # A sum of rapidity-binned grids is one grid only for one mass
            # (p + pbar, K+ + K-): cells of different masses at the same y
            # lie at different eta.
            if not (self.rapidity_binned and other.rapidity_binned
                    and abs(self.y_mass - other.y_mass) < 1e-6):
                raise ValueError('cannot sum rapidity-binned grids of different masses '
                                 '(or with an eta-binned grid)')
        assert self.axis_edges.shape == other.axis_edges.shape and \
            np.allclose(self.axis_edges, other.axis_edges) and \
            np.allclose(self.pt_edges, other.pt_edges), "grid mismatch"

        def _sum(a, b):
            if a is None or b is None:
                return None
            return a + b

        out = QGridData(Qn=self.Qn + other.Qn,
                        eta_edges=self.eta_edges,
                        y_edges=self.y_edges, y_mass=self.y_mass,
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

        On a rapidity-binned grid the eta window is mapped per pT cell with
        the grid's mass; whole y cells count by their centre (window_weights).
        """
        if self.rapidity_binned:
            return self._weighted(self.window_weights(eta_min, eta_max, pt_min, pt_max))
        ei = _snap_searchsorted(self.eta_edges, eta_min, eta_max)
        if getattr(self, 'pt_integrated', False):
            pi = (0, self.Qn.shape[2])
        else:
            pi = _snap_searchsorted(self.pt_edges, pt_min, pt_max)
        return self._window_sum(ei[0], ei[1], pi[0], pi[1])

    # Summed-area table of Qn over (axis, pT), complex128 with a zero first
    # row and column: a window sum is four lookups instead of a pass over
    # its cells (a 3D run on 40k O+O events spent ~85% of its time in
    # window sums).  Built lazily, once the direct sums on this Qn have
    # covered a grid's worth of cells (so a grid queried a few times, e.g.
    # a jackknife replicate, never pays for it); kept for this Qn object
    # (a new Qn, as on a masked copy, starts over).  Memory: ~2x the
    # complex64 Qn while the grid lives; set summed_area_tables = False to
    # sum directly.  Window sums from the table agree with direct sums to
    # rounding (~1e-16 of the grid totals; exactly on complex64 storage,
    # and exactly zero in a window without particles).
    summed_area_tables = True

    def _window_sum(self, e0, e1, p0, p1):
        """Sum of Qn over cells [e0:e1, p0:p1] -> (n_events, n_harm) complex128."""
        Q = self.Qn
        n_ev, n_h = Q.shape[0], Q.shape[3]
        # slice semantics: indices beyond the grid clip to it
        e0, e1 = (min(max(int(i), 0), Q.shape[1]) for i in (e0, e1))
        p0, p1 = (min(max(int(i), 0), Q.shape[2]) for i in (p0, p1))
        if e1 <= e0 or p1 <= p0:
            return np.zeros((n_ev, n_h), dtype=np.complex128)
        cache = self.__dict__.get('_sat_cache')
        if cache is None or cache['Qn'] is not Q:
            cache = {'Qn': Q, 'table': None, 'work': 0}
            self._sat_cache = cache
        S = cache['table']
        if S is None and self.summed_area_tables:
            cache['work'] += (e1 - e0) * (p1 - p0)
            if cache['work'] > Q.shape[1] * Q.shape[2]:
                S = cache['table'] = _summed_area(Q)
        if S is not None:
            out = S[:, e1, p1] - S[:, e0, p1] - S[:, e1, p0] + S[:, e0, p0]
            # An event without particles in the window: the corner
            # differences leave a residue ~1e-16 of the cumulative sums in the
            # harmonic slots (complex128 storage), which q Q* - self-pairs in a
            # nearly empty POI bin turned into -7.7e7 (O+O, 5 < pT < 5.5 GeV).
            # The count (slot 0) is exact for integer or dyadic weights; a
            # window whose count is below 1e-13 of the event's total (exactly
            # 0 then, or a rounding residue for other weights) is empty and
            # gets exactly zero, as the direct sum.
            empty = np.abs(out[:, 0]) <= 1e-13 * np.abs(S[:, -1, -1, 0])
            out[empty, :] = 0.0
            return out
        # accumulate in double precision: Qn may be stored as complex64
        return Q[:, e0:e1, p0:p1, :].sum(axis=(1, 2), dtype=np.complex128)

    def _separable_sum(self, fe, fp):
        """Sum of Qn x fe[axis] x fp[pT] -> (n_events, n_harm): the whole cells
        (fraction exactly 1, contiguous) from window sums, the at most two
        partial cells per axis as weighted strips; einsum on the bounding
        box otherwise."""
        def split(f):
            nz = np.nonzero(f)[0]
            if not len(nz):
                return None
            full = np.nonzero(f == 1.0)[0]
            if len(full) and full[-1] - full[0] + 1 != len(full):
                return 'box', nz[0], nz[-1] + 1
            part = [i for i in nz if f[i] != 1.0]
            if len(part) > 2:
                return 'box', nz[0], nz[-1] + 1
            return ('ok', (int(full[0]), int(full[-1]) + 1) if len(full) else (0, 0), part)
        se, sp = split(fe), split(fp)
        n_ev, n_h = self.Qn.shape[0], self.Qn.shape[3]
        if se is None or sp is None:
            return np.zeros((n_ev, n_h), dtype=np.complex128)
        if se[0] == 'box' or sp[0] == 'box':
            nze, nzp = np.nonzero(fe)[0], np.nonzero(fp)[0]
            e0, e1, p0, p1 = nze[0], nze[-1] + 1, nzp[0], nzp[-1] + 1
            return np.einsum('ijkh,j,k->ih', self.Qn[:, e0:e1, p0:p1, :], fe[e0:e1], fp[p0:p1],
                             dtype=np.complex128)
        (e0, e1), pe = se[1], se[2]
        (p0, p1), pp = sp[1], sp[2]
        out = self._window_sum(e0, e1, p0, p1)                 # whole x whole
        for j in pe:                                           # partial eta rows x every pT cell
            out = out + fe[j] * np.einsum('ikh,k->ih', self.Qn[:, j, :, :], fp, dtype=np.complex128)
        for k in pp:                                           # whole eta cells x partial pT cells
            out = out + fp[k] * self._window_sum(e0, e1, k, k + 1)
        return out

    def _weighted(self, W):
        """Sum of Qn over the cells with weights W (n_axis, n_pt) -> (n_events, n_harm),
        on the bounding box of the nonzero weights."""
        rows, cols = np.nonzero(W.any(axis=1))[0], np.nonzero(W.any(axis=0))[0]
        if not len(rows) or not len(cols):
            return np.zeros((self.Qn.shape[0], self.Qn.shape[3]), dtype=np.complex128)
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return np.einsum('ijkh,jk->ih', self.Qn[:, r0:r1, c0:c1, :], W[r0:r1, c0:c1],
                         dtype=np.complex128)

    def pair_gap_fractions(self, gap):
        """F[i, j]: the fraction of particle pairs, one in eta cell i and one in
        cell j (uniform within cells), with |eta_a - eta_b| > gap.  Exact for
        uniform densities: P(|x - y| > g) = 1 - [C(g) - C(-g)], C the CDF of
        x - y for x, y uniform on the two cells.  The diagonal is a fraction
        of the distinct pairs of a cell (self-pairs have Delta eta = 0).
        Eta grids only (a rapidity-binned grid has no eta cells)."""
        if self.rapidity_binned:
            raise ValueError('a pseudorapidity pair gap needs an eta-binned grid')
        e = np.asarray(self.eta_edges, dtype=float)
        a, w = e[:-1], np.diff(e)
        g = abs(float(gap))

        def ramp(u, width):              # int_0^u clip(v, 0, width) dv
            u = np.maximum(u, 0.0)
            return np.where(u <= width, 0.5 * u * u, 0.5 * width * width + width * (u - width))

        def cdf(t):                      # P(x_i - y_j <= t), x in cell i, y in cell j
            ai, wi = a[:, None], w[:, None]
            aj, wj = a[None, :], w[None, :]
            return (ramp(aj + wj + t - ai, wi) - ramp(aj + t - ai, wi)) / (wi * wj)
        F = np.clip(1.0 - (cdf(g) - cdf(-g)), 0.0, 1.0)
        # exactly 0 or 1 where the cells are wholly within or beyond the gap
        # (the CDF differences leave ~1e-16, which q Q* / m M would turn into
        # a finite value without pairs)
        F[F < 1e-12] = 0.0
        F[F > 1.0 - 1e-12] = 1.0
        return F

    def poi_acceptance(self, eta_min, eta_max, pt_min=0.0, pt_max=1e9):
        """Q of a particle of interest in an eta-pT window: acceptance() on
        eta grids; on a rapidity-binned grid the fractional cell weights of
        the mapped window (POI sums enter the POI x reference correlators
        with weight power 1, so fractional weights keep them unbiased;
        whole cells by centre distort the POI pT spectrum at the window
        edges, a few % on v2 for 0.1-wide y cells)."""
        if self.rapidity_binned:
            return self.acceptance_fractional(eta_min, eta_max, pt_min, pt_max)
        return self.acceptance(eta_min, eta_max, pt_min, pt_max)

    def shared(self, poi, ref):
        """Q of this grid's particles in the POI window ``poi`` that are also
        in the reference window ``ref`` (each (eta_lo, eta_hi, pt_lo, pt_hi)):
        the self-pair (diagonal) terms of POI x reference correlations.

        Eta grids: the whole cells common to both windows.  Rapidity-binned
        grids: the POI's fractional cell weights (poi_acceptance) times the
        fraction of each cell inside the reference window -- the expected
        self-pair sum, the reference coming from another (eta-binned) grid.
        Returns (n_events, n_harm)."""
        if not self.rapidity_binned:
            e0, e1 = max(poi[0], ref[0]), min(poi[1], ref[1])
            p0, p1 = max(poi[2], ref[2]), min(poi[3], ref[3])
            if e1 <= e0 or p1 <= p0:
                return np.zeros((self.n_events, self.Qn.shape[3]), dtype=np.complex128)
            return self.acceptance(e0, e1, p0, p1)
        W = self.window_weights(*poi, fractional=True) * self.window_weights(*ref, fractional=True)
        return self._weighted(W)

    def window_weights(self, lo, hi, pt_min=0.0, pt_max=1e9, var='eta', mass=None, fractional=False):
        """Per-cell weights (n_axis, n_pt) of a window lo < var < hi, pt_min < pT < pt_max.

        ``var`` is 'eta' or 'y'; a 'y' window is computed with ``mass``
        (default: the grid's species mass).  Whole cells (``fractional``
        False: unit-weight flow and correlation sums, whose self-correlation
        terms need unit weights): a window in the grid's own variable by the
        snapped edges, as acceptance(); a mapped window by the cell centre.
        Fractional (yields, spectra): overlap fractions, uniform in the axis
        variable and in pT**2.  On a rapidity-binned grid an eta window is
        mapped per pT cell with the grid's mass, y = asinh(pT sinh(eta) / mT);
        a y window of another mass goes through eta.
        """
        ax = self.axis_edges
        n_ax, n_pt = len(ax) - 1, len(self.pt_edges) - 1
        if var == 'y' and mass is None:
            mass = self.y_mass if self.rapidity_binned else getattr(self, 'mass', None)
            if mass is None:
                raise ValueError('a rapidity window needs a species mass')
        direct = (var == 'eta' and not self.rapidity_binned) or \
                 (var == 'y' and self.rapidity_binned and abs(mass - self.y_mass) < 1e-6)
        if getattr(self, 'pt_integrated', False):
            if not direct:
                raise ValueError('mapping eta <-> y needs pT-binned grids')
            fp = np.ones(n_pt)
        elif fractional:
            fp = _overlap_fractions(self.pt_edges, pt_min, pt_max, _pt_squared)
        else:
            pi = _snap_searchsorted(self.pt_edges, pt_min, pt_max)
            fp = np.zeros(n_pt)
            fp[pi[0]:pi[1]] = 1.0

        def axis_weights(a0, a1, snap):
            if fractional:
                return _overlap_fractions(ax, a0, a1)
            w = np.zeros(n_ax)
            if snap:
                i = _snap_searchsorted(ax, a0, a1)
                w[i[0]:i[1]] = 1.0
            else:
                c = 0.5 * (ax[:-1] + ax[1:])
                w[(c >= a0) & (c < a1)] = 1.0
            return w

        if direct:
            return np.outer(axis_weights(lo, hi, True), fp)
        W = np.zeros((n_ax, n_pt))
        for c in np.nonzero(fp)[0]:
            if fractional:      # the centre of the part of the cell inside the window
                pt = 0.5 * (max(self.pt_edges[c], pt_min) + min(self.pt_edges[c + 1], pt_max))
            else:
                pt = 0.5 * (self.pt_edges[c] + self.pt_edges[c + 1])
            if pt <= 0:
                continue
            e0, e1 = lo, hi
            if var == 'y':                                   # y of `mass` -> eta
                mt = np.sqrt(pt ** 2 + mass ** 2)
                e0, e1 = np.arcsinh(mt * np.sinh(e0) / pt), np.arcsinh(mt * np.sinh(e1) / pt)
            if self.rapidity_binned:                         # eta -> y of the grid's mass
                mt = np.sqrt(pt ** 2 + self.y_mass ** 2)
                e0, e1 = np.arcsinh(pt * np.sinh(e0) / mt), np.arcsinh(pt * np.sinh(e1) / mt)
            W[:, c] = fp[c] * axis_weights(e0, e1, False)
        return W

    def acceptance_fractional(self, eta_min, eta_max, pt_min=0.0, pt_max=1e9):
        """Like acceptance(), but cells cut by the window count with their
        overlap fraction: uniform in eta, and in pT**2 (constant invariant
        yield within a cell; meant for spectra).  A window inside one cell
        gets its share of that cell instead of nothing or the whole cell.
        Returns (n_events, n_harm)."""
        if self.rapidity_binned:
            return self._weighted(self.window_weights(eta_min, eta_max, pt_min, pt_max, fractional=True))
        fe = _overlap_fractions(self.eta_edges, eta_min, eta_max)
        if getattr(self, 'pt_integrated', False):
            fp = np.ones(self.Qn.shape[2])
        else:
            fp = _overlap_fractions(self.pt_edges, pt_min, pt_max, _pt_squared)
        return self._separable_sum(fe, fp)

    def acceptance_rapidity(self, y_min, y_max, assumed_mass, pt_min=0.0, pt_max=1e9):
        """Sum Q-vectors over a window in rapidity computed with one assumed
        mass for every particle (e.g. PHOBOS y_pi), within pT bounds.

        In each pT cell (at its centre pT) the window is the eta window
        eta = asinh(mT_a sinh y / pT), mT_a = sqrt(pT^2 + m_a^2); eta cells
        count with their overlap fraction (uniform density within a cell).
        pT cells count with their overlap fraction.  Returns (n_events, n_harm).
        On a rapidity-binned grid of the same mass the window is sliced
        directly; of another mass it goes through eta (window_weights).
        """
        if self.rapidity_binned:
            return self._weighted(self.window_weights(y_min, y_max, pt_min, pt_max, var='y',
                                                      mass=assumed_mass, fractional=True))
        if getattr(self, 'pt_integrated', False):
            raise ValueError('a mass-assumed rapidity window needs pT-binned grids')
        fp = _overlap_fractions(self.pt_edges, pt_min, pt_max, _pt_squared)
        out = np.zeros((self.Qn.shape[0], self.Qn.shape[3]), dtype=complex)
        for c in np.nonzero(fp)[0]:
            # the window at the centre pT of the part of the cell inside [pt_min, pt_max]
            pt = 0.5 * (max(self.pt_edges[c], pt_min) + min(self.pt_edges[c + 1], pt_max))
            if pt <= 0:
                continue
            mta = np.sqrt(pt ** 2 + assumed_mass ** 2)
            e0, e1 = np.arcsinh(mta * np.sinh(y_min) / pt), np.arcsinh(mta * np.sinh(y_max) / pt)
            fe = _overlap_fractions(self.eta_edges, e0, e1)
            out += fp[c] * (self.Qn[:, :, c, :] * fe[None, :, None]).sum(axis=1)
        return out

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
                    eta_edges=g.eta_edges, y_edges=g.y_edges, y_mass=g.y_mass,
                    pt_edges=g.pt_edges,
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

        # Specific identified species: exact charges when the grids hold
        # both (no charge-symmetry factor), else the charge-symmetry factor;
        # isospin estimates for pi0 / K0_S (as on a 2D PDG axis).  (Until
        # November 2026 both charges were summed AND scaled by the factor.)
        # run_pipeline skips records whose species is not among the grids
        # (species_unavailable); the combined fallback remains for callers.
        pdg_list = list(self.grids)
        idx, cf_axis, found = resolve_species_on_axis(species, pdg_list)
        if not found or not idx:
            warnings.warn(f"Species '{species}' not in sim grids; using combined")
            out = base
            out.charge_factor = 1.0
            return out
        sel = [pdg_list[i] for i in idx]
        out = self._sum_pdgs(sel)
        out = _scale_grid(out, cf_axis)
        out.charge_factor = cf_axis
        out.mass = SPECIES_MASS.get(sel[0], None)
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
                3334: 1.67245, -3334: 1.67245, 333: 1.01946,
                111: 0.13498, 310: 0.49761, 2112: 0.93957, -2112: 0.93957, 22: 0.0}
