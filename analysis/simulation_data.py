"""Load 2D/3D simulation data and apply kinematic cuts.

2D (boost-invariant): Qns_*.npy files contain per-rapidity Q-vectors.
  Apply any η window via Jacobian: cutQn(Qn, etarange).
  3D: expects per-event particle list; not yet implemented.

Reference: qnscript.py and ReadQns.ipynb of the JETSCAPE 2D campaigns
           (js-sims-bayes)
"""

import os
import glob
import numpy as np
from scipy import integrate
import scipy.stats

# ── Physical constants ──────────────────────────────────────────────────────

masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132],
                    dtype=float)
ptcuts = np.array([
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95,
    2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3., 3.2, 3.4, 3.6, 3.8, 4., 10.
], dtype=float)
ptlist = (ptcuts[1:] + ptcuts[:-1]) / 2.0

# Charged hadrons: sum over species 0 (pi), 1 (K), 2 (p) ×2
CHARGED_SPECIES = [0, 1, 2]

# Default delta_f model (0 = Grad)
DEFAULT_DELTA_F = 0

# ── Pseudorapidity Jacobian ─────────────────────────────────────────────────

def jac(eta, mass, pt):
    """Jacobian dη/dy for a given mass and pT at pseudorapidity η."""
    return pt * np.cosh(eta) / np.sqrt(mass * mass + pt * pt * np.cosh(eta) ** 2)


def intjac(etarange, mass, pt):
    """Integrate the Jacobian over etarange = [min, max]."""
    return integrate.quad(jac, etarange[0], etarange[1], args=(mass, pt))[0]


def jaclist(etarange):
    """Matrix of Jacobian integrals [species, pt_bin] for the given η range."""
    jl = np.zeros((len(masslist), len(ptlist)))
    for i in range(len(ptlist)):
        for j in range(len(masslist)):
            jl[j, i] = intjac(etarange, masslist[j], ptlist[i])
    return jl


# ── 3D simulation data (HDF5) ────────────────────────────────────────────────

# JETSCAPE3D grid: η × pT, same for all collision systems and events
# η: uniform 0.1 spacing, 141 bins, [-7.10, 7.10]
# NOTE (Aug 2026): built by integer division, NOT np.arange: arange's
# accumulated float error puts stored edges ~1e-14 BELOW nominal values
# (e.g. -5.400000000000006), so searchsorted() on nominal query edges
# (record bins like -5.4) skips a cell and every grid-aligned acceptance
# window shifts +0.1 in eta (forward dN/deta off by up to ~20%, eta
# asymmetry ~12%).  i/10 rounds each edge independently to the same
# double the user writes literally, so queries land exactly.
ETA_EDGES_3D = np.array([i / 10 for i in range(-71, 72)])

# pT: first bin 0–0.15 GeV, then 0.1-wide bins up to 4.15 GeV
# Same integer-division construction for the same reason.
PT_EDGES_3D = np.array(
    [0.00] + [(15 + 10 * i) / 100 for i in range(40)] + [4.15]
)


def load_events_3d_hdf5(h5_path, pt_bin_edges=None, meas_eta_ranges=None,
                          cent_eta_ranges=None, pt_min=0.0, pt_max=None,
                          n_harmonics=6, verbose=True, event_weights=None):
    """Read 3D HDF5 and sum Q-vectors over specified η-pT acceptance.

    Each spvn_results_N group in the HDF5 is one hydro event with per-cell
    Q-vectors from the hadronic afterburner.  The 5781 rows per event
    correspond to the 141 × 41 grid defined by ETA_EDGES_3D and
    PT_EDGES_3D.  Each row's η and pT values are the bin-averaged
    coordinates of particles in that cell; they are NOT row-major ordered.

    Column layout of particle_9999_pTeta_distribution.dat:
      col 0: η    (bin-averaged pseudorapidity)
      col 1: pT   (bin-averaged transverse momentum)
      col 2: N    = particle yield (Q0, per-afterburner-event)
      col 3: E_T
      cols 4-5, 6-7, ..., 18-19: Qn_real, Qn_imag for n=1..8

    Parameters
    ----------
    h5_path : str
    pt_bin_edges : ignored (hardwired PT_EDGES_3D used instead)
    meas_eta_ranges : dict of {name: (eta_min, eta_max)}
        η acceptance for each subevent, e.g. {'A': (1,3), 'B': (-3,-1)}
    cent_eta_ranges : list of (eta_min, eta_max)
        η ranges for centrality estimator, e.g. [(-3.9,-3.0), (3.0,3.9)]
    pt_min, pt_max : float — pT range
    n_harmonics : int — number of harmonics including Q0 (default 6)
    verbose : bool

    The pTeta_distribution stores per-oversample values (one afterburner run).
    The vndata file stores the per-event oversample count (N_hadronic_events)
    in its summary row, column 1.  This function reads that count, multiplies
    all measured quantities (yield and Q-vectors) to convert per-oversample →
    per-hydro-event accumulated values, and stores the count for future use
    in self-correlation observables (v_n{4}, etc.).

    Returns
    -------
    dict with:
        Qn_{name} : (n_events, n_harmonics) complex per subevent — accumulated
        cent_mult : (n_events,) float — per-oversample centrality multiplicity
        nsamples : (n_events,) float — oversample count per hydro event
        n_events : int
        n_subevents : int
    """
    import h5py

    pt_min = pt_min if pt_min is not None else 0.0
    pt_max = pt_max if pt_max is not None else 1e9

    # Precompute η,pT grid masks for each acceptance window:
    # which (η_idx, pT_idx) pairs belong to each window?
    n_eta = len(ETA_EDGES_3D) - 1
    n_pt = len(PT_EDGES_3D) - 1

    def _grid_mask(eta_min, eta_max, pT_min, pT_max):
        """Return boolean array (n_eta, n_pt) for given acceptance."""
        mask = np.ones((n_eta, n_pt), dtype=bool)
        eta_ok = (ETA_EDGES_3D[:-1] < eta_max) & (ETA_EDGES_3D[1:] > eta_min)
        pT_ok = (PT_EDGES_3D[:-1] < pT_max) & (PT_EDGES_3D[1:] > pT_min)
        mask[~eta_ok] = False
        mask[:, ~pT_ok] = False
        return mask

    cent_mask = np.zeros((n_eta, n_pt), dtype=bool)
    for emin, emax in (cent_eta_ranges or []):
        cent_mask |= _grid_mask(emin, emax, pt_min, pt_max)

    meas_masks = {}
    for name, (emin, emax) in (meas_eta_ranges or {}).items():
        meas_masks[name] = _grid_mask(emin, emax, pt_min, pt_max)

    f = h5py.File(h5_path, 'r')
    group_names = sorted(
        [k for k in f.keys() if k.startswith('spvn_results_')],
        key=lambda x: int(x.split('_')[-1]))

    if verbose:
        print(f"  HDF5: {len(group_names)} spvn_results groups (events)")

    # Column indices
    Q0_col = 2
    ET_col = 3
    qn_real = lambda n: 4 + 2 * (n - 1)
    qn_imag = lambda n: 5 + 2 * (n - 1)

    n_events = len(group_names)
    n_subevents = len(meas_eta_ranges or {})
    result = {}
    for name in (meas_eta_ranges or {}):
        result[f'Qn_{name}'] = np.zeros((n_events, n_harmonics), dtype=np.complex128)
    if not result:
        result['Qn_A'] = np.zeros((n_events, n_harmonics), dtype=np.complex128)
    result['cent_mult'] = np.zeros(n_events)
    result['nsamples'] = np.zeros(n_events, dtype=np.float64)

    max_harm = min(n_harmonics - 1, 8)

    n_good = 0
    for ie, gname in enumerate(group_names):
        g = f[gname]

        # Read oversample count from vndata summary row, column 1
        # (reference script reads [-1, 2]; actual data stores it in [-1, 1])
        vndata = g.get('particle_9999_vndata_eta_-0.5_0.5.dat')
        if vndata is None:
            continue
        nevents = float(vndata[:][-1, 1])

        if 'particle_9999_pTeta_distribution.dat' not in g:
            continue

        data = g['particle_9999_pTeta_distribution.dat'][:]  # (5781, 20)

        # Digitize each row to its grid cell
        eta_idx = np.digitize(data[:, 0], ETA_EDGES_3D) - 1
        pt_idx = np.digitize(data[:, 1], PT_EDGES_3D) - 1

        # Boolean mask of rows that map to valid grid cells
        in_range = (eta_idx >= 0) & (eta_idx < n_eta) \
                   & (pt_idx >= 0) & (pt_idx < n_pt)
        valid_ei = eta_idx[in_range]
        valid_pi = pt_idx[in_range]

        # Centrality: compute from per-oversample data BEFORE multiplying
        row_ok = np.zeros(len(data), dtype=bool)
        row_ok[in_range] = cent_mask[valid_ei, valid_pi]
        result['cent_mult'][ie] = data[row_ok, Q0_col].sum()

        # Convert per-oversample → accumulated (per-hydro-event)
        data[:, 2:] *= nevents

        # Subevents: Q0 + higher harmonics (accumulated values)
        for name in meas_eta_ranges:
            row_ok[:] = False
            row_ok[in_range] = meas_masks[name][valid_ei, valid_pi]
            result[f'Qn_{name}'][ie, 0] = data[row_ok, Q0_col].sum()
            for ih in range(1, max_harm + 1):
                Qn = data[row_ok, qn_real(ih)] + 1j * data[row_ok, qn_imag(ih)]
                result[f'Qn_{name}'][ie, ih] = Qn.sum()

        result['nsamples'][ie] = nevents
        n_good += 1

    f.close()

    result['n_events'] = n_good
    result['n_subevents'] = n_subevents
    result['event_weights'] = (np.asarray(event_weights, dtype=float)
                               if event_weights is not None else None)
    if verbose:
        print(f"  Loaded {n_good} good events out of {len(group_names)}")
    return result


def load_3d_grid(h5_path, pt_min=0.0, pt_max=1e9, n_harmonics=6, verbose=True,
                event_weights=None):
    h5_path = str(h5_path)
    import h5py
    from analysis.qvector import QGridData

    pt_min = pt_min if pt_min is not None else 0.0
    pt_max = pt_max if pt_max is not None else 1e9

    n_eta = len(ETA_EDGES_3D) - 1
    n_pt = len(PT_EDGES_3D) - 1
    pT_ok = (PT_EDGES_3D[:-1] < pt_max) & (PT_EDGES_3D[1:] > pt_min)

    f = h5py.File(h5_path, 'r')
    group_names = sorted(
        [k for k in f.keys() if k.startswith('spvn_results_')],
        key=lambda x: int(x.split('_')[-1]))
    if verbose:
        print(f"  HDF5: {len(group_names)} spvn_results groups (events)")

    Q0_col = 2
    ET_col = 3
    qn_real = lambda n: 4 + 2 * (n - 1)
    qn_imag = lambda n: 5 + 2 * (n - 1)
    max_harm = min(n_harmonics - 1, 8)

    # Single pass: accumulate into grid per event
    rows = []
    for gname in group_names:
        g = f[gname]
        vndata = g.get('particle_9999_vndata_eta_-0.5_0.5.dat')
        if vndata is None:
            continue
        nevs = float(vndata[:][-1, 1])
        if 'particle_9999_pTeta_distribution.dat' not in g:
            continue
        d = g['particle_9999_pTeta_distribution.dat'][:]
        rows.append((d, nevs))

    n_good = len(rows)
    if verbose:
        print(f"  Pre-scan: {n_good} valid events")

    Qn = np.zeros((n_good, n_eta, n_pt, n_harmonics), dtype=np.complex128)
    sum_et = np.zeros((n_good, n_eta, n_pt))
    cmult = np.zeros(n_good)
    nsamps = np.zeros(n_good)

    for ie, (data, nevents) in enumerate(rows):
        eta_idx = np.digitize(data[:, 0], ETA_EDGES_3D) - 1
        pt_idx = np.digitize(data[:, 1], PT_EDGES_3D) - 1
        ok = (eta_idx >= 0) & (eta_idx < n_eta) & (pt_idx >= 0) & (pt_idx < n_pt) & pT_ok[pt_idx]
        cmult[ie] = float(data[ok, Q0_col].sum())

        data[:, 2:] *= nevents
        ei = eta_idx[ok]
        pi = pt_idx[ok]
        np.add.at(Qn[ie], (ei, pi, np.zeros(ei.size, int)), data[ok, Q0_col])
        np.add.at(sum_et[ie], (ei, pi), data[ok, ET_col])
        for ih in range(1, max_harm + 1):
            qv = data[ok, qn_real(ih)] + 1j * data[ok, qn_imag(ih)]
            np.add.at(Qn[ie], (ei, pi, np.full(ei.size, ih, int)), qv)
        nsamps[ie] = nevents

    f.close()
    if verbose:
        print(f"  Loaded {n_good} events, grid={Qn.shape}")
    out = QGridData(Qn=Qn, eta_edges=ETA_EDGES_3D.copy(), pt_edges=PT_EDGES_3D.copy(),
                    cent_mult=cmult, nsamples=nsamps,
                    event_weights=(np.asarray(event_weights, dtype=float)
                                   if event_weights is not None else None),
                    # particle_9999 already contains all charged hadrons
                    # (verified Aug 2026 against JETSCAPE Bayesian totals);
                    # no charge-symmetry scaling on select().
                    positives_only=False)
    out.sum_ET = sum_et
    return out


def cutQn(Qn, etarange, rap_max=1.0):
    """Convert per-rapidity Qn to per-pseudorapidity for the given η range.

    Qn: array with shape (..., species, harmonic, pt) or (..., species, pt).
        The data is assumed to be per-rapidity, generated in |y| < rap_max.

    Returns: Qn multiplied by the Jacobian integral over etarange,
             divided by 2*rap_max (the full rapidity range width).
    """
    jl = jaclist(etarange)
    Qn = Qn.copy()
    Qn /= 2.0 * rap_max
    Qn *= jl[np.newaxis, :, np.newaxis, :]  # broadcast over event & harmonic dims
    return Qn


def cutpt(Qn, ptrange):
    """Sum Qn over pT bins within [min_pt, max_pt).

    pt_min, pt_max must exactly match values in ptcuts.
    """
    minbin = int(np.argwhere(ptcuts == ptrange[0])[0, 0])
    maxbin = int(np.argwhere(ptcuts == ptrange[1])[0, 0])
    return np.sum(Qn[..., minbin:maxbin], axis=-1)


# ── Centrality ──────────────────────────────────────────────────────────────

def sort_by_Nch(Qn, etarange, ptrange, rap_max=1.0):
    """Sort events by charged-hadron multiplicity in given η, pT range.

    Qn: per-rapidity Q-vectors, shape (events, species, harmonic, pt).
    Returns: array of event indices sorted by multiplicity (highest first).
    """
    Qn_cut = cutQn(Qn, etarange, rap_max)
    Qn_cut = cutpt(Qn_cut, ptrange)
    Q0 = Qn_cut[:, :, 0].real  # (events, species)
    Q0ch = 2.0 * np.sum(Q0[:, CHARGED_SPECIES], axis=1)
    return np.argsort(Q0ch)[::-1]


def select_centrality(eventlist, centbins):
    """Group event indices by centrality percentiles.

    eventlist: array of event indices sorted by multiplicity (descending).
    centbins: array of (min_cent%, max_cent%), e.g. [[0,5],[5,10],...].

    Returns: list of arrays, each containing event indices in that bin.
    """
    nev = len(eventlist)
    groups = []
    for cmin, cmax in centbins:
        lo = int(round(nev * cmin / 100.0))
        hi = int(round(nev * cmax / 100.0))
        groups.append(eventlist[lo:hi])
    return groups


def multiplicity_CDF(Q0, M):
    """Cumulative fraction of events with Poisson-fluctuated multiplicity ≤ M."""
    return np.mean([scipy.stats.poisson.cdf(M, q) for q in Q0])


def find_cdf_crossing(Q0, target, M_min=0, M_max=10000):
    """Binary search: smallest M such that multiplicity_CDF(Q0, M) ≥ target."""
    left, right = M_min, M_max
    while left < right:
        mid = (left + right) // 2
        if multiplicity_CDF(Q0, mid) < target:
            left = mid + 1
        else:
            right = mid
    return left


def centrality_weights(Q0ch, centbins):
    """Poisson-fluctuation centrality weights.

    Each hydro event contributes a fraction to each centrality bin,
    determined by the Poisson distribution of detected particle count.

    Q0ch: smooth charged multiplicity per event.
    centbins: array of (min_cent%, max_cent%).

    Returns: array (events, nbins) of fractional weights.
    """
    nev = len(Q0ch)
    nbins = len(centbins)
    weights = np.zeros((nev, nbins))
    maxQ0 = np.max(Q0ch)
    max_search = int(maxQ0 + 4 * np.sqrt(maxQ0))

    for ibin, (cmin, cmax) in enumerate(centbins):
        centmin = 100.0 - cmax
        centmax = 100.0 - cmin
        Mmin = find_cdf_crossing(Q0ch, centmin / 100.0, 0, max_search)
        Mmax = find_cdf_crossing(Q0ch, centmax / 100.0, 0, max_search)
        weights[:, ibin] = (
            scipy.stats.poisson.cdf(Mmax, Q0ch)
            - scipy.stats.poisson.cdf(Mmin, Q0ch)
        )
    return weights


# ── Observables ─────────────────────────────────────────────────────────────

def integratedvn2(QnQ0dQ0):
    """v_n{2} with proper self-correlation subtraction.

    QnQ0dQ0: array of shape (events, 3) where columns are:
      0: Q_n  = Σ w_i e^{inφ_i}
      1: Q_0  = Σ w_i    (single-weighted multiplicity)
      2: dQ_0 = Σ w_i²   (double-weighted multiplicity, from nested cutQn)

    Returns: v_n{2} float.
    """
    Qn = QnQ0dQ0[:, 0]
    Q0 = QnQ0dQ0[:, 1].astype(float)
    dQ0 = QnQ0dQ0[:, 2].astype(float)
    c2_num = np.sum(np.abs(Qn) ** 2 - dQ0)
    c2_den = np.sum(Q0 * Q0 - dQ0)
    c2 = c2_num / c2_den
    return np.sign(c2) * np.sqrt(np.abs(c2))


def integratedvn2_weighted(QnQ0dQ0weights):
    """v_n{2} with event weights (for Poisson-fluctuation centrality)."""
    Qn = QnQ0dQ0weights[:, 0]
    Q0 = QnQ0dQ0weights[:, 1].astype(float)
    dQ0 = QnQ0dQ0weights[:, 2].astype(float)
    weights = QnQ0dQ0weights[:, 3].astype(float)
    c2_num = np.sum((np.abs(Qn) ** 2 - dQ0) * weights)
    c2_den = np.sum((Q0 * Q0 - dQ0) * weights)
    c2 = c2_num / c2_den
    return np.sign(c2) * np.sqrt(np.abs(c2))


def multiplicity(Q0weights):
    """Average multiplicity, possibly with event weights."""
    Q0 = Q0weights[:, 0].astype(float)
    w = Q0weights[:, 1].astype(float)
    return np.average(Q0, weights=w).astype(float)


# ── Jackknife error ─────────────────────────────────────────────────────────

def jackknifeerror(function, arg):
    """Bias-corrected jackknife estimate and error.

    function: callable that takes arg and returns a float.
    arg: numpy array with first axis over events.
    Returns: [corrected_value, error].
    """
    val = function(arg)
    nevents = len(arg)
    jkdist = np.array([function(np.delete(arg, i, axis=0))
                        for i in range(nevents)])
    error = np.sqrt(nevents - 1) * np.std(jkdist, axis=0)
    corrval = nevents * val - (nevents - 1) * np.mean(jkdist, axis=0)
    return [corrval, error]


# ── Data loading (2D) ───────────────────────────────────────────────────────

def list_design_points(base_dir):
    """Return sorted list of design point indices available in base_dir."""
    files = sorted(glob.glob(os.path.join(base_dir, "Qns_*.npy")))
    return [int(os.path.basename(f).split("_")[1].split(".")[0])
            for f in files]


def load_design_point(base_dir, dp, delta_f=DEFAULT_DELTA_F):
    """Load 2D Q-vectors and SMASH sample counts for one design point.

    Returns:
        Qn:     (events, 5 species, 6 harmonics, 56 pt_bins) — complex, per-rapidity
        nsamples: (events,) — SMASH oversampling count per hydro event
    """
    Qn = np.load(os.path.join(base_dir, f"Qns_{dp}.npy"))
    ns = np.load(os.path.join(base_dir, f"Nsamples_{dp}.npy"))
    # Shape check: (4, events, 5, 6, 56) for Qns, (4, events) for ns
    if Qn.ndim != 5:
        raise ValueError(f"Unexpected Qns shape: {Qn.shape}")
    return Qn[delta_f].copy(), ns[delta_f].copy()


# ── Species-resolved 3D loading ─────────────────────────────────────────────

def load_3d_species_grids(h5_path, n_harmonics=6, verbose=True,
                          event_weights=None):
    """Load identified-particle tables from a JETSCAPE3D HDF5 (legacy 20-col
    writer: 1D eta scan, pT-integrated content, col1 NOT a pT axis).

    Each species becomes a QGridData with a SINGLE wide pT bin
    (pt_integrated=True): dN/deta and identified v_n(eta windows) are exact;
    pT-resolved quantities are meaningless for these tables and the
    species-sensitive prepares return NaN via that flag.
    Returns a MultiSpeciesGrid; ``combined`` is the full-fidelity charged
    9999 grid (irregular eta x pT, all moment slots filled).
    """
    import h5py
    from analysis.qvector import QGridData, MultiSpeciesGrid, SPECIES_MASS

    combined = load_3d_grid(h5_path, n_harmonics=n_harmonics,
                            verbose=verbose, event_weights=event_weights)
    eta_edges = combined.eta_edges
    spec_pt_edges = np.array([0.0, 1e9])          # single integrated bin

    f = h5py.File(str(h5_path), 'r')
    group_names = sorted([k for k in f.keys()
                          if k.startswith('spvn_results_')],
                         key=lambda x: int(x.split('_')[-1]))

    Q0_col = 2
    ET_col = 3
    qn_real = lambda n: 4 + 2 * (n - 1)
    qn_imag = lambda n: 5 + 2 * (n - 1)
    max_harm = min(n_harmonics - 1, 8)

    species_pdgs = sorted({int(k.split('_')[1])
                           for k in f[group_names[0]]
                           if k.startswith('particle_')
                           and k.endswith('_dNdy_pT_0.2_3.dat')})
    if verbose:
        print(f"  species datasets: {species_pdgs}")

    grids = {}
    for pdg in species_pdgs:
        dkey = f'particle_{pdg}_dNdy_pT_0.2_3.dat'
        kept = 0
        Qn = np.zeros((len(group_names), len(eta_edges) - 1, 1, n_harmonics),
                      dtype=np.complex128)
        nsamps = np.zeros(len(group_names))
        for ie, gname in enumerate(group_names):
            g = f[gname]
            if dkey not in g:
                continue
            data = np.asarray(g[dkey][:], dtype=float)
            # Species tables store per-hydro-event DENSITIES (dN/deta,
            # averaged over afterburner runs) on an irregular eta scan --
            # unlike particle_9999, which stores per-oversample counts.
            # Convert densities -> cell counts via local eta spacing so the
            # standard downstream semantics (Sum cells / Delta eta, unit
            # weights) apply unchanged.
            order = np.argsort(data[:, 0])
            data = data[order]
            x = data[:, 0]
            sp = np.empty_like(x)
            if len(x) > 1:
                sp[1:-1] = 0.5 * (x[2:] - x[:-2])
                sp[0] = x[1] - x[0]
                sp[-1] = x[-1] - x[-2]
            else:
                sp[:] = 1.0
            data[:, 2:] *= sp[:, None]

            eta_idx = np.digitize(data[:, 0], eta_edges) - 1
            ok = (eta_idx >= 0) & (eta_idx < len(eta_edges) - 1)
            ei = eta_idx[ok]
            nevents = 1.0
            np.add.at(Qn[kept], (ei, np.zeros(ei.size, int),
                                 np.zeros(ei.size, int)), data[ok, Q0_col])
            for ih in range(1, max_harm + 1):
                qv = data[ok, qn_real(ih)] + 1j * data[ok, qn_imag(ih)]
                np.add.at(Qn[kept],
                          (ei, np.zeros(ei.size, int),
                           np.full(ei.size, ih, int)), qv)
            nsamps[kept] = nevents
            kept += 1

        g_out = QGridData(Qn=Qn[:kept], eta_edges=eta_edges.copy(),
                          pt_edges=spec_pt_edges.copy(),
                          cent_mult=combined.cent_mult[:kept],
                          nsamples=nsamps[:kept])
        g_out.pt_integrated = True
        g_out.mass = SPECIES_MASS.get(pdg)
        grids[pdg] = g_out

    f.close()
    out = MultiSpeciesGrid(grids, combined=combined)
    if verbose:
        print(f"  MultiSpeciesGrid: {len(grids)} species, "
              f"{out.n_events} events")
    return out
