"""Toy Monte-Carlo validation harness.

Generates particle-level events with KNOWN flow / spectra / multiplicity
truth, fills the same containers the production pipeline uses (QVectorData
for 2D-style, QGridData for 3D-style), and provides brute-force reference
estimators that share no code with the analysis machinery.

Physics model
-------------
- Particles are flat in rapidity over |y| < rap_max (matching the 2D
  boost-invariant interpretation of QVectorData).
- Azimuthal distribution per particle:
      dN/dphi ∝ 1 + 2 Σ_n v_n(pT) cos(n(φ − Ψ_n))
  with event-wise random event-plane angles Ψ_n and optional event-by-event
  v_n magnitude fluctuations.
- pT spectra: mT-exponential  dN/dpT ∝ pT·exp(−mT/T)  per species.
- Species layout mirrors the JETSCAPE convention used by SPECIES_INDICES:
  index 0=pion, 1=kaon, 2=proton (3,4 exist but stay empty unless asked).

All truth values are computed analytically from the generation parameters,
or by brute-force sums over the generated particles using exact closed-form
rapidity→pseudorapidity Jacobians:

    y(η) = asinh(pT sinhη / mT),   dy/dη = pT coshη / sqrt(m² + pT² cosh²η)
"""
from __future__ import annotations

import numpy as np

from analysis.qvector import QVectorData, QGridData

# JETSCAPE-compatible species table (subset actually populated by generator)
SPECIES_NAMES = ['pion', 'kaon', 'proton', 'Sigma', 'Xi']
SPECIES_MASSES = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
SPECIES_INDEX = {n: i for i, n in enumerate(SPECIES_NAMES)}

# charged-hadron mix fractions within each species slot (+/- combined)
DEFAULT_MIX = {'pion': 0.83, 'kaon': 0.05, 'proton': 0.12}


# ── closed-form Jacobian helpers ────────────────────────────────────────────

def dy_detadeta_window(pt, mass, eta_min, eta_max):
    """Exact ∫_{ηmin}^{ηmax} dy/dη dη for a particle of given pT, mass."""
    mT = np.sqrt(mass ** 2 + np.asarray(pt) ** 2)
    a = lambda e: np.arcsinh(np.asarray(pt) * np.sinh(np.clip(e, -700, 700))
                             / mT)
    return a(eta_max) - a(eta_min)


def dy_deta_point(pt, mass, eta):
    return pt * np.cosh(eta) / np.sqrt(mass ** 2 + pt ** 2 * np.cosh(eta) ** 2)


# ── flow profiles ───────────────────────────────────────────────────────────

def vn_of_pt(pt, vn_type='const', vn_value=0.05, pt0=None):
    """v_n(pT) shape. 'const' or 'linear' rising to 2×vn_value at 4×pt0."""
    pt_arr = np.atleast_1d(np.asarray(pt, dtype=float))
    if vn_type == 'const':
        out = np.full_like(pt_arr, vn_value)
    elif vn_type == 'linear':
        if pt0 is None:
            raise ValueError("linear profile requires pt0")
        slope = vn_value / max(pt0, 1e-9)
        out = vn_value * 0.5 + slope * pt_arr
    else:
        raise ValueError(f"unknown vn_type {vn_type}")
    return out if np.ndim(pt) else float(out[0])


def _vn_profile_scale(flow_dict):
    """Reference scale of a flow profile (used for ebE fluct. normalisation)."""
    if flow_dict['kind'] == 'const':
        return flow_dict['value']
    return float(vn_of_pt(np.array([flow_dict.get('pt0', 1.0)]),
                          'linear', flow_dict['value'],
                          flow_dict.get('pt0'))[0])


def _sample_phi(rng, vn_funcs, psis, n, harmonics):
    """Per-particle rejection sampling of φ from 1 + 2Σ vₙᵢcos(n(φ−Ψₙ)).

    vn_funcs[k]/psis[k]/harmonics[k]: per-particle v_n(pT) array, event-plane
    angle, and true harmonic number n of entry k.
    """
    out = np.empty(n)
    if n == 0:
        return out
    todo = np.arange(n)
    for _ in range(100):
        m = len(todo)
        if m == 0:
            break
        dphi = rng.uniform(0, 2 * np.pi, m)
        w = np.ones(m)
        max_sum = 0.0
        for h, f, ps in zip(harmonics, vn_funcs, psis):
            if f is None:
                continue
            vh = np.atleast_1d(f)[todo]
            w += 2.0 * vh * np.cos(h * (dphi - ps))
            max_sum += float(np.abs(vh).max())   # v_n may be negative (odd v1)
        bound = 1.0 + 2.0 * max_sum
        acc = rng.uniform(0, bound, m) < w
        out[todo[acc]] = dphi[acc]
        todo = todo[~acc]
    out[todo] = rng.uniform(0, 2 * np.pi, len(todo))  # fallback
    return out % (2 * np.pi)


# ── generator ───────────────────────────────────────────────────────────────

class ToyGenConfig:
    def __init__(self, n_events=2000, mult_mean=300, mult_sigma=40,
                 rap_max=1.0, seed=42,
                 species_mix=None, temp={'pion': 0.22, 'kaon': 0.30,
                                         'proton': 0.35},
                 pt_min=0.05, pt_max=3.0,
                 flow=None, fluctuate=None, pt_rescale_sigma=0.0,
                 eta_shape=None):
        """
        flow : dict harmonic → dict(kind='const'|'linear', value=..., pt0=...)
                e.g. {2: dict(kind='const', value=0.05)}
        fluctuate : dict harmonic → sigma of Gaussian ebE fluctuation of v_n

        NOTE: like the JETSCAPE sim data, generated samples contain
        POSITIVE-charge particles only; experiment-style totals require the
        pipeline's charge_factor (see species_charge_factor).
        """
        self.n_events = n_events
        self.mult_mean = mult_mean
        self.mult_sigma = mult_sigma
        self.rap_max = rap_max
        self.seed = seed
        self.species_mix = dict(DEFAULT_MIX if species_mix is None
                                else species_mix)
        self.temp = temp
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.flow = flow or {}
        self.fluctuate = fluctuate or {}
        # harmonic -> f(eta): v_n(pT, eta) = v_n(pT) * f(eta), e.g. a
        # rapidity-odd directed flow (f odd, possibly negative).
        self.eta_shape = eta_shape or {}
        # Event-wise common pT rescale k_e ~ N(1, sigma) injects genuine
        # event-by-event mean-pT correlations (dynamical pT_fluct signal).
        self.pt_rescale_sigma = float(pt_rescale_sigma)


def generate_events(cfg: ToyGenConfig):
    """Generate events; returns dict with per-event ragged particle arrays.

    Returns dict:
        events : list of dict(y, eta, pt, phi, species_idx, mass)
        truth  : bookkeeping dict (per-event Ψ, input params, ...)
    """
    rng = np.random.RandomState(cfg.seed)

    # species sampling weights from mix (charged sum normalised)
    names = list(cfg.species_mix.keys())
    w_mix = np.array([cfg.species_mix[n] for n in names], dtype=float)
    w_mix /= w_mix.sum()
    sp_names = [SPECIES_INDEX[n] for n in names]
    sp_mass = SPECIES_MASSES[sp_names]
    sp_temp = np.array([cfg.temp.get(n, 0.25) for n in names])

    # harmonics present
    harms = sorted(cfg.flow.keys())

    # per-event Ψ_n
    psi = {h: rng.uniform(0, 2 * np.pi, cfg.n_events) for h in harms}

    # ebE fluctuating magnitudes
    vn_ebe = {}
    for h in harms:
        base = vn_of_pt(np.array([1.0]), cfg.flow[h]['kind'],
                        cfg.flow[h]['value'], cfg.flow[h].get('pt0'))[0]
        if h in cfg.fluctuate:
            vn_ebe[h] = np.clip(
                rng.normal(base, cfg.fluctuate[h], cfg.n_events), 0.0, 0.5)
        else:
            vn_ebe[h] = np.full(cfg.n_events, base)

    events = []
    for e in range(cfg.n_events):
        M = max(int(rng.normal(cfg.mult_mean, cfg.mult_sigma)), 20)
        s_idx = rng.choice(len(names), size=M, p=w_mix)
        pt = np.empty(M)
        for k in range(len(names)):
            sel = s_idx == k
            if not sel.any():
                continue
            m, T = sp_mass[k], sp_temp[k]
            # sample pT ∝ pT exp(-mT/T) on [pt_min, pt_max] via rejection
            lo, hi = cfg.pt_min, cfg.pt_max
            out = []
            need = sel.sum()
            while len(out) < need:
                cand = rng.uniform(lo, hi, int(need * 2) + 32)
                mt = np.sqrt(m ** 2 + cand ** 2)
                pdf = cand * np.exp(-mt / T)
                acc = cand[rng.uniform(0, pdf.max(), cand.size) < pdf]
                out.extend(acc.tolist())
            pt[sel] = np.array(out[:need])

        mass_arr = sp_mass[s_idx]
        y = rng.uniform(-cfg.rap_max, cfg.rap_max, M)
        if cfg.eta_shape:
            _mT = np.sqrt(mass_arr ** 2 + pt ** 2)
            eta_pre = np.arcsinh(_mT * np.sinh(y) / pt)
        vn_funcs, psis = [], []
        for h in harms:
            base_prof = vn_of_pt(pt, cfg.flow[h]['kind'],
                                 cfg.flow[h]['value'], cfg.flow[h].get('pt0'))
            base_at_ref = _vn_profile_scale(cfg.flow[h])
            f = base_prof * (vn_ebe[h][e] / max(base_at_ref, 1e-12))
            if h in cfg.eta_shape:
                f = f * cfg.eta_shape[h](eta_pre)
            vn_funcs.append(f)
            psis.append(psi[h][e])
        phi = _sample_phi(rng, vn_funcs, psis, M, harms)

        if cfg.pt_rescale_sigma > 0:
            k_e = max(rng.normal(1.0, cfg.pt_rescale_sigma), 0.05)
            pt = pt * k_e
            mT = np.sqrt(mass_arr ** 2 + pt ** 2)
            E = mT * np.cosh(y)
            p_tot = np.sqrt(pt ** 2 + (mT * np.sinh(y)) ** 2)
            eta = np.arctanh(np.clip(np.tanh(y) * E / p_tot,
                                     -1 + 1e-12, 1 - 1e-12))

        # tanh y = pz/E and tanh η = pz/|p|  ⇒  tanh η = tanh y / β,
        # β = |p|/E, E = mT cosh y, pz = mT sinh y  (⇒ |η| ≥ |y|)
        mT = np.sqrt(mass_arr ** 2 + pt ** 2)
        E = mT * np.cosh(y)
        p_tot = np.sqrt(pt ** 2 + (mT * np.sinh(y)) ** 2)
        beta = p_tot / E
        eta = np.arctanh(np.clip(np.tanh(y) / beta, -1 + 1e-12, 1 - 1e-12))

        events.append(dict(y=y, eta=eta, pt=pt, phi=phi,
                           species_idx=s_idx, mass=mass_arr))

    truth = dict(psi=psi, vn_ebe=vn_ebe, cfg=cfg,
                 species_names=[SPECIES_NAMES[i] for i in sp_names],
                 masses=sp_mass, temps=sp_temp)
    return events, truth


# ── container fillers ───────────────────────────────────────────────────────

def production_ptcuts():
    """The pT grid the 2D pipeline's _find_pt_slice expects (JETSCAPE)."""
    from analysis.pipeline_dispatch import PTCUTS
    return PTCUTS.copy()


def make_qvector_data(events, truth, n_pt=None, n_harm=6, pt_max=None,
                      with_sum_pt=True):
    """Fill a 2D-style QVectorData via the public converter.

    Uses the production PTCUTS grid by default (the 2D prepares slice on
    that grid historically); pass n_pt for a custom linspace grid.
    """
    cfg = truth['cfg']
    if n_pt is None:
        pt_edges = production_ptcuts()
        pt_edges = pt_edges[pt_edges <= (pt_max or cfg.pt_max)]
        if pt_edges[-1] < (pt_max or cfg.pt_max):
            pt_edges = np.append(pt_edges, pt_max or cfg.pt_max)
    else:
        pt_edges = np.linspace(cfg.pt_min, pt_max or cfg.pt_max, n_pt + 1)
    from analysis.converters import particles_to_qvector_data
    return particles_to_qvector_data(
        events, pt_edges, rap_max=cfg.rap_max,
        species_names=list(SPECIES_NAMES), masses=SPECIES_MASSES,
        n_harm=n_harm, with_sum_pt=with_sum_pt, with_sum_pt2=with_sum_pt,
        with_sum_mT=with_sum_pt)


def make_qgrid_data(events, truth, n_eta=50, n_pt=30, n_harm=6,
                    eta_half=6.0, pt_max=None, with_sum_pt=True):
    """Fill a 3D-style QGridData via the public converter."""
    cfg = truth['cfg']
    eta_edges = np.linspace(-eta_half, eta_half, n_eta + 1)
    pt_edges = np.linspace(cfg.pt_min, pt_max or cfg.pt_max, n_pt + 1)
    from analysis.converters import particles_to_qgrid_data
    cent_mult = np.array([len(ev['pt']) for ev in events], dtype=float)
    return particles_to_qgrid_data(
        events, eta_edges, pt_edges, n_harm=n_harm,
        cent_mult=cent_mult, with_sum_pt=with_sum_pt, with_sum_pt2=with_sum_pt,
        with_sum_ET=with_sum_pt, masses=SPECIES_MASSES)


# ── brute-force reference estimators (no analysis-code reuse) ───────────────

def ref_Qn(events, harm, eta_range=None, pt_range=None, species_sel=None,
           exact_jac_eta=None, rap_max=None):
    """Directly accumulate weighted Q_n from particle lists.

    If exact_jac_eta=(ηmin,ηmax) and rap_max given, each particle gets its
    exact closed-form weight  w = Δy(η window)/ (2·rap_max); otherwise w=1.
    Returns (Qn_complex[n_events], Q0[n_events], dQ0[n_events]).
    """
    Q = np.zeros(len(events), complex)
    Q0 = np.zeros(len(events))
    dQ0 = np.zeros(len(events))
    for e, ev in enumerate(events):
        sel = np.ones(len(ev['pt']), bool)
        if eta_range is not None:
            sel &= (ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
        if pt_range is not None:
            sel &= (ev['pt'] >= pt_range[0]) & (ev['pt'] <= pt_range[1])
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        if not sel.any():
            continue
        w = np.ones(sel.sum())
        if exact_jac_eta is not None and rap_max is not None:
            jac = dy_detadeta_window(ev['pt'][sel], ev['mass'][sel],
                                     exact_jac_eta[0], exact_jac_eta[1])
            w = jac / (2.0 * rap_max)
        ph = ev['phi'][sel]
        Q[e] = np.sum(w * np.exp(1j * harm * ph))
        Q0[e] = np.sum(w)
        dQ0[e] = np.sum(w ** 2)
    return Q, Q0, dQ0


def ref_vn2(events, harm, **kw):
    Q, Q0, dQ0 = ref_Qn(events, harm, **kw)
    c2 = np.sum(np.abs(Q) ** 2 - dQ0) / max(np.sum(Q0 ** 2 - dQ0), 1e-60)
    return np.sign(c2) * np.sqrt(abs(c2))


def analytic_vn2_from_ebe(v_e):
    """Exact v{2} for event-wise magnitudes v_e:  v{2}² = ⟨v_e²⟩."""
    return float(np.sqrt(np.mean(np.asarray(v_e) ** 2)))


def analytic_vn4_from_ebe(v_e):
    """Exact v{4} for event-wise magnitudes:  v{4}⁴ = 2⟨v²⟩² − ⟨v⁴⟩.

    Follows from ⟨2⟩=⟨v_e²⟩ and ⟨4⟩=⟨v_e⁴⟩ when each event has a common
    plane (valid for our generator).  For fixed v this returns v exactly
    (the classic no-fluctuation theorem), providing an analytic reference
    for the 4-particle cumulant chain that brute force cannot resolve
    (signal ~v⁴ is far below MC noise for realistic v).
    """
    v = np.asarray(v_e)
    m2 = np.mean(v ** 2)
    m4 = np.mean(v ** 4)
    val = 2.0 * m2 ** 2 - m4
    return float(np.sign(val) * abs(val) ** 0.25)


def ref_yield(events, eta_range, pt_range, species_sel=None, charge_factor=1.0):
    """⟨dN/dη⟩ in the window: particles counted ONCE if inside the window.

    NOTE: no extra Δy(pT) weight — selecting on η already applies the
    Jacobian; re-weighting would double-count it.
    """
    tot = np.zeros(len(events))
    for e, ev in enumerate(events):
        sel = ((ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
               & (ev['pt'] >= pt_range[0]) & (ev['pt'] <= pt_range[1]))
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        tot[e] = sel.sum()
    delta_eta = eta_range[1] - eta_range[0]
    return tot.mean() * charge_factor / delta_eta


def ref_spectra(events, pt_lo, pt_hi, eta_range, species_sel=None,
                charge_factor=1.0):
    """Invariant yield 1/(2π pT ΔpT) dN/dpT from plain window counts."""
    counts = np.zeros(len(events))
    for e, ev in enumerate(events):
        sel = ((ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
               & (ev['pt'] >= pt_lo) & (ev['pt'] < pt_hi))
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        counts[e] = sel.sum()
    delta_eta = eta_range[1] - eta_range[0]
    pt_c = 0.5 * (pt_lo + pt_hi)
    inv = counts.mean() * charge_factor / (
        2 * np.pi * pt_c * (pt_hi - pt_lo) * delta_eta)
    return inv


def ref_mean_pt(events, eta_range, pt_range, species_sel=None,
                charge_factor=1.0):
    """⟨pT⟩ over particles in the window (plain geometric selection)."""
    num = np.zeros(len(events))
    den = np.zeros(len(events))
    for e, ev in enumerate(events):
        sel = ((ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
               & (ev['pt'] >= pt_range[0]) & (ev['pt'] <= pt_range[1]))
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        num[e] = ev['pt'][sel].sum()
        den[e] = sel.sum()
    return (num.mean() * charge_factor) / max(den.mean(), 1e-30)


def snap_window(edges, lo, hi):
    """Snap a window onto container-bin edges with full coverage.

    Snaps inward to exact container-bin edges (drops straddling bins),
    avoiding straddling-bin double counting when slicing containers.
    """
    edges = np.asarray(edges)
    lo_s = float(edges[np.searchsorted(edges, lo, side='left')])
    hi_s = float(edges[min(np.searchsorted(edges, hi, side='left'),
                           len(edges) - 1)])
    return lo_s, hi_s


def ref_pT_fluct(events, eta_range, pt_range, species_sel=None):
    """Brute-force ALICE correlator from particle lists (exact pairs)."""
    S_pair = 0.0
    denom = 0.0
    pT_all = []
    per_ev = []
    for ev in events:
        sel = ((ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
               & (ev['pt'] >= pt_range[0]) & (ev['pt'] <= pt_range[1]))
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        p = ev['pt'][sel]
        N = len(p)
        if N < 2:
            continue
        per_ev.append(p)
        pT_all.append(p)
        # Σ_{i≠j} (pTi-M)(pTj-M) added after M known; store raw sums instead
    big = np.concatenate(pT_all)
    M = big.mean()
    for p in per_ev:
        d = p - M
        S_pair += float((d.sum() ** 2 - (d ** 2).sum()))
        denom += len(p) * (len(p) - 1)
    C = S_pair / denom
    return float(np.sqrt(max(C, 0.0)) / M)


def ref_dET(events, eta_range, pt_range, species_sel=None,
            charge_factor=1.0):
    """⟨dET/dη⟩ with exact E_T = E/coshη per particle."""
    tot = np.zeros(len(events))
    for e, ev in enumerate(events):
        sel = ((ev['eta'] >= eta_range[0]) & (ev['eta'] <= eta_range[1])
               & (ev['pt'] >= pt_range[0]) & (ev['pt'] <= pt_range[1]))
        if species_sel is not None:
            sel &= np.isin(ev['species_idx'], species_sel)
        if sel.any():
            ch = np.cosh(ev['eta'][sel])
            ET = np.sqrt(ev['mass'][sel] ** 2
                         + (ev['pt'][sel] * ch) ** 2) / ch
            tot[e] = ET.sum()
    return tot.mean() * charge_factor / (eta_range[1] - eta_range[0])


def ref_sp_diff(events, harm, poi_var_bins, ref_eta, fixed_pt=None,
                poi_eta=None):
    """Independent scalar-product differential reference from particles.

    poi_var_bins: list of ('pT'|'eta', lo, hi)
    Per event: mean phasor per POI bin times reference-window Q-vector,
    averaged with the ATLAS/CMS convention:
        < (mean_phasor_poi) * conj(Q_ref) > / sqrt(<|Q_ref|^2>)

    This is the DEFINITION used to check the pipeline, so it carries the
    reference's own auto-correlation (M_ref) and the POI/reference
    diagonal (M_poi for a nested or identical window) exactly as the
    pipeline does.  Both corrections are geometric and cancel when the
    window is disjoint, where they are zero.
    """
    per = []
    for ev in events:
        eta, pt, phi = ev['eta'], ev['pt'], ev['phi']
        rsel = ((eta >= ref_eta[0]) & (eta <= ref_eta[1]))
        if fixed_pt is not None:
            rsel &= (pt >= fixed_pt[0]) & (pt <= fixed_pt[1])
        if rsel.sum() == 0:
            continue
        n_ref = float(rsel.sum())
        q_ref = np.sum(np.exp(1j * harm * phi[rsel]))
        row_q = np.zeros(len(poi_var_bins), complex)
        row_n = np.zeros(len(poi_var_bins))
        row_s = np.zeros(len(poi_var_bins))   # POI n reference shared count
        for b, (var, lo, hi) in enumerate(poi_var_bins):
            if var == 'pT':
                s = (pt >= lo) & (pt < hi)
            else:
                s = (eta >= lo) & (eta <= hi)
            if poi_eta is not None:
                s &= (eta >= poi_eta[0]) & (eta <= poi_eta[1])
            if s.sum() > 0:
                row_q[b] = np.mean(np.exp(1j * harm * phi[s]))
                row_n[b] = s.sum()
                row_s[b] = float((s & rsel).sum())
        per.append((row_q, row_n, q_ref, n_ref, row_s))
    Q = np.stack([x[0] for x in per])
    Nn = np.stack([x[1] for x in per])
    R = np.array([x[2] for x in per])
    Nr = np.array([x[3] for x in per])
    Sh = np.stack([x[4] for x in per])
    # |Q_ref|^2 = v^2 M^2 + M, so M must come out for the estimator to be
    # exact; leaving it in biases v low by sqrt(1 + 1/(v^2 M_ref)).
    den = np.sqrt(max(np.mean(np.abs(R) ** 2) - np.mean(Nr), 0.0))
    out = []
    for b in range(len(poi_var_bins)):
        # numerator carries the POI/reference diagonal: with the per-event
        # mean-phasor normalisation a shared particle contributes 1/N_poi.
        num = np.real(np.mean(Q[:, b] * np.conj(R))) - np.mean(
            Sh[:, b] / np.maximum(Nn[:, b], 1e-9))
        out.append(float(num / max(den, 1e-30)))
    return np.array(out)


def exact_multi_particle_correlator(ph, harm, k):
    """Exact all-distinct k-particle cyclic correlator <k> from angles.

    Sums over all ORDERED distinct k-tuples with the standard sign
    pattern (+,...,+,-,...,-), divided by the falling factorial
    n!/(n-k)! -- matching the normalization of the direct-Q cumulant
    literature (Bilandzic/Snellings/Voloshin).  Feasible for n<~13, k<=6;
    independent ground truth for validating Q-vector implementations.
    """
    from itertools import permutations
    import numpy as _np
    n = len(ph)
    z = _np.exp(1j * harm * ph)
    if k == 2:
        idx = _np.array(list(permutations(range(n), 2)))
        S = _np.sum(z[idx[:, 0]] * _np.conj(z[idx[:, 1]]))
    elif k == 4:
        S = 0.0 + 0.0j
        for t in permutations(range(n), 4):
            S += (z[t[0]] * z[t[1]]
                  * _np.conj(z[t[2]]) * _np.conj(z[t[3]]))
    elif k == 6:
        S = 0.0 + 0.0j
        for t in permutations(range(n), 6):
            S += (z[t[0]] * z[t[1]] * z[t[2]]
                  * _np.conj(z[t[3]]) * _np.conj(z[t[4]])
                  * _np.conj(z[t[5]]))
    else:
        raise ValueError("k must be 2, 4 or 6")
    den = 1.0
    for j in range(k):
        den *= (n - j)
    return float(S.real / den)


def species_charge_factor(species_str):
    """Charge factor the pipeline applies for a record species string.

    The generator produces positive-charge particles only (mirroring the
    sim convention), so references must apply the same factor as the
    pipeline when comparing to experiment-style totals.
    """
    from analysis.pipeline_dispatch import _get_species_selection
    _, cf, found = _get_species_selection({'species': species_str})
    return cf if found else 1.0


# ── production-path driver ──────────────────────────────────────────────────

def _cent_mult_2d(qv, cent_eta_ranges):
    from scripts.compute_flow_vn import compute_charged_mult
    m = np.zeros(qv.n_events)
    for lo, hi in cent_eta_ranges:
        m += compute_charged_mult(qv, lo, hi)
    return m


def _cent_mult_3d(grid, cent_eta_ranges):
    m = np.zeros(grid.n_events)
    for lo, hi in cent_eta_ranges:
        m += grid.acceptance(lo, hi, 0.0, 1e9)[:, 0].real / grid.nsamples
    return m


def _combined_raw_factory(data, cw, entry, n_cent, min_weight=0.01):
    def _combined_raw(mask):
        dm = _mask_dict(data, mask)
        # Excluded bins (total weight below min_weight) emit NaN of the SAME
        # length as a real output so the flat cent-major layout stays aligned
        # for _extract_with_ncent.  Learn that length from the first real bin.
        out = []
        ref_len = None
        for ic in range(n_cent):
            w = cw[mask][:, ic] if cw.ndim == 2 else cw[mask]
            if np.sum(w) < min_weight:
                out.append(None)  # placeholder; filled below
                continue
            val = np.atleast_1d(entry['core_fn'](dm, w))
            if ref_len is None:
                ref_len = len(val)
            out.append(val)
        if ref_len is None:
            ref_len = 1  # all bins excluded (degenerate); shouldn't happen
        out = [np.full(ref_len, np.nan) if v is None else v for v in out]
        return np.concatenate(out)
    return _combined_raw


def _extract_with_ncent(vec, params, n_cent):
    """Select this record's harmonic (+bins) from flat cent-major output.

    Mirrors scripts/run_pipeline.py exactly -- the harness is only useful
    for validating production if it indexes identically.  Layouts:

      - single-row cores (one value per bin, harmonic-independent:
        vn2_ep, vn3_ep, mean_pt, yield_dNch_deta,
        yield_dNch_deta_per_eta, spectra_invariant_yield, dET_deta, sc_mn,
        nac_mn)  ->  idx = b
        (These were previously indexed by (harm-1), which read out of range
        for harm > 1 and silently returned the wrong number.)
      - everything else packs [v_1..v_N] per bin
        -> idx = (harm-1)*n_bins + b
        (vn2_single_diff_pT included: its 2D core used to emit an unused
        h=0 row, h*n_bins + b, which this harness followed while
        run_pipeline read (harm-1)*n_bins + b -- production returned
        v_{n-1}(pT); fixed Nov 2026.)
    """
    vec = np.atleast_1d(np.asarray(vec))
    nb = max(len(params.get('differential_bins') or []), 1)
    h = int(params.get('harmonic', 2))
    key = params.get('_layout_key')
    from analysis.pipeline_dispatch import SINGLE_ROW_KEYS as SINGLE_ROW
    per = len(vec) // max(n_cent, 1)
    out = np.full(n_cent * nb, np.nan)
    for c in range(n_cent):
        base = c * per
        if key in SINGLE_ROW:
            off = 0
        else:
            off = (h - 1) * nb
        for b in range(nb):
            idx = base + off + b
            if 0 <= idx < len(vec):
                out[c * nb + b] = vec[idx]
    return out


def evaluate(record, qv=None, grid=None, with_jackknife=True,
             n_blocks=20, cent_method='poisson', cent_anchor=None,
             min_weight=0.01):
    """Run a synthetic curated record through the production path.

    record → extract_measurement_params → resolve → PIPELINE_DISPATCH
    prepare_data_{2d,3d} → centrality weights → core_fn (per cent class),
    optionally wrapped in the pipeline's block jackknife (with its
    NaN/unphysical guards).

    ``cent_anchor`` lets the harness mirror a sim set that covers only a
    fraction ``f`` of the total cross section (see analysis.weights.resolve_anchor).
    Centrality bins with total weight below ``min_weight`` are excluded
    (NaN), exactly as in compute_flow_vn / run_pipeline.

    Returns (values, errors), each flat arrays ordered cent-major × bin.
    """
    from analysis.dispatcher import extract_measurement_params, resolve
    from analysis.pipeline_dispatch import get_pipeline_entry
    from analysis.weights import (direct_cent_weights_percentile,
                                  poisson_cdf_cent_weights, resolve_anchor,
                                  cent_weights)

    params = extract_measurement_params(record)
    key = resolve(record)['compute']
    params['_layout_key'] = key
    entry = get_pipeline_entry(key)
    if entry is None:
        raise ValueError(f"no pipeline entry for {key}")

    if qv is not None:
        data = entry['prepare_data_2d'](qv, params)
        n_events = qv.n_events
        mult = _cent_mult_2d(qv, params['cent_eta_ranges'])
    elif grid is not None:
        # Apply charge-symmetry / species selection once, up front, so the
        # 3D prepares read grid.charge_factor consistently with run_pipeline.
        from analysis.pipeline_dispatch import container_for_prepare
        grid = container_for_prepare(grid, params, entry)
        data = entry['prepare_data_3d'](grid, params)
        n_events = grid.n_events
        mult = _cent_mult_3d(grid, params['cent_eta_ranges'])
    else:
        raise ValueError("provide qv or grid")

    anchor = resolve_anchor(cent_anchor if cent_anchor is not None else 1.0,
                            n_events)
    edges = params['cent_edges'] or [(0, 100)]
    # Listed order is preserved (helper sorts internally as needed).
    if cent_method == 'simple':
        cw = cent_weights('simple', mult, edges, n_ref=anchor['n_ref'])
    else:
        cw = cent_weights('poisson', mult, edges,
                          efficiency=1.0,
                          anchor_cent=anchor['anchor_cent'])

    n_cent = len(edges)
    _combined_raw = _combined_raw_factory(data, cw, entry, n_cent, min_weight)

    if not with_jackknife:
        raw = _combined_raw(np.ones(n_events, bool))
        vals = _extract_with_ncent(raw, params, n_cent)
        return vals, np.zeros_like(vals)

    from scripts.run_pipeline import run_block_jackknife
    corr, errors, _cov = run_block_jackknife(_combined_raw, n_events,
                                             n_blocks=n_blocks)
    return (_extract_with_ncent(corr, params, n_cent),
            _extract_with_ncent(errors, params, n_cent))


def _mask_dict(d, mask):
    """Apply boolean event mask to numpy arrays inside the data dict."""
    import copy as _copy
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k == 'qv' and hasattr(v, 'Qn'):
                mv = _copy.copy(v)
                mv.Qn = v.Qn[mask]
                mv.nsamples = v.nsamples[mask]
                if getattr(v, 'sum_pt', None) is not None:
                    mv.sum_pt = v.sum_pt[mask]
                out[k] = mv
            else:
                out[k] = _mask_dict(v, mask)
        return out
    if isinstance(d, list):
        return [_mask_dict(x, mask) for x in d]
    if isinstance(d, np.ndarray) and d.shape[:1] == (len(mask),):
        return d[mask]
    return d


# ── synthetic curated-record builders (drive real extract/resolve code) ─────

def record_flow_vn2(eta=(-1, 1), pt=(0.2, 2.0), cent_classes=('0-10%',
                   '10-40%', '40-80%'), cent_eta=((-3.9, -3.0), (3.0, 3.9)),
                   harmonic=2, method='cumulant', order=2):
    return dict(
        observable_type='flow_vn', method=method,
        cumulant_order=order, dimensionality='integrated',
        harmonic=harmonic, value_type=f'v_{harmonic}{{2}}',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        kinematic_range=dict(eta=dict(min=eta[0], max=eta[1]),
                             pT=dict(min=pt[0], max=pt[1])),
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=a, max=b) for a, b in cent_eta]),
    )


def record_flow_vn4(**kw):
    r = record_flow_vn2(**kw)
    r['cumulant_order'] = 4
    return r


def record_flow_ep(harmonic=2, eta=(-1, 1), pt=(0.2, 2.0),
                   cent_classes=('0-10%', '10-40%', '40-80%'),
                   cent_eta=((-3.9, -3.0), (3.0, 3.9)),
                   dimensionality='integrated', bins=None):
    """An event_plane flow record -- single-acceptance, no declared reference.

    Added Sep 2026 because toy_mc had NO helper producing method=event_plane,
    so the `vn2_ep` / `vn3_ep` / `vn2_ep_diff_*` family (872 curated
    records) had no test coverage at all: every existing helper built a
    cumulant or yield record.

    Note what this exercises: with no `flow_reference` the pipeline uses
    the single-particle AMPLITUDE estimator, not the resolution-corrected
    event-plane estimator.  That is deliberate -- it is the documented
    fallback for records that declare no reference.
    """
    rec = dict(
        observable_type='flow_vn', method='event_plane',
        dimensionality=dimensionality, harmonic=harmonic,
        value_type=f'v_{harmonic}',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        kinematic_range=dict(eta=dict(min=eta[0], max=eta[1]),
                             pT=dict(min=pt[0], max=pt[1])),
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=a, max=b) for a, b in cent_eta]),
    )
    # `bins` must be a FLAT list of bin EDGES (the curated-record
    # convention); a list of (lo, hi) pairs silently yields one
    # "bin" that is a tuple of tuples and breaks the prepare step.
    if dimensionality == 'differential_pT':
        rec['kinematic_range']['pT']['bins'] = _as_edges(bins)
    elif dimensionality == 'differential_eta':
        rec['kinematic_range']['eta']['bins'] = _as_edges(bins)
    return rec


def _as_edges(bins):
    """Accept a flat edge list or (lo, hi) pairs; return a flat edge list.

    Adjacent bins share a boundary, so it must be emitted ONCE -- naively
    concatenating every pair's endpoints yields duplicate edges and hence
    zero-width bins.
    """
    b = list(bins)
    if b and isinstance(b[0], (tuple, list)):
        return [b[0][0]] + [hi for _, hi in b]
    return b


def record_flow_ep_diff_pt(bins, **kw):
    return record_flow_ep(dimensionality='differential_pT', bins=bins, **kw)


def record_flow_ep_diff_eta(bins, **kw):
    return record_flow_ep(dimensionality='differential_eta', bins=bins, **kw)


def record_flow_gap(sub_a=(-4, -1), sub_b=(1, 4), pt=(0.2, 2.0), **kw):
    r = record_flow_vn2(pt=pt, **kw)
    r['particles'] = 'AB'
    r['eta_gap'] = sub_b[0] - sub_a[1]
    r['subevents'] = {
        'A': dict(kinematic_range=dict(eta=dict(min=sub_a[0], max=sub_a[1]),
                                       pT=dict(min=pt[0], max=pt[1]))),
        'B': dict(kinematic_range=dict(eta=dict(min=sub_b[0], max=sub_b[1]),
                                       pT=dict(min=pt[0], max=pt[1]))),
    }
    return r


def record_flow_diff_pt(bins, eta=(-1, 1), harmonic=2, order=2, **kw):
    edges = [bins[0][0]] + [b[1] for b in bins]
    r = record_flow_vn2(eta=eta, pt=(edges[0], edges[-1]), harmonic=harmonic,
                        order=order, **kw)
    r['dimensionality'] = 'differential_pT'
    r['kinematic_range']['pT']['bins'] = edges
    return r


def record_flow_diff_eta(bins, pt=(0.2, 2.0), harmonic=2, order=2, **kw):
    edges = [bins[0][0]] + [b[1] for b in bins]
    r = record_flow_vn2(eta=(edges[0], edges[-1]), pt=pt, harmonic=harmonic,
                        order=order, **kw)
    r['dimensionality'] = 'differential_eta'
    r['kinematic_range']['eta']['bins'] = edges
    return r


def record_yield(species='charged', eta=(-0.5, 0.5), pt=(0.15, 3.0),
                 cent_classes=('0-10%', '10-40%', '40-80%')):
    return dict(
        observable_type='particle_yields', value_type='dN_dy',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        particle_id=dict(species=species),
        kinematic_range=dict(eta=dict(min=eta[0], max=eta[1]),
                             pT=dict(min=pt[0], max=pt[1])),
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
    )


def record_spectra(species='charged', eta=(-0.5, 0.5),
                   bins=None, cent_classes=('0-10%', '10-40%', '40-80%')):
    bins = bins or [(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
    edges = [bins[0][0]] + [b[1] for b in bins]
    return dict(
        observable_type='particle_spectra', value_type='invariant_yield',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        particle_id=dict(species=species),
        kinematic_range=dict(eta=dict(min=eta[0], max=eta[1]),
                             pT=dict(min=edges[0], max=edges[-1],
                                     bins=edges)),
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
    )


def _kr_eta_y(eta):
    kr = {}
    if eta is None:
        kr['y'] = {'min': -1.0, 'max': 1.0}
    else:
        kr['eta'] = {'min': eta[0], 'max': eta[1]}
    return kr


def record_pT_fluct(species='charged', eta=(-0.5, 0.5), pt=(0.15, 3.0),
                    cent_classes=('0-10%', '10-40%', '40-80%')):
    return dict(
        observable_type='fluctuations', value_type='pT_fluct',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        particle_id=dict(species=species),
        kinematic_range={**_kr_eta_y(eta),
                         'pT': dict(min=pt[0], max=pt[1])},
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
    )


def record_dET(species='charged', eta=(-0.5, 0.5), pt=(0.15, 3.0),
               cent_classes=('0-10%', '10-40%', '40-80%')):
    return dict(
        observable_type='transverse_energy', value_type='dET_deta',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        particle_id=dict(species=species),
        kinematic_range={**_kr_eta_y(eta),
                         'pT': dict(min=pt[0], max=pt[1])},
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
    )


def record_mean_pt(species='charged', eta=(-0.5, 0.5), pt=(0.15, 3.0),
                   cent_classes=('0-10%', '10-40%', '40-80%')):
    return dict(
        observable_type='mean_pT', value_type='mean_pT',
        hepdata_file='toy.yaml', inspire_id='0', version=1,
        particle_id=dict(species=species),
        kinematic_range=dict(eta=dict(min=eta[0], max=eta[1]),
                             pT=dict(min=pt[0], max=pt[1])),
        centrality=dict(classes=list(cent_classes),
                        eta=[dict(min=-3.9, max=-3.0), dict(min=3.0, max=3.9)]),
    )
