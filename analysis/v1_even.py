"""Rapidity-even directed flow v1^even(pT) from the factorization of v11.

The experiments (STAR PLB 784 (2018) 26, ATLAS PRC 86 (2012) 014907)
measure the pair coefficient v11(pTa, pTb) = <cos(phi_a - phi_b)> with a
pseudorapidity gap and fit
    v11(pTa, pTb) = v1(pTa) v1(pTb) - K pTa pTb,
the second term the correlation from global transverse-momentum
conservation.  A hydrodynamic model without that correlation is compared
with the fitted v1 directly: its v11 matrix on the fit's pT grid is
factorized as v1(pTa) v1(pTb), the K = 0 fit.

Given the class-averaged pair matrix V (pairs of distinct particles, the
gap applied where the container resolves eta) over a set of pT bins, and
the numbers of pairs W behind each element:
  - the grid block G = V[grid, grid] is fitted by v1_grid minimizing
        sum_kl W_kl (G_kl - v_k v_l)^2,
    the diagonal included: a chi^2 fit with errors ~ 1/sqrt(pairs), as the
    experiments' fit (the unweighted fit, the leading eigenvector, is
    dominated by the sparse high-pT elements: in a toy with ~7 particles
    per event above 2.2 GeV its spread was up to twice the weighted one);
    the stationary condition v_k = sum_l W_kl G_kl v_l / sum_l W_kl v_l^2
    is iterated from the leading eigenvector, damped (the undamped map
    oscillates in the overall scale);
  - any bin t (a record bin, a window, the sign window) is the weighted
    least-squares projection on that factor,
        v1(t) = sum_g W_tg V_tg v_g / sum_g W_tg v_g^2,
    which equals v1_grid for t on the grid;
  - the overall sign is open (Psi_1 -> Psi_1 + pi): the record's
    sign_convention fixes the sign of v1 in a pT window (positive at high
    pT, where the experiments define it), or the record tabulates |v1|.
A non-positive leading eigenvalue (no dipole correlation resolved) or a
zero sign-window value, or a fit that does not converge, gives NaN.
"""
import numpy as np


def class_pairs(D, w):
    """Pair counts behind class_matrix: sum_e w_e D_e[k, l]."""
    return np.tensordot(np.asarray(w, dtype=float), D, axes=(0, 0))


def class_matrix(N, D, w):
    """Pair-averaged matrix V[k, l] = sum_e w_e N_e[k, l] / sum_e w_e D_e[k, l].

    N, D: (n_events, K, K) per-event sums of cos(phi_i - phi_j) and of the
    pair weights over distinct pairs (i in bin k, j in bin l); w: event
    weights of the class."""
    w = np.asarray(w, dtype=float)
    num = np.tensordot(w, N, axes=(0, 0))
    den = np.tensordot(w, D, axes=(0, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def rank1_fit(G, W, tol=1e-13, max_iter=5000):
    """Symmetric rank-1 fit v minimizing sum_kl W_kl (G_kl - v_k v_l)^2; NaN if
    the leading eigenvalue of G is not positive or the iteration fails."""
    G = 0.5 * (G + G.T)
    W = 0.5 * (W + W.T)
    nan = np.full(len(G), np.nan)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(W))):
        return nan
    lam, U = np.linalg.eigh(G)
    if not lam[-1] > 0:
        return nan
    v = np.sqrt(lam[-1]) * U[:, -1]
    for _ in range(max_iter):
        den = W @ (v ** 2)
        if not np.all(den > 0):
            return nan
        new = 0.5 * (v + (W * G) @ v / den)
        if np.max(np.abs(new - v)) <= tol * np.max(np.abs(new)):
            return new
        v = new
    return nan


def v1_from_matrix(V, grid, targets, sign_bin=None, sign=1.0, absolute=False, weights=None):
    """v1^even of the target bins from the pair matrix V (see module doc).

    grid, targets: index lists into V; sign_bin: index of the sign window
    (None with absolute=True); sign: +1 or -1, the sign v1 takes there;
    weights: the pair counts behind V (default uniform)."""
    V = np.asarray(V, dtype=float)
    W = np.ones_like(V) if weights is None else np.asarray(weights, dtype=float)
    out = np.full(len(targets), np.nan)
    v1g = rank1_fit(V[np.ix_(grid, grid)], W[np.ix_(grid, grid)])
    if not np.all(np.isfinite(v1g)):
        return out

    def proj(t):
        w = W[t, grid]
        den = float(w @ (v1g ** 2))
        return float((w * V[t, grid]) @ v1g) / den if den > 0 else np.nan

    vals = np.array([proj(t) for t in targets])
    if absolute:
        return np.abs(vals)
    s = proj(sign_bin)
    if not np.isfinite(s) or s == 0:
        return out
    return vals * (np.sign(s) * np.sign(sign))
