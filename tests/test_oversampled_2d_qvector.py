"""Oversampled 2D containers: harmonic indexing and self-correlations.

Containers built from per-particle averages store Q_n = N * <e^{in phi}>
per pT bin, summed over oversampled particlizations.  The pipeline's
vn2_single core returns harmonic-major v1..v_{N-1}, so a v2 record must
read index 1, not 0; and the self-correlation term must scale with the
oversampling count.  Toy Q-vectors with a known v2/v3 check both.

Run: python3 tests/test_oversampled_2d_qvector.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analysis.qvector import QVectorData
from analysis.pipeline_dispatch import prepare_vn2_single_2d
from analysis.observables_2d import compute_vn2_single_per_cent_bin

NEV = 2000
NSP = 13
NH = 6
NPT = 41
V2_TRUE = 0.05
V3_TRUE = 0.02

PDG = [211, -211, 321, -321, 2212, -2212, 3122, -3122,
       3312, -3312, 3334, -3334, 333]


def _make_qv(coherent=True, seed=42):
    rng = np.random.default_rng(seed)
    pt_edges = np.linspace(0.05, 4.05, NPT + 1)
    qn = np.zeros((NEV, NSP, NH, NPT), dtype=complex)
    for e in range(NEV):
        psi2_ev = rng.uniform(0, 2 * np.pi)
        psi3_ev = rng.uniform(0, 2 * np.pi)
        for p in range(NPT):
            n = 1000.0
            psi2 = psi2_ev if coherent else rng.uniform(0, 2 * np.pi)
            psi3 = psi3_ev if coherent else rng.uniform(0, 2 * np.pi)
            qn[e, 0, 0, p] = n
            qn[e, 0, 2, p] = n * V2_TRUE * np.exp(1j * 2 * psi2)
            qn[e, 0, 3, p] = n * V3_TRUE * np.exp(1j * 3 * psi3)
    return QVectorData(
        Qn=qn, pt_bin_edges=pt_edges,
        masses=np.ones(NSP),
        species_names=[f's{i}' for i in range(NSP)],
        nsamples=np.ones(NEV), rap_max=5.0, pdg_ids=PDG)


def _make_single_bin_qv(nsamples, m_per_sample, v2=0.05, seed=7):
    """One species, one pT bin, ``m_per_sample`` particles per sample.

    ``nsamples`` is the oversampling factor; the stored count is the
    total over all samples, exactly as the campaign HDF5 does.
    """
    rng = np.random.default_rng(seed)
    psi2 = rng.uniform(0, 2 * np.pi)
    n_total = float(nsamples) * float(m_per_sample)
    # Two pT bins so the [0.2, 4.05] slice is non-empty: put all
    # particles in the second bin, whose lower edge is exactly 0.2.
    qn = np.zeros((1, 1, NH, 2), dtype=complex)
    qn[0, 0, 0, 1] = n_total
    qn[0, 0, 2, 1] = n_total * v2 * np.exp(2j * psi2)
    return QVectorData(
        Qn=qn, pt_bin_edges=np.array([0.0, 0.2, 4.05]),
        masses=np.array([0.13957]), species_names=['pi+'],
        nsamples=np.array([float(nsamples)]), rap_max=5.0, pdg_ids=[211])


def test_self_correlation_scaling():
    """dQ0/Q0^2 must scale as 1/(M*nsamples), not 1/M.

    The campaign stores the count summed over all oversampled shots, so
    the per-particle weight carries one factor of 1/nsamples and dQ0
    carries two.  At the campaign's nsamples ~ 10^2-10^3 this is ~10^-6
    of Q0^2 and must not move v2.
    """
    params = {'single_range': {'eta': (-1.0, 1.0), 'pT': (0.2, 100.0)}}
    m_per_sample = 500.0
    print("nsamples   dQ0/Q0^2     (dQ0/Q0^2)*M*ns   v2_sub    v2_nosub")
    for ns in (1, 10, 100, 1000):
        qv = _make_single_bin_qv(ns, m_per_sample)
        d = prepare_vn2_single_2d(qv, params)
        q0 = d['Qn'][0, 0].real
        ratio = d['dQ0'][0] / q0 ** 2
        scaled = ratio * m_per_sample * ns
        sub = compute_vn2_single_per_cent_bin(
            d['Qn'], d['dQ0'], np.ones(1))[1]
        nosub = compute_vn2_single_per_cent_bin(
            d['Qn'], np.zeros_like(d['dQ0']), np.ones(1))[1]
        print(f"{ns:8d}   {ratio:.3e}   {scaled:.6f}          "
              f"{sub:.6f}  {nosub:.6f}")
        assert abs(scaled - 1.0) < 1e-6, "self-correlation must scale as 1/(M*ns)"
        if ns >= 1000:
            assert abs(sub - nosub) / nosub < 1e-3, \
                "self-correlation subtraction must be negligible at high ns"


def main():
    params = {'single_range': {'eta': (-1.0, 1.0), 'pT': (0.2, 2.0)}}

    coherent = _make_qv(coherent=True)
    d = prepare_vn2_single_2d(coherent, params)
    vals = compute_vn2_single_per_cent_bin(d['Qn'], d['dQ0'],
                                           np.ones(NEV))
    v1, v2, v3 = vals[0], vals[1], vals[2]
    print(f"coherent pT phases: v1={v1:.6f} v2={v2:.6f} "
          f"v3={v3:.6f} (inputs {V2_TRUE}, {V3_TRUE})")
    assert abs(v2 - V2_TRUE) < 5e-3, "v2 not recovered from coherent Q-vectors"
    assert abs(v3 - V3_TRUE) < 5e-3, "v3 not recovered from coherent Q-vectors"
    assert abs(v1) < 1e-2, "v1 should be zero in this toy"

    # Sanity: if the phase is randomized per pT bin, the vector sum cancels.
    random_phase = _make_qv(coherent=False)
    d2 = prepare_vn2_single_2d(random_phase, params)
    vals2 = compute_vn2_single_per_cent_bin(d2['Qn'], d2['dQ0'],
                                            np.ones(NEV))
    print(f"random pT phases:   v1={vals2[0]:.6f} v2={vals2[1]:.6f} "
          f"v3={vals2[2]:.6f}")
    # The random-phase vector sum is a finite-sample residual, not zero;
    # it must be much smaller than the coherent input.
    assert abs(vals2[1]) < 0.25 * V2_TRUE, \
        "random pT phases should strongly suppress the v2 vector sum"
    assert abs(vals2[2]) < 0.5 * V3_TRUE, \
        "random pT phases should strongly suppress the v3 vector sum"

    test_self_correlation_scaling()

    print("OK")


if __name__ == '__main__':
    main()
