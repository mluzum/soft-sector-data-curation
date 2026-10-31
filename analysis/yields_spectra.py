import numpy as np


def multiplicity(qv, event_weights):
    n_cent = event_weights.shape[1]
    Q0 = qv.Qn[:, :, 0, :]
    M_per_pt = (Q0 / qv.nsamples[:, None, None]).real
    cent_M = np.array([
        np.average(M_per_pt, axis=0, weights=event_weights[:, c])
        for c in range(n_cent)
    ])
    M_total = cent_M.sum(axis=-1)
    return {
        "dNch_deta": M_total,
        "dNch_deta_per_species": cent_M,
        "species_names": qv.species_names,
        "pt_centers": qv.pt_centers,
    }


def mean_pt(qv, event_weights):
    n_cent = event_weights.shape[1]
    pt = qv.pt_centers
    Q0 = qv.Qn[:, :, 0, :].real
    Q1_re = qv.Qn[:, :, 1, :].real
    M = Q0 / qv.nsamples[:, None, None]
    pT_sum = Q1_re / qv.nsamples[:, None, None]
    M[M < 1e-30] = 0.0
    pT_per_event = np.divide(pT_sum, M, out=np.zeros_like(pT_sum), where=M > 1e-30)
    cent_pt = np.array([
        np.average(pT_per_event, axis=0, weights=event_weights[:, c] * M.sum(axis=(1, 2)))
        for c in range(n_cent)
    ])
    return {
        "mean_pT": cent_pt.sum(axis=-1),
        "mean_pT_per_species": cent_pt,
        "species_names": qv.species_names,
    }


def pt_spectra(qv, event_weights, delta_eta=2.0):
    n_cent = event_weights.shape[1]
    pt = qv.pt_centers
    dpt = np.diff(qv.pt_bin_edges)
    Q0 = qv.Qn[:, :, 0, :].real / qv.nsamples[:, None, None]
    cent_spectra = np.array([
        np.average(Q0, axis=0, weights=event_weights[:, c])
        for c in range(n_cent)
    ])
    cent_spectra /= delta_eta
    inv_2pi_pt_dpt = 1.0 / (2 * np.pi * pt * dpt)
    cent_spectra *= inv_2pi_pt_dpt[np.newaxis, :]
    return {
        "spectra": cent_spectra,
        "spectra_per_species": cent_spectra,
        "pt_centers": pt,
        "pt_bin_edges": qv.pt_bin_edges,
        "species_names": qv.species_names,
    }
