import numpy as np
from .qvector import QVectorData
from .measurement_db import MeasurementDB
from .weights import direct_cent_weights_percentile
from .observables_2d import (
    cut_qvectors,
    compute_vn2_integrated_per_cent_with_jackknife,
    compute_vn2_differential_pt_per_cent,
)
from .yields_spectra import multiplicity as compute_multiplicity
from .comparison import format_table, make_comparison_dict


class Computer2D:
    def __init__(self, manifest_path="analysis_manifest.json"):
        self.db = MeasurementDB(manifest_path)
        self.qv = None
        self.Qn_corrected = None
        self.jacobian = None
        self.event_weights = None
        self.cent_edges = None

    def load_qvectors(self, qn_path, nsamples_path, pt_bin_edges, delta_f_index=0):
        self.qv = QVectorData.from_2d_npy(qn_path, nsamples_path, pt_bin_edges,
                                           delta_f_index=delta_f_index)
        return self.qv

    def apply_eta_cut(self, eta_min=-1.0, eta_max=1.0):
        self.Qn_corrected, self.jacobian = cut_qvectors(self.qv, eta_min, eta_max)
        return self.Qn_corrected

    def compute_centrality(self, cent_pct_edges=None):
        if cent_pct_edges is None:
            cent_pct_edges = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80])
        Q0 = self.Qn_corrected[:, :, 0, :].sum(axis=(1, 2)).real
        self.event_weights = direct_cent_weights_percentile(Q0, cent_pct_edges)
        self.cent_edges = cent_pct_edges
        cent_labels = [f"{cent_pct_edges[i]:.0f}-{cent_pct_edges[i+1]:.0f}%"
                       for i in range(len(cent_pct_edges) - 1)]
        return cent_labels

    def compute_vn2_integrated(self, harmonics=None, n_jk_blocks=20):
        if self.Qn_corrected is None:
            raise RuntimeError("Run apply_eta_cut first")
        if self.event_weights is None:
            raise RuntimeError("Run compute_centrality first")
        vn2, err, bias = compute_vn2_integrated_per_cent_with_jackknife(
            self.Qn_corrected, self.qv, self.jacobian, self.event_weights, n_jk_blocks=n_jk_blocks)
        if harmonics is not None:
            vn2 = vn2[:, harmonics]
            err = err[:, harmonics]
        return vn2, err

    def compute_vn2_differential_pt(self, pt_bin_indices=None, n_jk_blocks=20):
        if self.Qn_corrected is None:
            raise RuntimeError("Run apply_eta_cut first")
        if self.event_weights is None:
            raise RuntimeError("Run compute_centrality first")
        if pt_bin_indices is not None:
            Qn = self.Qn_corrected[:, :, :, pt_bin_indices]
        else:
            Qn = self.Qn_corrected
        vn2, err, _ = compute_vn2_differential_pt_per_cent(
            Qn, self.qv, self.jacobian, self.event_weights, n_jk_blocks=n_jk_blocks)
        return vn2, err

    def compute_multiplicity(self):
        if self.Qn_corrected is None:
            raise RuntimeError("Run apply_eta_cut first")
        if self.event_weights is None:
            raise RuntimeError("Run compute_centrality first")
        return compute_multiplicity(self.qv, self.event_weights)

    def query_experiment(self, observable_type="flow_vn", experiment="STAR",
                         energy_gev=200.0, harmonic=2, method="cumulant",
                         cumulant_order=2, dimensionality="integrated"):
        records = self.db.query(
            observable_type=observable_type,
            experiment=experiment,
            energy_gev=energy_gev,
            harmonic=harmonic,
            method=method,
            cumulant_order=cumulant_order,
            dimensionality=dimensionality,
        )
        if not records and (method == "cumulant" and cumulant_order == 2):
            # Also try without cumulant filter for EP method
            records = self.db.query(
                observable_type=observable_type,
                experiment=experiment,
                energy_gev=energy_gev,
                harmonic=harmonic,
                dimensionality=dimensionality,
            )
        return records

    def compare_vn2_integrated(self, experiment_records,
                               sim_vn2, sim_err, cent_labels,
                               harmonic=2):
        if not experiment_records:
            return None
        for rec in experiment_records:
            dim = rec.get("dimensionality")
            if dim != "integrated":
                continue
            raw = self.db.get_raw_values(rec)
            if raw is None:
                continue
            exp_vals = raw["values"]
            exp_errs = [e.get("stat", 0.0) for e in raw["errors"]]
            exp_cent = raw["centralities"]
            if exp_cent is None:
                continue
            n_cent = min(len(cent_labels), len(sim_vn2), len(exp_vals))
            table = format_table(
                exp_cent[:n_cent],
                sim_vn2[:n_cent, harmonic],
                sim_err[:n_cent, harmonic],
                exp_vals[:n_cent],
                exp_errs[:n_cent],
                label=f"v_{harmonic}{{2}}",
            )
            return table
        return None
