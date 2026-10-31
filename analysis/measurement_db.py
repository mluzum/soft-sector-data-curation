import json
import numpy as np
from collections import defaultdict


class MeasurementDB:
    def __init__(self, manifest_path="analysis_manifest.json"):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.records = manifest.get("records", [])
        self.manifest_path = manifest_path

    def query(self, observable_type=None, experiment=None, harmonic=None,
              method=None, cumulant_order=None, particle_species=None,
              dimensionality=None, centrality_class=None,
              energy_gev=None, reaction=None, value_type=None):
        results = []
        for r in self.records:
            if observable_type is not None and r.get("observable_type") != observable_type:
                continue
            if experiment is not None and str(r.get("experiment", "")).upper() != str(experiment).upper():
                continue
            if harmonic is not None and r.get("harmonic") != harmonic:
                continue
            if method is not None and str(r.get("method", "")).lower() != str(method).lower():
                continue
            if cumulant_order is not None and r.get("cumulant_order") != cumulant_order:
                continue
            if particle_species is not None:
                pid = r.get("particle_id", {})
                if not isinstance(pid, dict):
                    pid = {}
                ps = str(pid.get("species", ""))
                if ps.upper() != particle_species.upper():
                    continue
            if dimensionality is not None and r.get("dimensionality") != dimensionality:
                continue
            if centrality_class is not None:
                cent = r.get("centrality", {})
                classes = cent.get("classes", []) if isinstance(cent, dict) else []
                if not any(centrality_class in c for c in classes):
                    continue
            if energy_gev is not None:
                cs = r.get("collision_system", {})
                if not isinstance(cs, dict):
                    cs = {}
                if cs.get("energy_gev") != energy_gev:
                    continue
            if reaction is not None:
                cs = r.get("collision_system", {})
                if not isinstance(cs, dict):
                    cs = {}
                if reaction.upper() not in str(cs.get("reaction", "")).upper():
                    continue
            if value_type is not None and r.get("value_type") != value_type:
                continue
            results.append(r)
        return results

    def one(self, **kwargs):
        results = self.query(**kwargs)
        if len(results) == 1:
            return results[0]
        if len(results) == 0:
            raise KeyError(f"No record matches {kwargs}")
        raise KeyError(f"{len(results)} records match {kwargs}")

    def group_by_centrality(self, records):
        groups = defaultdict(list)
        for r in records:
            cent = r.get("centrality", {})
            if not isinstance(cent, dict):
                cent = {}
            key = (
                cent.get("estimator"),
                cent.get("centrality_eta"),
                cent.get("centrality_pt"),
            )
            groups[key].append(r)
        return dict(groups)

    def group_by_kinematic_cuts(self, records):
        groups = defaultdict(list)
        for r in records:
            kr = r.get("kinematic_range", {})
            if not isinstance(kr, dict):
                kr = {}
            pt = kr.get("pT", {})
            eta = kr.get("eta", {})
            key = (
                pt.get("min") if isinstance(pt, dict) else None,
                pt.get("max") if isinstance(pt, dict) else None,
                eta.get("min") if isinstance(eta, dict) else None,
                eta.get("max") if isinstance(eta, dict) else None,
            )
            groups[key].append(r)
        return dict(groups)

    def get_raw_values(self, record):
        rd = record.get("raw_data", {})
        dvs = rd.get("dependent_variables", [])
        if not dvs:
            return None

        values = []
        errors = []
        for v in dvs[0].get("values", []):
            values.append(v.get("value"))
            err = v.get("errors", [{}])
            err_dict = {}
            for e in err:
                if e.get("symerror") is not None:
                    label = e.get("label") or "stat"
                    err_dict[label] = e["symerror"]
            errors.append(err_dict)

        centralities = None
        cent_info = record.get("centrality", {})
        if isinstance(cent_info, dict):
            centralities = cent_info.get("classes")

        pt_bins = None
        eta_bins = None
        kr = record.get("kinematic_range", {})
        if isinstance(kr, dict):
            pt_info = kr.get("pT", {})
            if isinstance(pt_info, dict):
                pt_bins = pt_info.get("bins")
            eta_info = kr.get("eta", {})
            if isinstance(eta_info, dict):
                eta_bins = eta_info.get("bins")

        return {
            "values": np.array(values, dtype=np.float64),
            "errors": errors,
            "centralities": centralities,
            "pt_bins": np.array(pt_bins) if pt_bins else None,
            "eta_bins": np.array(eta_bins) if eta_bins else None,
            "inspire_id": record.get("inspire_id"),
            "value_type": record.get("value_type"),
            "dimensionality": record.get("dimensionality"),
        }

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        return f"MeasurementDB({len(self.records)} records, {self.manifest_path})"
