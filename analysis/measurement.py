import re
import yaml
import numpy as np


class MeasurementRecord:
    """Wraps a curated YAML record for programmatic access to kinematic cuts."""

    def __init__(self, record):
        self.r = record

    # --- Kinematic cuts ---

    @property
    def pt_min(self):
        kr = self.r.get('kinematic_range', {}) or {}
        pt = kr.get('pT', {}) or {}
        return pt.get('min')

    @property
    def pt_max(self):
        kr = self.r.get('kinematic_range', {}) or {}
        pt = kr.get('pT', {}) or {}
        return pt.get('max')

    @property
    def eta_min(self):
        kr = self.r.get('kinematic_range', {}) or {}
        eta = kr.get('eta', {}) or {}
        return eta.get('min')

    @property
    def eta_max(self):
        kr = self.r.get('kinematic_range', {}) or {}
        eta = kr.get('eta', {}) or {}
        return eta.get('max')

    # --- Observable specs ---

    @property
    def harmonic(self):
        return self.r.get('harmonic')

    @property
    def method(self):
        return self.r.get('method')

    @property
    def cumulant_order(self):
        return self.r.get('cumulant_order') or self.r.get('cumulant')

    @property
    def dimensionality(self):
        return self.r.get('dimensionality')

    @property
    def value_type(self):
        return self.r.get('value_type', '')

    # --- Collision system ---

    @property
    def energy_gev(self):
        cs = self.r.get('collision_system', {}) or {}
        return cs.get('energy_gev')

    @property
    def reaction(self):
        cs = self.r.get('collision_system', {}) or {}
        return cs.get('reaction', '')

    @property
    def experiment(self):
        return self.r.get('experiment') or self.r.get('source_file', '').split('_')[0]

    # --- Particle ---

    @property
    def species(self):
        pid = self.r.get('particle_id', {}) or {}
        return pid.get('species', 'charged')

    # --- Centrality ---

    @property
    def centrality_classes(self):
        cent = self.r.get('centrality', {}) or {}
        return cent.get('classes', [])

    @property
    def centrality_estimator(self):
        cent = self.r.get('centrality', {}) or {}
        est = cent.get('estimator', {}) or {}
        return est.get('detector'), est.get('name')

    # --- η-gap heuristic ---

    def extract_eta_gap(self):
        """Guess η-gap from value_type or table_label.

        Returns (eta_min_A, eta_max_A, eta_min_B, eta_max_B) or None.
        """
        vt = self.value_type
        label = str(self.r.get('hepdata_table_label', ''))

        # Check for "ab" suffix (PHENIX two-subevent notation)
        if 'ab' in vt.lower():
            eta_accept = self.r.get('kinematic_range', {}).get('eta', {})
            if eta_accept:
                eta_lo = eta_accept.get('min', -1.0)
                eta_hi = eta_accept.get('max', 1.0)
                gap = 2.0
                return (eta_lo, -gap / 2, gap / 2, eta_hi)

        # Check for "|Δη| >" in label
        m = re.search(r'\|\s*[ΔD]eta\s*\|\s*>\s*([\d.]+)', label)
        if m:
            gap = float(m.group(1))
            return (None, None, None, gap)  # gap value known, ranges TBD

        # Check for "η-gap" / "pseudorapidity gap"
        if 'gap' in label.lower():
            m = re.search(r'\|[ΔD]eta\|\s*>\s*([\d.]+)', label)
            if m:
                gap = float(m.group(1))
                return (None, None, None, gap)

        return None

    # --- Raw data ---

    def get_centrality_values(self):
        """Return (centralities, values, stat_errors) from the record."""
        raw = self.r.get('raw_data', {})
        dvs = raw.get('dependent_variables', [])
        if not dvs:
            return None, None, None
        cent = self.centrality_classes
        vals = []
        errs = []
        for v in dvs[0].get('values', []):
            val = v.get('value')
            if val is None:
                continue
            vals.append(val)
            err_list = v.get('errors', [])
            stat = None
            for e in err_list:
                if e.get('symerror') is not None:
                    stat = e['symerror']
                    break
            errs.append(stat or 0.0)
        return cent, np.array(vals, dtype=np.float64), np.array(errs, dtype=np.float64)

    # --- Display ---

    def __repr__(self):
        vt = self.value_type[:40]
        harm = self.harmonic
        method = self.method
        co = self.cumulant_order
        en = self.energy_gev
        spec = self.species
        return (f"Record(v_{harm}{{{co}}}={vt!r}, "
                f"{en}GeV, {spec}, cent={len(self.centrality_classes)} bins)")

    @staticmethod
    def from_yaml_file(filepath, doc_index=0):
        with open(filepath) as f:
            docs = list(yaml.safe_load_all(f))
        return MeasurementRecord(docs[doc_index])
