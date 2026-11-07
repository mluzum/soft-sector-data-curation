"""Dispatch curated records to computation functions.

DISPATCH_TABLE maps (method, cumulant_order, particles_pattern, dimensionality)
to a compute function reference plus the set of metadata fields required.
Adding a new observable type = adding an entry + the corresponding function.

DISPATCH_TABLE_BY_TYPE maps observable_type → compute key for non-flow
observables (yields, spectra, etc.) that don't have method/cumulant fields.
"""

import re
import os
import numpy as np
import yaml

# libyaml's parser when installed (~8x faster on the record files; the same
# Python constructors, so the same documents -- checked on every record
# file and every HEPData table they read)
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def yaml_load_all(f):
    """yaml.safe_load_all with the C loader when available."""
    return yaml.load_all(f, Loader=_YAML_LOADER)


_TABLE_CACHE = {}


def _load_table(path):
    """A parsed HEPData table, cached by path (and mtime, size): the records
    of one table (ATLAS 3170344: 27-29 classes per file) share one parse.
    Read-only for callers."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _TABLE_CACHE:
        if len(_TABLE_CACHE) >= 512:
            _TABLE_CACHE.clear()
        with open(path) as f:
            _TABLE_CACHE[key] = yaml.load(f, Loader=_YAML_LOADER)
    return _TABLE_CACHE[key]

from analysis.species import is_charged, species_known
from analysis.cent_combine import combination_plan


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RECORD_STATUSES = ("published", "preliminary")


def record_roots(extra=None):
    """Record directories laid out like data/<observable_type>/: this repository's
    data/ (relative, as the scripts run from the repository root), then each
    directory of `extra` and of $SOFT_SECTOR_EXTRA_RECORDS (os.pathsep-separated),
    e.g. a private collection of preliminary or not yet vetted records."""
    roots = ["data"]
    env = os.environ.get("SOFT_SECTOR_EXTRA_RECORDS", "")
    for d in list(extra or []) + [p for p in env.split(os.pathsep) if p]:
        d = os.path.abspath(d)
        if d not in roots:
            roots.append(d)
    return roots


# ── Dispatch table ──────────────────────────────────────────────────────────
# Keys: (method, cumulant_order, particles_pattern, dimensionality)
#   particles_pattern: None for single-acceptance, "AB"/"AABB"/etc for subevent
# Values: {
#   "compute": str — key in COMPUTE_REGISTRY
#   "requires": list of str — dot-separated field paths that must exist
#   "description": str — human-readable
# }

DISPATCH_TABLE = {
    ("cumulant", 8, None, "integrated"): {
        "compute": "vn8_single",
        "requires": ["kinematic_range.eta", "kinematic_range.pT"],
        "description": "v_n{8} integrated cumulant",
    },
    ("event_plane_mixed", None, None, "integrated"): {
        "compute": "vn_mixed_plane",
        "requires": ["flow_reference.eta", "flow_reference.harmonic", "flow_reference.resolution_subevents"],
        "description": "v_n relative to a lower-order plane (integrated)",
    },
    ("event_plane_mixed", None, None, "differential_pT"): {
        "compute": "vn_mixed_plane",
        "requires": ["flow_reference.eta", "flow_reference.harmonic", "flow_reference.resolution_subevents"],
        "description": "v_n relative to a lower-order plane (differential_pT)",
    },
    ("event_plane_mixed", None, None, "differential_y"): {
        "compute": "vn_mixed_plane",
        "requires": ["flow_reference.eta", "flow_reference.harmonic", "flow_reference.resolution_subevents"],
        "description": "v_n relative to a lower-order plane (differential_y)",
    },
    ("event_plane_mixed", None, None, "differential_eta"): {
        "compute": "vn_mixed_plane",
        "requires": ["flow_reference.eta", "flow_reference.harmonic", "flow_reference.resolution_subevents"],
        "description": "v_n relative to a lower-order plane (differential_eta)",
    },
    ("three_subevent_2pc", None, None, "integrated"): {
        "compute": "vn_3x2pc",
        "requires": ["flow_reference.subevents.B.eta", "flow_reference.subevents.C.eta"],
        "description": "v_n from three two-particle correlations, 3x2PC (integrated)",
    },
    ("three_subevent_2pc", None, None, "differential_pT"): {
        "compute": "vn_3x2pc",
        "requires": ["flow_reference.subevents.B.eta", "flow_reference.subevents.C.eta"],
        "description": "v_n from three two-particle correlations, 3x2PC (differential_pT)",
    },
    ("three_subevent_2pc", None, None, "differential_eta"): {
        "compute": "vn_3x2pc",
        "requires": ["flow_reference.subevents.B.eta", "flow_reference.subevents.C.eta"],
        "description": "v_n from three two-particle correlations, 3x2PC (differential_eta)",
    },
    ("three_subevent_2pc", None, None, "differential_y"): {
        "compute": "vn_3x2pc",
        "requires": ["flow_reference.subevents.B.eta", "flow_reference.subevents.C.eta"],
        "description": "v_n from three two-particle correlations, 3x2PC (differential_y)",
    },
    ("cumulant_partner", 2, None, "differential_eta"): {
        "compute": "vn2_eta_partner",
        "requires": ["flow_reference.segments", ["kinematic_range.eta.bins", "kinematic_range.eta.bin_ranges"]],
        "description": "v_n{2}(eta), side-dependent reference normalized by its partner region",
    },
    ("cumulant", 6, None, "integrated"): {
        "compute": "vn6_single",
        "requires": ["kinematic_range.eta", "kinematic_range.pT"],
        "description": "v_n{6} integrated cumulant",
    },
    ("cumulant", 6, None, "differential_pT"): {
        "compute": "vn6_single_diff_pT",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges"]],
        "description": "v_n{6} pT-differential (particle of interest)",
    },
    ("cumulant", 6, None, "differential_eta"): {
        "compute": "vn6_single_diff_eta",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.eta.bins", "kinematic_range.eta.points", "kinematic_range.eta.bin_ranges"]],
        "description": "v_n{6} eta-differential (particle of interest)",
    },
    ("cumulant", 8, None, "differential_pT"): {
        "compute": "vn8_single_diff_pT",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges"]],
        "description": "v_n{8} pT-differential (particle of interest)",
    },
    ("cumulant", 8, None, "differential_eta"): {
        "compute": "vn8_single_diff_eta",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.eta.bins", "kinematic_range.eta.points", "kinematic_range.eta.bin_ranges"]],
        "description": "v_n{8} eta-differential (particle of interest)",
    },
    ("scalar_product", None, None, "differential_pT"): {
        "compute": "vn2_sp_diff_pT",
        "requires": "kinematic_range.eta",
        "description": "scalar-product differential v_n(pT){SP}",
    },
    ("scalar_product", None, None, "differential_eta"): {
        "compute": "vn2_sp_diff_eta",
        "requires": "kinematic_range.pT",
        "description": "scalar-product differential v_n(eta){SP}",
    },
    ("scalar_product", None, None, "integrated"): {
        "compute": "vn2_sp",
        "requires": "kinematic_range.eta",
        "description": "scalar-product integrated v_n{SP}",
    },
    ("event_plane", None, None, "differential_eta"): {
        "compute": "vn2_ep_diff_eta",
        "requires": "kinematic_range.pT",
        "description": "event-plane differential v_n(eta){EP} (per-paper exception for PHOBOS v2(eta))",
    },
    ("event_plane", None, None, "differential_pT"): {
        "compute": "vn2_ep_diff_pT",
        "requires": "kinematic_range.eta",
        "description": "event-plane differential v_n(pT){EP}",
    },
    ("event_plane", 3, None, "integrated"): {
        "compute": "vn3_ep",
        "requires": "kinematic_range.eta",
        "description": "event-plane v3 integrated (per STAR 2013qio records)",
    },
    ("event_plane", None, None, "integrated"): {
        "compute": "vn2_ep",
        "requires": "kinematic_range.eta",
        "description": "event-plane v2 integrated (fallback for event_plane records without cumulant_order)",
    },
    ("cumulant", 2, None, "integrated"): {
        "compute": "vn2_single",
        "requires": ["kinematic_range.eta", "kinematic_range.pT"],
        "description": "v_n{2} single-acceptance, integrated over pT/eta",
    },
    ("cumulant", 2, "AB", "integrated"): {
        "compute": "vn2_gap",
        "requires": [
            "subevents.A.kinematic_range.eta",
            "subevents.A.kinematic_range.pT",
            "subevents.B.kinematic_range.eta",
            "subevents.B.kinematic_range.pT",
        ],
        "description": "v_n{2}ab two-subevent, integrated over pT/eta",
    },
    ("cumulant", 2, None, "differential_pT"): {
        "compute": "vn2_single_diff_pT",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges"]],
        "description": "v_n{2} pT-differential, single centrality bin",
    },
    ("cumulant", 2, None, "differential_eta"): {
        "compute": "vn2_single_diff_eta",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.eta.bins", "kinematic_range.eta.points", "kinematic_range.eta.bin_ranges"]],
        "description": "v_n{2} eta-differential, single centrality bin",
    },
    ("cumulant", 2, "AB", "differential_pT"): {
        # POI in each pT bin of one subevent vs the reference integrated in
        # the other (vn2_gap_diff_pT correlated each bin with itself).
        "compute": "vn2_gap_poi",
        "requires": [
            "subevents.A.kinematic_range.eta",
            "subevents.A.kinematic_range.pT",
            "subevents.B.kinematic_range.eta",
            "subevents.B.kinematic_range.pT",
            ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges"],
        ],
        "description": "v_n{2}ab pT-differential, two-subevent",
    },
    ("cumulant", 2, "AB", "differential_eta"): {
        "compute": "vn2_gap_diff_eta",
        "requires": [
            "subevents.A.kinematic_range.eta",
            "subevents.A.kinematic_range.pT",
            "subevents.B.kinematic_range.eta",
            "subevents.B.kinematic_range.pT",
            ["kinematic_range.eta.bins", "kinematic_range.eta.points", "kinematic_range.eta.bin_ranges"],
        ],
        "description": "v_n{2}ab eta-differential, two-subevent",
    },
    ("cumulant", 4, None, "integrated"): {
        "compute": "vn4_single",
        "requires": ["kinematic_range.eta", "kinematic_range.pT"],
        "description": "v_n{4} single-acceptance, integrated over pT/eta",
    },
    ("cumulant", 4, None, "differential_pT"): {
        "compute": "vn4_single_diff_pT",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges"]],
        "description": "v_n{4} pT-differential, single centrality bin",
    },
    ("cumulant", 4, None, "differential_eta"): {
        "compute": "vn4_single_diff_eta",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", ["kinematic_range.eta.bins", "kinematic_range.eta.points", "kinematic_range.eta.bin_ranges"]],
        "description": "v_n{4} eta-differential, single centrality bin",
    },
    ("cumulant", 4, "AABB", "integrated"): {
        "compute": "vn4_gap",
        "requires": [
            "subevents.A.kinematic_range.eta",
            "subevents.A.kinematic_range.pT",
            "subevents.B.kinematic_range.eta",
            "subevents.B.kinematic_range.pT",
        ],
        "description": "v_n{4} AABB two-subevent, integrated",
    },
}

# ── Non-flow dispatch (by observable_type) ──────────────────────────────────
DISPATCH_TABLE_BY_TYPE = {
    "particle_yields": {
        "compute": "yield_dNch_deta",
        "requires": [["kinematic_range.eta", "kinematic_range.y", "centrality.eta"]],
        "description": "dN/dy, dNch/dη, multiplicity",
    },
    "particle_spectra": {
        "compute": "spectra_invariant_yield",
        "requires": [["kinematic_range.eta", "kinematic_range.y", "centrality.eta"], ["kinematic_range.pT.bins", "kinematic_range.pT.points", "kinematic_range.pT.bin_ranges", "kinematic_range.pT.min", "kinematic_range.pT.max"]],
        "description": "Invariant yield dN/(2π pT dpT dη)",
    },
    "fluctuations": {
        "compute": "pT_fluct",
        "value_type_contains": "pT",
        "requires": [["kinematic_range.eta", "kinematic_range.y", "centrality.eta"]],
        "description": "pT fluctuation correlator Σ",
    },
    "flow_pT_correlations": {
        # rho(v_n^2, [pT]) (value_type rho_n), cov(v_n^2, [pT]) (cov_n) or
        # cov / <[pT]> (cov_n_over_pT);
        # windows and methods from the record (analysis/flow_pt.py)
        "compute": "rho_vn2_pT",
        "requires": ["kinematic_range.eta", "kinematic_range.pT", "harmonic"],
        "description": "flow-[pT] correlation rho(v_n^2, [pT]) and cov(v_n^2, [pT])",
    },
    "mean_pT_fluctuations": {
        # value_type 'sqrt_<dpt1_dpt2>_over_<pT>' (relative) or '<dpt1_dpt2>'
        # (the covariance itself, GeV^2)
        "compute": "pT_fluct",
        "value_type_contains": "<dpt1_dpt2>",
        "requires": [["kinematic_range.eta", "kinematic_range.y", "centrality.eta"]],
        "description": "dynamical pT correlation <Delta pT_i Delta pT_j>, or its root over <<pT>>",
    },
    "transverse_energy": {
        "compute": "dET_deta",
        "requires": [["kinematic_range.eta", "centrality.eta"]],
        "description": "dET/deta (grid containers only)",
    },
    "radial_flow": {
        "compute": "radial_flow",
        "requires": ["subevents", "reference_pT", "quantity", "kinematic_range.pT"],
        "description": "v0, v0(pT) and the normalized [pT]-spectrum covariance (two subevents)",
    },
    "mean_pT": {
        "compute": "mean_pt",
        "requires": [["kinematic_range.eta", "kinematic_range.y", "centrality.eta"]],
        "description": "Mean transverse momentum ⟨pT⟩",
    },
    "flow_harmonic_correlations": {
        "compute": {
            "symmetric_cumulant": "sc_mn",
            "event_plane_correlation": "nac_mn",
            "mode_coupling": "mode_coupling",
            "plane_correlation": "plane_corr_sp",
            "pT_flow_vector_fluctuations": "flow_vector_fluct",
            "longitudinal_decorrelation": "rnn_decorrelation",
        },
        "requires": [["kinematic_range.eta", "centrality.eta"],
                     "harmonics", "correlation_type"],
        "requires_per_ct": {
            "mode_coupling": [["kinematic_range.eta"], "harmonics", "correlation_type", "quantity"],
            "plane_correlation": [["kinematic_range.eta"], "harmonics", "correlation_type", "planes",
                                  "plane_subevents", "estimator"],
            "pT_flow_vector_fluctuations": [["kinematic_range.eta"], "harmonics", "correlation_type", "quantity",
                                            "flow_reference.pT"],
            "longitudinal_decorrelation": [["kinematic_range.eta"], "harmonics", "correlation_type", "quantity",
                                           "decorrelation_reference.eta"],
            "event_plane_correlation": [
                ["kinematic_range.eta", "centrality.eta"],
                "harmonics", "correlation_type",
                "particles", "subevents",
            ],
        },
        "description": "Mixed-harmonic flow correlations: SC(n,m), subevent NAC(n,m), the linear and "
                       "mode-coupled flow harmonics (analysis.mode_coupling), or symmetry-plane "
                       "correlations, SP method (analysis.plane_correlations)",
    },
}

COMPUTE_REGISTRY = {}


def register(key):
    def decorator(f):
        COMPUTE_REGISTRY[key] = f
        return f
    return decorator


def get_function(record):
    """Resolve a record to its registered compute function.

    Returns the callable function, or None if resolve fails or
    the function key is not registered.
    """
    try:
        entry = resolve(record)
    except ValueError:
        return None
    fn = COMPUTE_REGISTRY.get(entry["compute"])
    return fn


# A ratio or power OF FLOW COEFFICIENTS (v4/v2^2, v2{4}/v2{2}^2, v6/v4) is
# not itself a flow coefficient: no v_n estimator can produce it, it needs a
# ratio of two independently measured coefficients.  Such records used to
# dispatch to an ordinary v_n key and silently return that coefficient
# instead -- the record claimed "v_4/v_2^2" while the pipeline computed v_2.
#
# Deliberately NOT matched: a coefficient divided by a NON-flow quantity
# (v_2/n_q, v_2/N_part). Those ARE normalised coefficients, and the v_n
# estimator is the right one for them; what they are missing is an explicit
# normalisation field on the record, which is a separate (flagged) issue.
# A v-token with an optional subscript and an optional cumulant-order
# suffix, so all of  v2 , {v_2} , v_{2} , v_2{4} , v_2{2}  are matched.
_VN_TOK = r'v_?\s*_?\s*\{?\s*\d\s*\}?\s*(?:\{\s*\d+\s*\})?'
_COEFF_RATIO = re.compile(
    rf'{_VN_TOK}\s*(?:/|\^|/\s*)\s*(?:\{{)?{_VN_TOK}',
    re.I)


def multiplicity_classes_invalid(classes):
    """True when classes in track count (centrality.multiplicity_bins) are
    not ranges 'lo-hi' of the count: no '%', 0 <= lo < hi, no overlap."""
    spans = []
    for c in list(classes or []):
        m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*', str(c))
        if not m:
            return True
        spans.append((float(m.group(1)), float(m.group(2))))
    if not spans or any(hi <= lo for lo, hi in spans):
        return True
    srt = sorted(spans)
    return any(srt[i][1] > srt[i + 1][0] + 1e-9 for i in range(len(srt) - 1))


def centrality_classes_suspicious(classes):
    """True when `centrality.classes` is not a list of percentile bins.

    2203 curated records carry something else in that field -- overwhelmingly
    kinematic bin edges with a spurious '%' suffix (jet xT bins such as
    ['0.31623-0.35481%', ...], pT bins, mT bins) -- plus 57 genuinely
    OVERLAPPING centrality selections and some unparseable labels.  All of
    it parses as numbers, so nothing downstream noticed: such a record is
    handed to the percentile binning as though it declared two CENTRALITY
    classes, and the pipeline returns confidently wrong numbers.

    A gapped but integer list ('0-5%', '30-40%') is LEGAL -- a publication
    need not tile the percentile axis, and weights.cent_weights slices such
    lists per class -- so only overlap, non-integer bounds, values outside
    0-100, or unparseable labels are flagged.
    """
    classes = list(classes or [])
    if not classes:
        return False
    spans = []
    for c in classes:
        parts = str(c).replace('%', '').split('-')
        if len(parts) != 2:
            return True
        try:
            lo, hi = float(parts[0]), float(parts[1])
        except ValueError:
            return True
        spans.append((lo, hi))
    if any(hi > 100.0 or lo < 0.0 or hi <= lo for lo, hi in spans):
        return True
    # A single label must be a percentile bin too: 'MB' (minimum bias) is
    # not a centrality class of the model's events (until November 2026
    # one-class lists were not checked, and 'MB' records passed as public).
    if len(spans) < 2:
        return False
    srt = sorted(spans)
    if any(srt[i][1] > srt[i + 1][0] + 1e-9 for i in range(len(srt) - 1)):
        return True
    # Fractional boundaries are NOT by themselves suspicious: STAR and ALICE
    # publish equal-multiplicity bins like '5-11.25%' and '42.5-60%', and
    # flagging those would block ~76 legitimate flow records.  A non-integer
    # list is only called suspicious when the WHOLE list lies inside a span
    # far too narrow to be a percentile axis -- a kinematic bin set such as
    # jet xT ['0.31623-0.35481%', ...] or pT ['0.3-0.4%', ...], which never
    # reaches a few percent.  Deliberately conservative: this gate exists to
    # catch UNAMBIGUOUS errors, not to guess.  The broader non-integer signal
    # is still reported by validate_metadata.py as a curation hint.
    if any(lo != int(lo) or hi != int(hi) for lo, hi in spans):
        # A genuine percentile list starts near the most-central bin and
        # runs well out into peripheral territory.  Kinematic bin sets do
        # neither: pT/xT/mT edges sit at arbitrary offsets
        # ('24.75-29.65%', '29.65-34.6%' -- mean-pT-fluctuation mT bins
        # starting at 25, or jet xT below 1).  Anything whose span does not
        # look like a centrality axis is treated as kinematic.
        lo_min = min(lo for lo, _ in spans)
        hi_max = max(hi for _, hi in spans)
        return not (lo_min <= 5.0 and hi_max >= 10.0)
    return False


def is_coefficient_ratio(value_type):
    """True when value_type is a ratio/power of two flow coefficients."""
    return bool(_COEFF_RATIO.search(str(value_type or '')))


def _has_flow_reference(record):
    """True when the record declares a reference sample.

    Includes `flow_reference.same_sample_as_poi`: the event plane built
    from the POI tracks themselves is still a two-set measurement needing
    the SP estimator and its diagonal subtraction, so it must not fall
    through to the single-set amplitude fallback.
    """
    fr = record.get("flow_reference")
    if not fr:
        return False
    if isinstance(fr, dict):
        return True
    return bool(fr)


# Compute keys that fall back to the SINGLE-SET amplitude estimator
# sqrt(<|Q_n|^2>/<M^2>) when a record declares no reference.  That estimator
# is a two-particle correlation over the POI sample, but it contains NO event
# plane and therefore no resolution factor -- it is NOT the observable an
# event_plane record reports.  Kept as an explicit, greppable set so the
# pipeline can label such output instead of passing it off as computed.
AMPLITUDE_FALLBACK_KEYS = frozenset(
    {'vn2_ep', 'vn3_ep', 'vn2_ep_diff_pT', 'vn2_ep_diff_eta'})


def reference_status(record):
    """'curated' | 'assumed' | 'absent' for a record's flow reference.

    'curated' -- a real reference window is declared (provenance: paper).
    'assumed' -- declared but inferred, e.g. STAR provenance=assumed_tpc.
    'absent'  -- nothing declared; a two-set method is being approximated by
                 the single-set amplitude estimator.
    """
    fr = record.get('flow_reference')
    if not fr:
        return 'absent'
    prov = fr.get('provenance') if isinstance(fr, dict) else None
    return 'assumed' if prov and str(prov).startswith('assumed') else 'curated'


# ── combinations of cumulant orders (v_n{6}/v_n{4}, differences, gamma1) ──

COMBINATION_KINDS = {'ratio': 2, 'difference': 2, 'gamma1': 3}
COMBINATION_ENTRY = {
    'compute': 'vn_cumulant_combination',
    'requires': [],
    'description': 'a combination of v_n{2k} of several orders (ratio, difference, skewness '
                   'estimate gamma1) from the same events',
}


def combination_terms(record):
    """The per-order sub-records of a ``cumulant_combination`` record.

    ``cumulant_combination``: kind ratio (v{a}/v{b}) | difference (v{a} -
    v{b}) | gamma1 (-6 sqrt2 v{4}^2 (v{4} - v{6}) / (v{2}^2 - v{4}^2)^{3/2},
    orders [2, 4, 6]); ``orders``; optional ``scale`` (a factor on the
    result, e.g. 1/11); optional ``two_particle``: {particles, subevents,
    eta_gap, flow_reference} for the v{2} term (ALICE's gamma1 uses v2{2,
    |Delta eta| > 1}; CMS's 2PC v{2} its own associated-pT reference);
    optional ``harmonics`` (ratio, difference): the harmonic of each term,
    e.g. orders [2, 2] with harmonics [3, 2] for v3{2}/v2{2} (default the
    record's harmonic for every term).
    Each term is the record itself as a single-order cumulant record (its
    acceptance, species, centrality), so it computes with the existing
    v_n{2k} keys.  Returns [(order, sub_record)].
    """
    import copy
    comb = record.get('cumulant_combination')
    if not isinstance(comb, dict):
        raise ValueError('cumulant_combination must be a mapping')
    kind = comb.get('kind')
    if kind not in COMBINATION_KINDS:
        raise ValueError(f"cumulant_combination.kind {kind!r}: one of {sorted(COMBINATION_KINDS)}")
    orders = [int(k) for k in (comb.get('orders') or ([2, 4, 6] if kind == 'gamma1' else []))]
    if len(orders) != COMBINATION_KINDS[kind] or any(k not in (2, 4, 6, 8) for k in orders):
        raise ValueError(f"cumulant_combination {kind}: orders {orders} (orders 2, 4, 6, 8)")
    if kind == 'gamma1' and orders != [2, 4, 6]:
        raise ValueError("cumulant_combination gamma1 takes orders [2, 4, 6]")
    if record.get('dimensionality', 'integrated') != 'integrated':
        raise ValueError("cumulant_combination is implemented for integrated records")
    harmonics = comb.get('harmonics')
    if harmonics is None:
        harmonics = [int(record.get('harmonic', 2))] * len(orders)
    else:
        if kind == 'gamma1':
            raise ValueError("cumulant_combination gamma1 takes the record's harmonic")
        harmonics = [int(h) for h in harmonics]
        if len(harmonics) != len(orders) or any(not 1 <= h <= 10 for h in harmonics):
            raise ValueError(f"cumulant_combination harmonics {harmonics}: one per order, 1-10")
    if len(set(zip(orders, harmonics))) != len(orders):
        raise ValueError(f"cumulant_combination: identical terms (orders {orders}, harmonics {harmonics})")
    out = []
    for k, h in zip(orders, harmonics):
        sub = copy.deepcopy({key: v for key, v in record.items() if key != 'cumulant_combination'})
        sub.update(method='cumulant', cumulant_order=k, harmonic=h, value_type=f"v{h}{{{k}}}")
        for key in ('particles', 'subevents', 'eta_gap'):
            sub.pop(key, None)
        if k == 2 and isinstance(comb.get('two_particle'), dict):
            sub.update(copy.deepcopy(comb['two_particle']))
        out.append((k, sub))
    return out


def resolve(record):
    """Look up dispatch entry for a record, validate required fields.

    First tries DISPATCH_TABLE keyed by (method, cumulant_order, particles, dim),
    then falls back to DISPATCH_TABLE_BY_TYPE keyed by observable_type.
    BY_TYPE entries may carry ``value_type_contains``: a substring that must
    appear in record['value_type'] (guards heterogeneous schemas where one
    observable_type spans several distinct measurements).

    Returns the entry dict with keys: compute, requires, description.
    Raises ValueError if no match or missing fields.
    """
    # A declared combination of cumulant orders (v2{6}/v2{4}, ...) is formed
    # from its terms, each an ordinary v_n{2k} record.
    if record.get('cumulant_combination') is not None:
        for _k, sub in combination_terms(record):
            resolve(sub)
        return dict(COMBINATION_ENTRY)

    # Classes in the estimator's track count are absolute: no percentile
    # machinery (actual ranges, a cross-section normalization, sub-class
    # combination, points) applies to them.
    cent_ = record.get("centrality") or {}
    if cent_.get("multiplicity_bins"):
        clash = [k for k in ("actual_classes", "normalization", "combined_from", "points") if cent_.get(k)]
        if clash:
            raise ValueError(f"centrality.multiplicity_bins with {', '.join(clash)}: classes in track count "
                             f"are absolute")
        if multiplicity_classes_invalid(cent_.get("classes")):
            raise ValueError(f"centrality.multiplicity_bins: classes {cent_.get('classes')} are not "
                             f"non-overlapping track-count ranges 'lo-hi'")

    # A total-momentum cut (kinematic_range.p: STAR's PID limit p < 1.6
    # GeV/c) is applied by the rapidity-odd v1 prepare only; elsewhere it
    # would be ignored silently.
    if (record.get("kinematic_range") or {}).get("p") is not None and not (
            record.get("observable_type") == "flow_vn" and record.get("harmonic") == 1
            and record.get("v1_even") is None):
        raise ValueError("a total-momentum cut (kinematic_range.p) is implemented for "
                         "rapidity-odd v1 only")

    # Refuse a ratio of flow coefficients rather than return a coefficient
    # that the record does not claim.  A silent substitution here is worse
    # than a visible failure: the value is plausible and the label is wrong.
    if is_coefficient_ratio(record.get('value_type')):
        raise ValueError(
            f"value_type {record.get('value_type')!r} is a ratio/power of "
            f"flow coefficients, which is not itself a coefficient; no v_n "
            f"estimator can produce it (needs a ratio of two measured "
            f"coefficients). Refusing to substitute v_n.")

    # Detector-level quantities (not corrected for efficiency/acceptance)
    # and flow relative to the spectator plane cannot be formed from a
    # model's particles; refuse them with the reason instead of returning
    # a plausible number of a different quantity.
    # A detector-level yield with a stated detector efficiency is computed
    # as efficiency x N (pipeline_dispatch: absolute quantities).
    if record.get("efficiency_corrected") is False and not (
            record.get("detector_efficiency")
            and record.get("observable_type") in ("particle_yields", "particle_spectra")):
        raise ValueError(
            f"{record.get('value_type')!r} is detector-level (not corrected "
            f"for efficiency); it needs a detector-response model")
    fr = record.get("flow_reference")
    if isinstance(fr, dict) and fr.get("plane") == "spectator":
        raise ValueError(
            f"{record.get('value_type')!r} is measured relative to the "
            f"spectator plane ({fr.get('detector')}); a model without "
            f"spectators cannot form that reference")
    # Class values combined from narrow multiplicity bins (STAR's
    # centrality-bin-width correction): the pipeline computes over the
    # whole class, which keeps the multiplicity fluctuations inside it.
    cent = record.get("centrality")
    est = (cent or {}).get("estimator") if isinstance(cent, dict) else None
    if isinstance(est, dict) and est.get("observable") == "spectator_neutrons":
        raise ValueError(
            f"{record.get('value_type')!r} classes come from spectator "
            f"neutrons ({est.get('detector')}); a model without spectators "
            f"cannot form that centrality")
    if isinstance(cent, dict) and cent.get("combined_from") == "unit_multiplicity":
        raise ValueError(
            f"{record.get('value_type')!r} combines values measured in "
            f"unit-multiplicity bins; the pipeline computes over the whole "
            f"centrality class")
    # Classes averaged from narrower percentile classes (ALICE 1%, PHENIX
    # 10%): computed per sub-class and combined (analysis.cent_combine);
    # sub-classes that do not tile every class are refused.
    combination_plan(cent)

    # A species the simulations cannot provide (heavy flavour, quarkonia,
    # leptons, photons, eta, resonances, ...) would compute as zero, not be
    # refused.  (pi0 and K0_S resolve by isospin, analysis.species.)
    species = (record.get("particle_id") or {}).get("species")
    if species is not None and not species_known(species):
        raise ValueError(
            f"species {species!r} is not produced by the pipeline "
            f"(analysis.species lists the supported ones)")

    # Spectra: the invariant yield or d2N/(dpT dy) (value_type); another
    # form would be compared with the invariant yield.
    if record.get("observable_type") == "particle_spectra" and record.get(
            "value_type", "invariant_yield") not in ("invariant_yield", "dN_dpT_dy", "d2N_dpT_dy"):
        raise ValueError(
            f"particle_spectra value_type {record.get('value_type')!r}: the pipeline computes "
            f"'invariant_yield' (1/(2 pi pT) d2N/(dpT dy)) or 'dN_dpT_dy'")

    # Directed flow.  Rapidity-odd v1 needs an eta-odd reference (windows at
    # eta < 0 weighted -1) and a sign convention; the even-harmonic
    # machinery would return ~0 or an unrelated number.  Rapidity-even v1
    # (v1_even) is the factorization of the model's v11 matrix: the
    # experiments' fit without its momentum-conservation term, which the
    # models lack (analysis/v1_even.py).
    if record.get("observable_type") == "flow_vn" and record.get("harmonic") == 1 \
            and record.get("v1_even") is not None:
        _check_v1_even(record)
        return dict(V1_EVEN_ENTRY)
    if record.get("observable_type") == "flow_vn" and record.get("harmonic") == 1:
        if not (isinstance(fr, dict) and fr.get("odd")):
            raise ValueError(
                "directed flow (harmonic 1) needs a rapidity-odd reference "
                "(flow_reference.odd) or, for rapidity-even v1, v1_even")
        if record.get("dimensionality") not in ("differential_eta", "differential_y"):
            raise ValueError("rapidity-odd v1 is implemented for eta- or y-differential records")
        if record.get("dimensionality") == "differential_y" and (
                species is None or is_charged(species)):
            raise ValueError("rapidity-odd v1 in rapidity bins needs an identified species "
                             "(the mass of the eta <-> y mapping)")
        sc = fr.get("sign_convention")
        if not isinstance(sc, dict):
            raise ValueError("rapidity-odd v1 needs flow_reference.sign_convention "
                             "(an odd reference leaves the overall sign open)")
        if sc.get("y") is not None and is_charged(sc.get("species") or species or "charged"):
            raise ValueError("a sign-convention rapidity window needs an identified species")
        return dict(V1_ODD_ENTRY)

    method = record.get("method")
    order = record.get("cumulant_order")
    particles = record.get("particles")
    dim = record.get("dimensionality", "integrated")
    # If cumulant_order is missing but harmonic is set, derive it.
    # For v_n{2} records, harmonic=2 (or omitted when value_type is v2(2)).
    if order is None:
        harm = record.get("harmonic")
        if harm is not None:
            order = harm
    # --- event_plane records that DECLARE a reference are two-set ---
    # The event-plane / subevent method IS a reference-vs-POI correlation:
    # v_n is built from Q_n of a *separate* acceptance window and therefore
    # carries the event-plane resolution factor.  The SP path implements
    # that correlation and is the only one that consumes `flow_reference`
    # (per-bin windows, `segments`, `mirror`).  The event_plane compute
    # keys are the single-set amplitude sqrt(<|Q_n|^2>/<M^2>), which
    # contains no event plane and whose prepare ignores flow_reference
    # outright -- so routing a reference-bearing record there would
    # silently discard the reference the record declares.
    # Records with no declared reference keep the amplitude path, which is
    # then explicitly only an approximation of the reported observable.
    _fr = record.get("flow_reference") if isinstance(record.get("flow_reference"), dict) else {}
    # v_n{2}(eta) with side-dependent reference regions normalized by a
    # partner region (ALICE 2679248): analysis.eta_regions.
    if method == "cumulant" and order == 2 and particles is None and \
            any("partner" in (s or {}) for s in (_fr.get("segments") or [])):
        method = "cumulant_partner"
    # Flow relative to a lower-order plane (STAR's v2{Psi_1} at 3-4.5 GeV):
    # the mixed-harmonic estimator (analysis.mixed_plane).
    if method == "event_plane" and _fr.get("harmonic") is not None and \
            int(_fr["harmonic"]) != int(record.get("harmonic") or 0):
        method, order = "event_plane_mixed", None
    elif method == "event_plane" and _has_flow_reference(record):
        # The SP entries are keyed without a harmonic because the cores
        # return EVERY harmonic: run_pipeline indexes them as
        # (harm - 1) * n_bins, so a v_3/v_4 record served by vn2_sp reads
        # its own harmonic rather than being downgraded to v_2.  That makes
        # the order-less SP entry correct for any harmonic.
        _sp_key = ("scalar_product", None, particles, dim)
        if _sp_key in DISPATCH_TABLE:
            method = "scalar_product"
            order = None
    # An event plane from the opposite subevent (STAR eta-sub: the POI in
    # one TPC half, the plane from the other), or a scalar product of the
    # two subevents (ALICE p-Pb), declared as subevents (particles: AB):
    # the two-subevent correlation, v_POI = <u_A Q_B*> / sqrt(<Q_A Q_B*>),
    # the event-plane result at low resolution (as the SP estimator serves
    # full-reference event-plane records).
    elif method in ("event_plane", "scalar_product") and particles == "AB":
        method, order = "cumulant", 2
    key = (method, order, particles, dim)
    entry = DISPATCH_TABLE.get(key)
    if entry is None:
        if method != "cumulant":
            key = (method, None, particles, dim)
            entry = DISPATCH_TABLE.get(key)
    if entry is None:
        # Fall back to type-based dispatch for non-flow observables
        obs_type = record.get("observable_type")
        entry = DISPATCH_TABLE_BY_TYPE.get(obs_type)
        if entry is not None:
            # If `compute` is a dict, route by the record's correlation_type
            # (or other field).  This lets one observable_type span multiple
            # sub-types (e.g. flow_harmonic_correlations has symmetric_cumulant
            # and event_plane_correlation variants).
            comp = entry.get("compute")
            ct = record.get("correlation_type")
            if isinstance(comp, dict):
                comp = comp.get(ct) if ct is not None else None
                if comp is None:
                    raise ValueError(
                        f"{obs_type} record correlation_type "
                        f"'{ct}' not routable (known: {list(entry['compute'].keys())})")
                entry = dict(entry)
                entry["compute"] = comp
                # `requires` may also be per-correlation_type
                reqs_per = entry.get("requires_per_ct", {})
                if ct in reqs_per:
                    entry["requires"] = reqs_per[ct]
            # Symmetric cumulants: SC(m, n) only, and the record must say
            # whether it is normalized (NSC = SC / <<2>>_m <<2>>_n); until
            # November 2026 NSC records were compared with unnormalized SC.
            if entry.get("compute") == "sc_mn":
                harms = record.get("harmonics") or []
                if len(harms) != 2:
                    raise ValueError(f"symmetric cumulant of {len(harms)} harmonics {harms}: only SC(m, n) "
                                     f"is implemented")
                if "normalized" not in record:
                    raise ValueError("symmetric_cumulant record needs 'normalized' (true: NSC, false: SC)")
            # Linear and mode-coupled harmonics: [k, n, m] with k = n + m, or
            # [k, n, m, l] with k = n + m + l, a known quantity, two subevents
            # A, B or one window, pT-differential or integrated
            # (analysis.mode_coupling).
            if entry.get("compute") == "mode_coupling":
                from analysis.mode_coupling import QUANTITIES, harmonics_ok
                harms = [int(h) for h in (record.get("harmonics") or [])]
                if not harmonics_ok(harms):
                    raise ValueError(f"mode coupling: harmonics {harms} must be [k, n, m] (k = n + m) or "
                                     "[k, n, m, l] (k = n + m + l)")
                if record.get("quantity") not in QUANTITIES:
                    raise ValueError(f"mode coupling: quantity {record.get('quantity')!r} (one of "
                                     f"{', '.join(QUANTITIES)})")
                if record.get("particles") not in (None, "AB"):
                    raise ValueError(f"mode coupling: particles {record.get('particles')!r} (AB or one window)")
                if record.get("dimensionality", "integrated") not in ("integrated", "differential_pT"):
                    raise ValueError("mode coupling: integrated or differential_pT")
                if record.get("chi_denominator") not in (None, "P4", "factorized"):
                    raise ValueError(f"mode coupling: chi_denominator {record.get('chi_denominator')!r} "
                                     "(P4 | factorized)")
            # Symmetry-plane correlations: the scalar-product method only (the
            # event-plane method depends on the detector's event-plane
            # resolution, which a model does not have); planes [(n, c)], two or
            # three, sum c n = 0, harmonics their n (analysis.plane_correlations).
            # pT-dependent flow-vector fluctuations (ALICE 2771093): one
            # harmonic, pT bins, the correlators each quantity needs
            # (analysis.flow_vector_fluct).
            if entry.get("compute") == "rnn_decorrelation":
                if record.get("quantity") != "r_nn":
                    raise ValueError(f"longitudinal decorrelation: quantity {record.get('quantity')!r} (r_nn: the "
                                     "opposite- over same-side correlation; others not computed)")
                if len(record.get("harmonics") or []) != 1:
                    raise ValueError("r_{n|n}: one harmonic")
                if record.get("dimensionality") != "differential_eta":
                    raise ValueError("r_{n|n}: dimensionality differential_eta (|eta| bins)")
            if entry.get("compute") == "flow_vector_fluct":
                from analysis.flow_vector_fluct import QUANTITIES, needs
                q = record.get("quantity")
                if q not in QUANTITIES:
                    raise ValueError(f"flow-vector fluctuations: quantity {q!r} (one of {', '.join(QUANTITIES)})")
                if len(record.get("harmonics") or []) != 1:
                    raise ValueError("flow-vector fluctuations: one harmonic")
                if record.get("dimensionality") != "differential_pT":
                    raise ValueError("flow-vector fluctuations: dimensionality differential_pT (POI pT bins)")
                if ("two" in needs(q) or "trigger" in needs(q)) and record.get("particles") != "AB":
                    raise ValueError(f"flow-vector fluctuations ({q}): two-particle subevents (particles: AB)")
                if "trigger" in needs(q) and not record.get("trigger_pT"):
                    raise ValueError("factorization ratio r_n needs trigger_pT")
                if "four" in needs(q) and not record.get("four_particle_subevents"):
                    raise ValueError(f"flow-vector fluctuations ({q}): four_particle_subevents A and B")
            if entry.get("compute") == "plane_corr_sp":
                if record.get("estimator") != "SP":
                    raise ValueError(f"plane correlation: estimator {record.get('estimator')!r}; only the "
                                     "scalar-product method (SP) is computed (EP depends on the detector's "
                                     "event-plane resolution)")
                planes = record.get("planes") or []
                try:
                    pl = [(int(p["harmonic"]), int(p["coefficient"])) for p in planes]
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"plane correlation: planes {planes!r} need harmonic and coefficient")
                if len(pl) not in (2, 3) or any(c == 0 or n < 1 for n, c in pl) or sum(n * c for n, c in pl):
                    raise ValueError(f"plane correlation: planes {pl} must be two or three (n, c), n >= 1, "
                                     "c != 0, sum c n = 0")
                if sorted(int(h) for h in record.get("harmonics") or []) != sorted(n for n, _ in pl):
                    raise ValueError(f"plane correlation: harmonics {record.get('harmonics')} are not the "
                                     f"planes' {[n for n, _ in pl]}")
            # For particle_yields, choose per-eta vs integrated based on
            # whether the record has eta bin edges (differential_eta).
            if obs_type == "particle_yields":
                kr = record.get("kinematic_range", {})
                if len(axis_bins(kr.get("eta", {}))) > 1 or len(axis_bins(kr.get("y", {}))) > 1:
                    # Per-eta yield: needs the per-eta compute key.
                    entry = dict(entry)
                    entry["compute"] = "yield_dNch_deta_per_eta"
            want = entry.get("value_type_contains")
            if want is not None and want not in str(
                    record.get("value_type", "")):
                raise ValueError(
                    f"{obs_type} record value_type "
                    f"'{record.get('value_type')}' not routable to "
                    f"'{entry['compute']}' (needs substring '{want}')")
            want_ct = entry.get("correlation_type_eq")
            if want_ct is not None and ct != want_ct:
                raise ValueError(
                    f"{obs_type} record correlation_type "
                    f"'{ct}' not routable to "
                    f"'{entry['compute']}' (needs '{want_ct}')")
    if entry is None:
        raise ValueError(
            f"No dispatch entry for method={method}, "
            f"order={order}, particles={particles}, dim={dim}, "
            f"observable_type={record.get('observable_type')}")

    reqs = entry["requires"]
    if isinstance(reqs, str):
        reqs = [reqs]
    for req in reqs:
        if isinstance(req, (list, tuple)):
            if not any(_field_exists(record, alt) for alt in req):
                raise ValueError(
                    f"Record missing required fields (need one of {req}) for dispatch "
                    f"key {key}")
        elif not _field_exists(record, req):
            raise ValueError(
                f"Record missing required field '{req}' for dispatch "
                f"key {key}")

    # Two-subevent v_n{2} of identified particles: the particle of interest
    # (the record's species and kinematic_range) in one subevent, correlated
    # with the reference (charged hadrons, or subevents.<X>.particle_id) in
    # the other, and divided by the reference v_n{2} -- vn2_gap would return
    # the reference's own v_n{2}.
    if entry["compute"] == "vn2_gap" and species is not None and not is_charged(species):
        entry = dict(VN2_GAP_POI_ENTRY)
        for req in entry["requires"]:
            if not _field_exists(record, req):
                raise ValueError(f"Record missing required field '{req}' for vn2_gap_poi")

    # A rapidity window with an assumed mass (e.g. y_pi) is implemented for
    # spectra only; elsewhere it would be read as a true-rapidity window.
    y_kr = (record.get("kinematic_range") or {}).get("y") or {}
    if y_kr.get("mass") is not None and entry["compute"] not in MASS_ASSUMED_RAPIDITY_KEYS:
        raise ValueError(
            f"kinematic_range.y.mass (rapidity with an assumed mass) is not "
            f"implemented for '{entry['compute']}'")

    return entry


# Computations that implement kinematic_range.y.mass.
MASS_ASSUMED_RAPIDITY_KEYS = {"spectra_invariant_yield", "yield_dNch_deta", "yield_dNch_deta_per_eta",
                              "mean_pt"}

VN2_GAP_POI_ENTRY = {
    "compute": "vn2_gap_poi",
    "requires": [
        "subevents.A.kinematic_range.eta",
        "subevents.B.kinematic_range.eta",
        "kinematic_range.pT",
    ],
    "description": "v_n{2} of identified particles, two subevents: POI in one subevent with the "
                   "reference in the other, over the reference v_n{2}ab",
}

V1_EVEN_ENTRY = {
    "compute": "v1_even_factorization",
    "requires": "kinematic_range.pT",
    "description": "rapidity-even directed flow v1^even: rank-1 factorization of the v11 "
                   "pair matrix (no momentum-conservation term), sign by convention",
}


def _check_v1_even(record):
    """Refuse a v1_even block the computation cannot honour."""
    ve = record.get("v1_even")
    if not isinstance(ve, dict) or ve.get("fit") != "factorization":
        raise ValueError("v1_even.fit must be 'factorization' (v11 = v1(pTa) v1(pTb), the "
                         "momentum-conservation term subtracted by the experiment)")
    dim = record.get("dimensionality", "integrated")
    if dim not in ("integrated", "differential_pT"):
        raise ValueError(f"rapidity-even v1 is implemented vs centrality and vs pT, not {dim}")
    grid = ve.get("fit_pT_bins")
    if grid is not None and (not isinstance(grid, list) or not all(
            isinstance(b, (list, tuple)) and len(b) == 2 and float(b[0]) < float(b[1]) for b in grid)):
        raise ValueError("v1_even.fit_pT_bins: a list of [lo, hi] pT bins")
    if ve.get("absolute"):
        return
    sc = ve.get("sign_convention")
    if not (isinstance(sc, dict) and isinstance(sc.get("pT"), dict)
            and sc.get("sign") in ("positive", "negative")):
        raise ValueError("rapidity-even v1 leaves the sign open: v1_even.sign_convention "
                         "{pT: {min, max}, sign: positive | negative}, or absolute: true for |v1|")
    n_grid = len(grid) if grid else max(len(axis_bins((record.get("kinematic_range") or {}).get("pT"))), 1)
    if n_grid < 2:
        raise ValueError("a signed v1^even needs a fit grid of two or more pT bins "
                         "(v1_even.fit_pT_bins, or the record's own bins)")


V1_ODD_ENTRY = {
    "compute": "v1_odd_sp_diff_eta",
    "requires": "kinematic_range.pT",
    "description": "rapidity-odd directed flow v1(eta) or v1(y): scalar product with an "
                   "eta-odd charged reference, segments aligned, sign by convention (3D only)",
}


def _field_exists(doc, path):
    parts = path.split(".")
    obj = doc
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return False
    if isinstance(obj, (list, tuple)):
        return len(obj) > 0
    return obj is not None


# ── Record helpers (moved from compare_phenix_fvtx.py) ──────────────────────

def load_record(record_path, value_type_substring, dimensionality=None):
    """Load the first curated record whose value_type contains the substring.

    If multiple docs match and dimensionality is given, prefer the one
    whose dimensionality field matches exactly.
    """
    with open(record_path) as f:
        docs = list(yaml_load_all(f))
    candidates = []
    for doc in docs:
        if doc is None:
            continue
        doc["_record_file"] = os.path.abspath(record_path)   # resolves a relative value_table
        vt = doc.get("value_type", "")
        if value_type_substring in vt:
            if dimensionality is None or doc.get("dimensionality") == dimensionality:
                candidates.append(doc)
    if not candidates and dimensionality is not None:
        # Fallback: accept any dimensionality match
        return load_record(record_path, value_type_substring)
    return candidates[0] if candidates else None


def _plan_or_none(cent):
    try:
        return combination_plan(cent)
    except ValueError:
        return None


def extract_measurement_params(exp_rec):
    """Read ALL measurement parameters from a curated flow_vn record.

    Returns a dict describing what to compute and with which cuts.
    """
    top_kr = exp_rec.get('kinematic_range', {})
    particles = exp_rec.get('particles', None)
    cent = exp_rec.get('centrality', {})

    subevent_ranges = {}
    subevents = exp_rec.get('subevents', {})
    if particles and subevents:
        for label in set(particles):
            sk = subevents.get(label, {}).get('kinematic_range', top_kr)
            subevent_ranges[label] = {
                'eta': _kr_eta(sk),
                'pT': _kr_pT(sk),
                # species of this subevent's particles (None: the record's)
                'particle_id': subevents.get(label, {}).get('particle_id'),
            }

    if subevent_ranges:
        first_label = sorted(subevent_ranges.keys())[0]
        single_kr = subevent_ranges[first_label]
    else:
        eta_lo, eta_hi = _kr_eta(top_kr)
        # NOTE: DO NOT fall back to cent.eta. The centrality estimator's η
        # acceptance is for the centrality detector (BBC, V0, FCal), NOT
        # for the particles of interest. All records must have explicit
        # kinematic_range.eta (filled in during curation).
        pt_lo, pt_hi = _kr_pT(top_kr)
        if pt_lo is None:
            pt_lo = 0.1
        if pt_hi is None:
            pt_hi = 10.0
        single_kr = {
            'eta': (eta_lo, eta_hi),
            'pT': (pt_lo, pt_hi),
        }
    # Rapidity-window observables (dN/dy from flat-y sims): expose the
    # record's kinematic_range.y alongside eta so prepare functions can
    # take a dy-mode branch.
    top_y = top_kr.get('y', {}) or {}
    single_kr['y'] = (top_y.get('min'), top_y.get('max'))
    # A rapidity window computed with one assumed mass for every particle
    # (e.g. PHOBOS y_pi): kinematic_range.y.mass, in GeV.
    single_kr['y_mass'] = top_y.get('mass')
    cent_eta_raw = cent.get('eta', [])
    cent_eta_ranges = [(e['min'], e['max']) for e in cent_eta_raw]
    cent_pt_raw = cent.get('pT') or {}
    cent_pT_range = (cent_pt_raw.get('min'), cent_pt_raw.get('max'))
    cent_classes = cent.get('classes', [])
    # Percentile edges as floats: classes such as 0.15-0.2% or 5-11.25%
    # are common (ALICE, ATLAS, CMS fine bins).  Until November 2026 they
    # were truncated to integers, so 0.15-0.2% became an empty 0-0% bin and
    # 5-11.25% became 5-11%.
    # Per-system actual percentile ranges (STAR integer-Ntrk classes: label
    # 40-50%, actual 39.89-49.86% in Ru+Ru) assign the model's classes; the
    # labels stay nominal.
    actual = cent.get('actual_classes')
    if actual:
        if len(actual) != len(cent_classes):
            raise ValueError(f"centrality.actual_classes has {len(actual)} entries for "
                             f"{len(cent_classes)} classes")
    cent_edges = []
    for c in (actual or cent_classes):
        parts = c.replace('%', '').split('-')
        try:
            cent_edges.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            cent_edges.append((0.0, 100.0))  # fallback: MB

    # Flag a centrality.classes that is not centrality.  The record is still
    # returned (so schema validation and export see it) but the pipeline must
    # not build percentile bins from it -- see centrality_classes_suspicious.
    # Classes in the estimator's track count (multiplicity_bins) have their
    # own check: numeric ranges without '%'.
    mult_bins = cent.get('multiplicity_bins') or None
    cent_suspicious = (multiplicity_classes_invalid(cent_classes) if mult_bins
                       else centrality_classes_suspicious(cent_classes))

    dim = exp_rec.get('dimensionality', 'integrated')
    diff_var, diff_bins, diff_fixed = extract_differential_bins(exp_rec)

    combination = None
    if exp_rec.get('cumulant_combination') is not None:
        comb = exp_rec['cumulant_combination']
        combination = {'kind': comb.get('kind'), 'scale': float(comb.get('scale', 1.0)),
                       'terms': [{'order': k, 'key': resolve(sub)['compute'],
                                  'params': extract_measurement_params(sub)}
                                 for k, sub in combination_terms(exp_rec)]}

    return {
        # terms of a combination of cumulant orders (None if not)
        'cumulant_combination': combination,
        # Average of particle and antiparticle, e.g. (h+ + h-)/2: absolute
        # quantities are half of the sum (pipeline_dispatch._with_charge_average).
        'charge_average': bool((exp_rec.get('particle_id') or {}).get('charge_average')),
        'harmonic': exp_rec.get('harmonic', 1),
        'harmonics': exp_rec.get('harmonics'),
        'particles': particles,
        'particle_id': exp_rec.get('particle_id'),
        'subevent_ranges': subevent_ranges,
        'single_range': single_kr,
        # The particle of interest's own acceptance (top-level
        # kinematic_range); single_range is subevent A's when subevents exist.
        'poi_range': {'eta': _kr_eta(top_kr), 'pT': _kr_pT(top_kr)},
        # Total-momentum window of the particle of interest (kinematic_range.p;
        # rapidity-odd v1 only, resolve() refuses it elsewhere)
        'momentum_range': _kr_p(top_kr),
        'dimensionality': dim,
        'differential_var': diff_var,
        'differential_bins': diff_bins,
        'differential_fixed_range': diff_fixed,
        'eta_gap': exp_rec.get('eta_gap', None),
        'has_gap': (exp_rec.get('eta_gap', 0) or 0) > 0,
        # rapidity-even v1: fit grid, sign convention or |v1|
        'v1_even': exp_rec.get('v1_even'),
        'flow_reference': _flow_reference(exp_rec),
        # the record's own flow_reference (the mixed-plane subevents)
        'flow_reference_raw': exp_rec.get('flow_reference') if isinstance(exp_rec.get('flow_reference'), dict)
        else None,
        # Reference pT window (also without reference eta windows, which
        # _flow_reference requires): the cumulant keys' reference sample.
        'flow_reference_pT': _ref_pt(exp_rec),
        'flow_reference_eta': _ref_eta(exp_rec),
        'cent_eta_ranges': cent_eta_ranges,
        'cent_pT_range': cent_pT_range,
        'cent_classes': cent_classes,
        'cent_classes_suspicious': cent_suspicious,
        'cent_edges': cent_edges,
        # sub-classes of classes combined from narrower ones (None if not);
        # an invalid plan is refused by resolve()
        'cent_combine': _plan_or_none(cent),
        'cent_estimator': cent.get('estimator', {}),
        'hepdata_file': exp_rec.get('hepdata_file', ''),
        'value_table': exp_rec.get('value_table'),
        # false: the measurement exists only as a ratio (the linked ratio
        # record holds the values); the record computes and loads none
        'experimental_values': exp_rec.get('experimental_values', True) is not False,
        'record_file': exp_rec.get('_record_file'),
        'status': exp_rec.get('status', 'published'),
        'inspire_id': exp_rec.get('inspire_id', 0),
        'version': int(exp_rec.get('version', 1)),
        'value_type': exp_rec.get('value_type', ''),
        'dv_index': int(exp_rec.get('dv_index', 0) or 0),
        'dv_average': (list(exp_rec.get('dv_average', None) or [])
                       if exp_rec.get('dv_average', None) else None),
        # Rows of the table this record holds (ascending), e.g. the 0-5..70-80%
        # rows of a table that also has an overlapping 0-1% row.
        'table_rows': ([int(r) for r in exp_rec['table_rows']]
                       if exp_rec.get('table_rows') is not None else None),
        'value_scale': exp_rec.get('value_scale'),
        # error components contained in another one (not added again)
        'errors_excluded': exp_rec.get('errors_excluded'),
        # radial_flow: the quantity, the [pT] reference range, and the
        # multiplicity at fixed value of which deviations are taken
        'quantity': exp_rec.get('quantity'),
        'reference_pT': _kr_pT({'pT': exp_rec['reference_pT']}) if exp_rec.get('reference_pT') else None,
        'multiplicity_baseline': exp_rec.get('multiplicity_baseline'),
        # pT fluctuations: pooled pairs (ALICE) or the event average (STAR)
        'averaging': exp_rec.get('averaging'),
        # rho(v_n^2, [pT]): the cov method, the var(v_n^2) terms, the c_k window
        'covariance_method': exp_rec.get('covariance_method'),
        'variance_terms': exp_rec.get('variance_terms'),
        'ck_window': exp_rec.get('ck_window'),
        # symmetric cumulants: NSC (divided by <<2>>_m <<2>>_n) and the eta
        # gap of those two-particle terms
        'normalized': bool(exp_rec.get('normalized')),
        'normalization_eta_gap': exp_rec.get('normalization_eta_gap'),
        # mode coupling: chi's denominator, <v_n^2 v_m^2> (default) or
        # 'factorized' <v_n^2><v_m^2> (STAR's differential chi_{5,23})
        'chi_denominator': exp_rec.get('chi_denominator'),
        # symmetry-plane correlations: the planes [{harmonic, coefficient}],
        # their subevents {name: [eta ranges]} and the method (SP)
        'planes': exp_rec.get('planes'),
        'plane_subevents': exp_rec.get('plane_subevents'),
        'estimator': exp_rec.get('estimator'),
        # pT-dependent flow-vector fluctuations: the four-particle subevents
        # and the trigger pT of r_n
        'four_particle_subevents': exp_rec.get('four_particle_subevents'),
        'trigger_pT': _kr_pT({'pT': exp_rec['trigger_pT']}) if exp_rec.get('trigger_pT') else None,
        # r_{n|n}(eta): the reference window at +-|eta_ref|
        'decorrelation_reference': exp_rec.get('decorrelation_reference'),
        # Experiment-defined centrality normalization (threshold_anchor) and
        # a detector efficiency for detector-level yields.
        'cent_normalization': cent.get('normalization'),
        # classes in the estimator's track count, N_trk = Poisson(efficiency
        # x N) (analysis.weights.MultiplicityBinWeights); None for percentiles
        'cent_multiplicity_bins': mult_bins,
        'detector_efficiency': exp_rec.get('detector_efficiency'),
        # Asymmetric systems: frame of the record's windows (lab / nn_cms),
        # the centre-of-mass rapidity in the lab, and the +eta beam.
        'frame': (exp_rec.get('collision_system') or {}).get('frame'),
        'rapidity_shift': (exp_rec.get('collision_system') or {}).get('rapidity_shift'),
        'forward_beam': (exp_rec.get('collision_system') or {}).get('forward_beam'),
    }


def _ref_windows(eta_field):
    """Normalize a flow_reference eta field to [(lo, hi), ...].

    Accepts a single {min, max} mapping or a list of them, and drops
    entries that are not usable windows.
    """
    if not eta_field:
        return []
    cands = eta_field if isinstance(eta_field, list) else [eta_field]
    out = []
    for w in cands:
        if not isinstance(w, dict):
            continue
        lo, hi = w.get('min'), w.get('max')
        if lo is None or hi is None:
            continue
        out.append((float(lo), float(hi)))
    return out


def _ref_eta(exp_rec):
    """One eta window of the record's flow_reference (the reference of the
    particle-of-interest cumulants, CMS: |eta| < 2.4 for POI in |eta| < 1),
    or None."""
    fr = exp_rec.get('flow_reference') if isinstance(exp_rec.get('flow_reference'), dict) else {}
    eta = (fr or {}).get('eta')
    if isinstance(eta, dict):
        eta = [eta]
    if not eta or len(eta) != 1 or eta[0].get('min') is None or eta[0].get('max') is None:
        return None
    return (float(eta[0]['min']), float(eta[0]['max']))


def _ref_pt(exp_rec):
    pt = ((exp_rec.get('flow_reference') or {}) if isinstance(exp_rec.get('flow_reference'), dict) else {}).get('pT') or {}
    if pt.get('min') is None or pt.get('max') is None:
        return None
    return (float(pt['min']), float(pt['max']))


def _flow_reference(exp_rec):
    """Read the record's flow reference into a normalized structure.

    Returns None when the record declares none (self-contained methods, or
    a reference equal to the particle-of-interest acceptance).  Otherwise

        {'detector': str|None, 'mirror': bool, 'eta': [(lo, hi), ...],
         'pT': (lo, hi)|None, 'notes': str|None,
         'segments': [{... 'applies_to_eta': (lo, hi)}, ...]}

    `mirror` means the reference is the eta-mirrored image of `eta` for
    each differential bin (the "sub-event on the opposite side" method).
    """
    raw = exp_rec.get('flow_reference')
    if not isinstance(raw, dict):
        return None

    def _one(d, extra=None):
        out = {
            'detector': d.get('detector'),
            'mirror': bool(d.get('mirror', False)),
            'eta': _ref_windows(d.get('eta')),
            'pT': None,
            'notes': d.get('notes'),
        }
        pt = d.get('pT') or {}
        if pt.get('min') is not None and pt.get('max') is not None:
            out['pT'] = (float(pt['min']), float(pt['max']))
        if extra:
            out.update(extra)
        return out

    out = _one(raw)
    # Rapidity-odd reference (directed flow): windows at eta < 0 enter with
    # weight -1.  sign_convention fixes the overall sign, which an odd
    # reference leaves open: ((lo, hi), +1 or -1) -- v1 averaged over that
    # POI window has that sign.
    out['odd'] = bool(raw.get('odd', False))
    # The window is in eta or (an identified species) in rapidity; its
    # species and pT default to the record's (STAR BES: pi+ + pi- at
    # 0 < y < 1, negative).
    sc = raw.get('sign_convention') or {}
    sc_var = 'y' if sc.get('y') is not None else 'eta'
    sc_win = sc.get(sc_var) or {}
    if sc_win.get('min') is not None and sc_win.get('max') is not None and sc.get('sign') in ('positive', 'negative'):
        out['sign_convention'] = ((float(sc_win['min']), float(sc_win['max'])),
                                  1 if sc['sign'] == 'positive' else -1)
        sc_pt = sc.get('pT') or {}
        out['sign_var'] = sc_var
        out['sign_species'] = sc.get('species')
        out['sign_pT'] = ((float(sc_pt['min']), float(sc_pt['max']))
                          if sc_pt.get('min') is not None and sc_pt.get('max') is not None else None)
    else:
        out['sign_convention'] = None
    segs = raw.get('segments')
    if isinstance(segs, list):
        parsed = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            rng = s.get('applies_to_eta') or {}
            if rng.get('min') is None or rng.get('max') is None:
                continue
            seg = _one(s, {'applies_to_eta': (float(rng['min']),
                                              float(rng['max']))})
            if seg['eta']:
                parsed.append(seg)
        out['segments'] = parsed
    else:
        out['segments'] = []
    if not out['eta'] and not out['segments']:
        return None
    return out


def _kr_p(kr):
    p = (kr or {}).get('p') or {}
    if p.get('min') is None and p.get('max') is None:
        return None
    return (float(p.get('min') or 0.0), float(p['max']) if p.get('max') is not None else float('inf'))


def _kr_eta(kr):
    e = kr.get('eta', {})
    lo, hi = e.get('min'), e.get('max')
    return lo, hi


def _kr_pT(kr):
    p = kr.get('pT', {})
    return p.get('min'), p.get('max')


def axis_bins(axis):
    """(lo, hi) bins of a kinematic_range axis.

    Consecutive `bins` edges; explicit `bin_ranges` [lo, hi] pairs, for bins
    that are not contiguous (e.g. PHOBOS eta bins with gaps between
    detectors); or, for a measurement published at mean positions without
    bin edges (e.g. PHOBOS v2 at <eta>), windows of width `point_window`
    centred on its `points`, a curation choice the record states.
    """
    if not isinstance(axis, dict):
        return []
    edges = axis.get('bins') or []
    if len(edges) > 1:
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    ranges = axis.get('bin_ranges') or []
    if ranges:
        return [(float(lo), float(hi)) for lo, hi in ranges]
    points = axis.get('points') or []
    if points:
        w = axis.get('point_window')
        if not w:
            raise ValueError('kinematic_range points need a point_window')
        return [(p - w / 2, p + w / 2) for p in points]
    return []


def extract_differential_bins(exp_rec):
    """Extract differential bins from a differential record (see axis_bins).

    Returns:
        var: 'pT' or 'eta' — the differential variable, or None if integrated
        bins: list of (lo, hi) tuples for each bin
        fixed: (min, max) for the non-differential variable (fixed acceptance)
    """
    kr = exp_rec.get('kinematic_range', {}) or {}
    pT = kr.get('pT', {}) or {}
    eta = kr.get('eta', {}) or {}
    y = kr.get('y', {}) or {}
    dim = exp_rec.get('dimensionality')
    if dim == 'differential_pT':
        return 'pT', axis_bins(pT), (eta.get('min'), eta.get('max'))
    elif dim == 'differential_eta':
        return 'eta', axis_bins(eta), (pT.get('min'), pT.get('max'))
    elif dim == 'differential_y':
        # bins in rapidity (kinematic_range.y; with y.mass one assumed mass:
        # ATLAS y_pi)
        return 'y', axis_bins(y), (pT.get('min'), pT.get('max'))
    elif dim == 'integrated' or dim is None:
        # Spectra-type records carry their measurement pT bins explicitly;
        # expose them so prepare functions see the real binning.
        if exp_rec.get('observable_type') == 'particle_spectra':
            fixed = (eta.get('min'), eta.get('max'))
            bins = axis_bins(pT)
            if len(bins) > 1:
                return 'pT', bins, fixed
            if pT.get('min') is not None and pT.get('max') is not None:
                return 'pT', [(pT['min'], pT['max'])], fixed
        # dN/deta per-eta records (e.g. PHOBOS dN/deta vs eta) are curated
        # with dimensionality None but carry explicit kinematic_range.eta.bins.
        # Expose them as eta-differential so the per-eta prepare path runs.
        # (Added Aug 2026 for the JETSCAPE cross-validation; covered by
        # test_dispatcher per-eta yield cases.)
        if exp_rec.get('observable_type') == 'particle_yields':
            bins = axis_bins(eta)
            if len(bins) > 1:
                return 'eta', bins, (pT.get('min'), pT.get('max'))
            bins = axis_bins(y)
            if len(bins) > 1:
                return 'y', bins, (pT.get('min'), pT.get('max'))
        return None, [], None
    # Fallback: if kinematic_range.pT.bins exists, treat as pT-differential
    bins = axis_bins(pT)
    if bins:
        return 'pT', bins, (eta.get('min'), eta.get('max'))
    # If pT has min/max but no bins, create a single bin covering the range
    pT_min = pT.get('min')
    pT_max = pT.get('max')
    if pT_min is not None and pT_max is not None:
        return 'pT', [(pT_min, pT_max)], (eta.get('min'), eta.get('max'))
    return None, [], None


def _safe_float(v, default=np.nan):
    """Convert a value to float, returning default on failure."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def load_experimental(params):
    """Load experimental values from the raw HEPData YAML table."""
    inspire_id = params['inspire_id']
    if params.get('experimental_values') is False:       # values only in a ratio record
        return None, None
    # A record may carry its own HEPData-format table (data without a HEPData
    # record, e.g. preliminary results): `value_table`, relative to the record file.
    value_table = params.get('value_table')
    if value_table:
        base = os.path.dirname(params.get('record_file') or '') or _PROJECT_ROOT
        params = dict(params, _raw_file=value_table if os.path.isabs(value_table)
                      else os.path.join(base, value_table))

    # PHOBOS records use non-standard inspire_id strings (not numeric INSPIRE IDs)
    # and store data as plain ASCII text, not HEPData YAML — skip with clear message
    if not value_table and isinstance(inspire_id, str) and inspire_id.startswith('PHOBOS'):
        print(f"  PHOBOS experimental data not in HEPData YAML format — skipped")
        return None, None
    
    raw_file = params.get('_raw_file') or os.path.join(
        _PROJECT_ROOT,
        "hepdata_raw/records",
        f"ins{inspire_id}",
        f"HEPData-ins{inspire_id}-v{params['version']}-yaml",
        params['hepdata_file'],
    )
    try:
        raw = _load_table(raw_file)
        dv_idx = int(params.get('dv_index', 0) or 0)
        dv_avg = params.get('dv_average', None)
        if dv_avg:
            idxs = [int(i) for i in dv_avg]
            series = [raw["dependent_variables"][i] for i in idxs]
        else:
            series = [raw["dependent_variables"][dv_idx]]
        # Percent-style units (e.g. STAR tables storing v_n "in %" or as
        # "v_3*100") normalize to fractions; MeV-scale headers (e.g.
        # PHENIX "Mean pT MeV/c") normalize to GeV, so experimental
        # overlays match computed values.
        values = []
        errors = []
        keep = params.get('table_rows')
        for j, vrows in enumerate(zip(*[s["values"] for s in series])):
            if keep is not None and j not in keep:
                continue
            acc_val, acc_err = 0.0, 0.0
            row_ok = True
            for v, s in zip(vrows, series):
                header = s.get('header', {}) or {}
                units = str(header.get('units', ''))
                name = str(header.get('name', ''))
                low = f'{units} {name}'.lower()
                # '%' units, 'v_3*100', or a '(%)' in the column name (STAR
                # 1210062 'v_3(TPC) (%)'; not a bare '%', which centrality
                # labels such as 'in centrality 0-50%' carry).  A record's
                # explicit value_scale replaces this guess (it is divided out
                # below), so a pre-scaled column is never scaled twice.
                if params.get('value_scale') is None and (
                        '%' in units or '*100' in name.replace(' ', '') or re.search(r'\(\s*%\s*\)', name)):
                    scale = 0.01
                elif 'mev' in low and 'gev' not in low:
                    # (MeV/c)^2 (STAR <Delta pT_i Delta pT_j>): GeV^2
                    scale = 1e-6 if re.search(r'mev(/c)?\)?\s*\^\s*\{?2', low) else 0.001
                else:
                    scale = 1.0
                val = _safe_float(v.get("value"))
                if np.isnan(val):
                    row_ok = False
                    break
                acc_val += val * scale
                acc_err += _point_error(v.get("errors") or [], val,
                                        params.get('errors_excluded')) * scale
            if not row_ok:
                continue
            values.append(acc_val / len(series))
            errors.append(acc_err / len(series))
        if not values:
            print(f"  No valid experimental values found")
            return None, None
        # A record's own factor (e.g. 1000 for a column tabulated "x 10^3").
        vs = float(params.get('value_scale') or 1.0)
        return np.array(values) / vs, np.array(errors) / vs
    except Exception as e:
        print(f"  Could not load experimental values: {e}")
        return None, None


def _error_size(x, val):
    """One error number: absolute, or 'x%' relative to the value."""
    if isinstance(x, str) and x.strip().endswith('%'):
        return abs(val) * _safe_float(x.strip()[:-1]) / 100
    return abs(_safe_float(x))


def _point_error(errs, val, excluded=None):
    """Total error of a HEPData point: all components (statistical and every
    systematic one) in quadrature; an asymmetric component enters as the mean
    of its two sides.  ``excluded``: a record's errors_excluded, components
    that are part of another one (ALICE 'syst. uncorr.', the
    centrality-uncorrelated part of 'syst.') or that the record's quantity
    does not carry (ATLAS 'syst, TAA' of a per-event yield); '<label>#<k>'
    names the k-th component with that label."""
    total = 0.0
    excluded = set(excluded or ())
    seen = {}
    for e in errs:
        lab = e.get('label')
        seen[lab] = seen.get(lab, 0) + 1
        # 'sys#2': the second component labelled 'sys' (ALICE 1244523 tabulates
        # the total and its multiplicity-uncorrelated part both as 'sys')
        if lab in excluded or f'{lab}#{seen[lab]}' in excluded:
            continue
        if 'symerror' in e:
            size = _error_size(e['symerror'], val)
        elif 'asymerror' in e:
            a = e['asymerror'] or {}
            sides = [_error_size(a.get(k), val) for k in ('plus', 'minus')]
            sides = [x for x in sides if not np.isnan(x)]
            size = sum(sides) / len(sides) if sides else np.nan
        else:
            size = np.nan
        if not np.isnan(size):
            total += size ** 2
    return float(np.sqrt(total))


def iter_records(record_dir, value_type_filter=None):
    """Iterate over all curated records in a directory, optionally filtered.

    Yields (file_path, doc) for each matching document.
    """
    for fname in sorted(os.listdir(record_dir)):
        if not fname.endswith(".yaml"):
            continue
        fpath = os.path.join(record_dir, fname)
        with open(fpath) as f:
            docs = list(yaml_load_all(f))
        for doc in docs:
            if doc is None:
                continue
            if value_type_filter:
                vt = doc.get("value_type", "")
                if value_type_filter not in vt:
                    continue
            yield fpath, doc
