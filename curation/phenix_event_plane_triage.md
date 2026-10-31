# PHENIX event_plane papers: triage of the reference geometry

Produced while working through `data/flow_vn/phenix_*.yaml` paper-by-paper.
Each row was read from the paper's own full text (ar5iv render of the arXiv
id in `hepdata_raw/records/ins*/metadata.json`), not inferred from the
collaboration.

The reason a per-paper pass is necessary: PHENIX alone uses at least six
different event-plane geometries, and the reference is never the POI
acceptance and never (in these papers) the reaction-plane detector used for
some other purpose.

    | class | geometry | shareable with POI? |
    |-------|----------|---------------------|
    | A | combined two-arm BBC, 3 < |eta| < 3.9 | disjoint from the central arms |
    | B | single-arm BBC (BBC-S / Al- or Au-going), -3.9 < eta < -3.0/-3.1 | disjoint |
    | C | FVTX-S, -3.0 < eta < -1.0 | disjoint |
    | D | MPC, 3 < |eta| < 4 | disjoint |
    | E | RXN inner / outer / combined, 1.0 < |eta| < 1.5 and 1.5 < |eta| < 2.8 | disjoint |
    | F | three-subevent triples (e.g. BBCS + FVTXS + CNT) | depends which pair |

Key traps found in this pass:

* **The reaction-plane detector is often not the flow reference.** In
  INSPIRE 819672 (arXiv:0905.1070) a central-arm track event plane exists and
  is used -- but only "to estimate the resolution of BBC and ZDC-SMD event
  planes". Assuming the CNT is the reference would be wrong.
* **A paper can define several event planes and publish v2 against each.**
  INSPIRE 1254476 (arXiv:1309.4437) compares v2(pT) from MPC
  (3 < |eta| < 4), RXN_in (1.5 < |eta| < 2.8) and RXN_out
  (1.0 < |eta| < 1.5). Which one a given record belongs to is a property of
  the table, not of the paper, so it needs a per-record decision (the schema
  supports this via `flow_reference.segments`).
* **v_1 is not the BBC.** INSPIRE 1394897 uses the SMD *spectator* plane for
  Psi_1 while v_2 / v_3 use BBCS + BBCN.
* **Some of these papers are not event-plane measurements at all.**
  INSPIRE 585347 (arXiv:nucl-ex/0204005) is "Flow measurements via
  two-particle azimuthal correlations" -- a two-particle cumulant method.
  Its records carry `method: event_plane`, which is a misclassification, not
  a missing reference. Same suspicion for INSPIRE 1222874 (arXiv:1303.1794,
  quadrupole anisotropy in two-dihadron azimuthal correlations) and the
  jet/reaction-plane papers INSPIRE 816486 and 656142, where v2 is defined
  *with respect to* the reaction plane rather than measured by it.

## Status

Curated (10 papers, 210 records) -- see `curation/flow_reference.yaml`.

Remaining PHENIX papers, with the decision each one needs:

| INSPIRE | arXiv | rec | class | decision needed |
|---|---|---:|---|---|
| 1672133 | 1805.02973 | 16 | B | DONE - south BBC -3.9<eta<-3.1 |
| 2026169 | 2107.06634 | 12 | F | label says "BBCS-FVTXS-CNT"; need which two of the three form the correlator |
| 1254476 | 1309.4437 | 10 | D/E | three event planes (MPC, RXN_in, RXN_out); needs `segments` per record |
| 1684475 | 1807.11928 | 10 | B | DONE - single-arm BBC -3.9<eta<-3.1 |
| 900703 | 1105.3928 | 9 | D/E | BBC, RXN and MPC all defined; which one for the published v2 |
| 585347 | nucl-ex/0204005 | 7 | -- | NOT an event-plane method: two-particle cumulant. Reclassify. |
| 858845 | 1006.3740 | 7 | E | label: "reaction plane determined with MPC and RXN in combined" -- need the window |
| 731134 | nucl-ex/0611007 | 5 | ? | no reference sentence located; high-pT pi0 |
| 850211 | 1003.5586 | 5 | D/E | BBC + MPC + RXN(i,o,io) all defined; which for the published v2 |
| 2830400 | 2409.12715 | 5 | C | forward-rapidity charged v2; three-subevent -- need the windows |
| 1296108 | 1405.3301 | 4 | -- | "event plane is constructed in two different windows: the South or North side of the RXNP" |
| 2830407 | 2409.12756 | 4 | F | J/psi v2: Psi2_A FVTX, Psi2_B opposite BBC, Psi2_C CNT -- need which |
| 1632759 | 1710.09736 | 4 | B/F | BBC-S or FVTX-S per system (p+Au vs 3He+Au); three-subevent BBC-S/FVTX-S/CNT |
| 816486 | 0903.4886 | 2 | -- | high-pT pi0 *with respect to* the reaction plane: reaction-plane-correlated, not an EP measurement |
| 854475 | 1005.1627 | 2 | ? | no reference sentence located; open-heavy-flavour v2 |
| 1384274 | 1507.06273 | 1 | B/C | BBC-S -3.9<eta<-3.0 and FVTX-S -3.0<eta<-1.0 both defined; which for v2 |
| 619061 | nucl-ex/0305013 | 1 | ? | no reference sentence located (older paper, short text) |
| 1222874 | 1303.1794 | 1 | -- | NOT an event-plane method: two-dihadron correlations |
| 1658594 | 1803.01749 | 1 | E | "event planes is determined by the BBC and the RXN" -- which |
| 656142 | nucl-ex/0408007 | 1 | -- | jet structure; v2 likely reaction-plane-correlated |
| 676189 | nucl-ex/0502009 | 1 | ? | single-electron anisotropy; no reference sentence located |

Updated after the second pass:

* **INSPIRE 858845** resolved to "MPC and RXN_in combined" from the Figure 1
  caption -- "(a)-(f): pi0 v2 using reaction plane determined with MPC and
  RXN_in combined ... (g)-(i): ratios of v2 measured separately using MPC and
  RXN_in to combined result".  NOT stamped: the two papers disagree on which
  ring is which -- INSPIRE 1254476 describes "RXN_in (1.5 < |eta| < 2.8)" and
  "RXN_out (1 < |eta| < 1.5)", i.e. the INNER ring is the OUTER rapidity
  band.  Until that is pinned down from the RXN geometry, a summed
  MPC + RXN reference would be guesswork.
* **INSPIRE 2026169** table label reads "BBCS-FVTXS-CNT", so its v2 uses a
  three-subevent construction; which two of the three form the correlator
  still needs reading.
* **INSPIRE 1384274** defines BBC-S (-3.9<eta<-3.0) AND FVTX-S
  (-3.0<eta<-1.0) and says v2/v3 are "measured with respect to their
  corresponding event planes" -- which does not say which detector supplies
  which.  Unresolved.
* **INSPIRE 1672133**: psi_3 is explicitly "measured using the south BBC";
  v_2 is taken from the same section with the same gap statement, and the
  record says so rather than asserting it.

`scripts/apply_flow_reference.py` prints this TODO list on every run, so it
stays current as entries land.
