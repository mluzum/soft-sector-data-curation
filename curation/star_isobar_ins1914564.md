# STAR isobar paper (ins1914564): curation coverage

STAR, *Search for the chiral magnetic effect with isobar collisions at
√s_NN = 200 GeV*, PRC 105, 014901 (2022), arXiv:2109.00131. Its HEPData
record has 225 tables from five blind-analysis groups, many of them the
same data plotted in several figures. This note lists what is curated, what
is deliberately not, and why. It was checked by comparing every uncurated
table numerically with the curated ones.

## Bulk observables

| quantity | tables | curated as |
|---|---|---|
| v₂{2}, v₂{2}(\|Δη\|>1), v₂{4}, v₂{EP}, v₂{EP,TPC–EPD}, v₂{SP}, v₂{ZDC}; v₃{2}, v₃{2}(\|Δη\|>1), v₃{EP,EPD}, v₃{SP}; Ru+Ru and Zr+Zr | Fig. 4, Fig. 10 | `flow_vn` |
| their Ru/Zr ratios | Fig. 4, Fig. 10 | `observable_ratios`, linked to the per-system records |
| ⟨N_trk^offline⟩, Ru+Ru and Zr+Zr | Fig. 3 (top) | `particle_yields` with `efficiency_corrected: false` |
| ⟨N_trk^offline⟩ ratio | Fig. 3 (bottom) | `observable_ratios` with `efficiency_cancels: true` |
| P(N_trk^offline), Ru+Ru and Zr+Zr, and their ratio | Fig. 2 (data) | `multiplicity_distribution` (`efficiency_corrected: false`) and `observable_ratios` |

In this repository, as computable records:
- the flow coefficients except v₂{ZDC}, and their ratios;
- the ⟨N_trk⟩ ratio, which is computed as a ratio of corrected multiplicities because STAR applies no efficiency correction precisely because it cancels in such ratios;
- the per-system ⟨N_trk^offline⟩ records it links to.

Curated but kept outside this repository until they can be computed:
- **v₂{ZDC}** and its ratio, measured relative to the spectator plane (`flow_reference.plane: spectator`);
- **P(N_trk^offline)** and its ratio, not corrected for the 85–90% tracking efficiency (Secs. II–III), so they need a detector-response model.

Not curated:
- **Identical copies** of curated tables: Fig. 6 (v₂{EP} and its ratio), the per-system and ratio panels of Fig. 10 that repeat Fig. 4 or each other, and Fig. 23's v₂{4} and v₂{ZDC} with their ratios. All agree with the curated tables exactly.
- **Fig. 23 v₂{2}** (Group 4, sub-events with a 0.2 η gap) and its ratio: an alternative extraction of the curated Fig. 4 sub-event v₂. It agrees within 1.4–1.9%, and its ratio within 0.5%. Curating both would count one measurement twice in a calibration.
- **Fig. 2 Glauber curves** (Case-2, Case-3): model output.
- **Fig. 24b/e**, v₂{2} vs q₂ in 20–50% (event-shape selection): would need an event-shape-engineering observable type.

## CME observables

The γ₁₁₂ and Δγ correlators, Δδ, κ, r(m_inv), the R_Ψ₂ widths and their ratios are curated (`charge_dependent_correlations`, `observable_ratios`), outside this repository, since no computation exists for them. The curation is partial: Figs. 7–9, 11–14, 18–19, 22 and 26–27 are uncurated, and the method fields of the curated tables were not individually verified. A hydrodynamic model without charge-dependent initial conditions and local charge conservation gives only their flow-driven background, so a dedicated CME curation pass is needed before they are used.
