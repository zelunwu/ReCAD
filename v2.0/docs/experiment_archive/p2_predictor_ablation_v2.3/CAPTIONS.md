# Figure captions

## fig01_sss_rmse_matrix.png

Figure 1. SSS held-out RMSE for every feature variant and validation scheme; cell colors are normalized to the mandatory-environment A2 baseline while annotations retain absolute PSU values.

## fig02_fco2_rmse_matrix.png

Figure 2. fCO2 held-out RMSE for every feature variant and validation scheme; pooled gains from background SSS are small compared with the variation among validation designs.

## fig03_optional_effects.png

Figure 3. Relative RMSE effects of background SSS, ADT, and training-only missing-field dropout across all five decision schemes; positive bars denote lower candidate error.

## fig04_common_support.png

Figure 4. Optional-field effects on all observations and the identical strict common-support subset, preventing coverage reduction from masquerading as predictor skill.

## fig05_bootstrap_intervals.png

Figure 5. Cruise-level paired bootstrap mean squared-error contrasts with 95% intervals; an optional group passes this gate only when the complete interval is below zero in cruise and spatial validation.

## fig06_lme_distributions.png

Figure 6. Distribution of per-LME errors under whole-LME validation; background-SSS residual correction sharply reduces SSS transfer error but substantial regional heterogeneity remains.

## fig07_worst_lme.png

Figure 7. Worst per-LME RMSE under whole-LME validation for the principal baseline and retained variants, emphasizing the tail risk hidden by pooled global metrics.

## fig08_time_blocks.png

Figure 8. Cruise-OOF RMSE in four frozen time blocks for the retained SSS input and frozen fCO2 input, documenting temporal rather than only pooled behavior.

## fig09_forward_periods.png

Figure 9. Strict forward-time RMSE in 2019-2021 and 2022-2025 for the mandatory baseline and principal optional candidates; later years are harder for both targets.

## fig10_annual_bias.png

Figure 10. Annual cruise-OOF bias for the retained target-specific inputs; year-to-year changes mix model drift with changing observational composition and are not interpreted as source discontinuities.

## fig11_missing_strata.png

Figure 11. Cruise-OOF RMSE by background-SSS and ADT availability strata; unavailable SSS uses the frozen same-fold A2 fallback and ADT absence never removes a prediction.

## fig12_retained_decision.png

Figure 12. Frozen target-specific gate outcome: background-SSS residual correction is retained only for SSS, while ADT and missing-field dropout are rejected for both targets.
