# P3 one-time independent evaluation freeze

## Prepared state

The SSS and fCO2 candidates, calibration objects, product masks, checkpoints, inputs, metrics, report policy, and conditional release contract are now frozen before any new independent target label is opened. The machine lock is `configs/frozen/p3_lock_v2.4.json`; it records the final Issue 44 products, 21 checkpoint/config artifacts, calibration hashes, predictor-cube and coastal-mask hashes, and candidate source commit `f4405bead157c94a544d8534d0ee8d0c837952be`.

The lock status is **`prepared_waiting_for_eligible_independent_set`** and `label_access_authorized` is false. No independent target label was read, no access count was consumed, and no 2026 provisional label was opened.

## Frozen candidates

| Target | Candidate | Frozen product SHA256 | Evaluation status |
|---|---|---|---|
| SSS | Issue 62 CatBoost background-residual three-seed ensemble | `3d3fbc01d8c63d2c8a955f601e4c4fead6c019d2108f93c47f538f70da12e0db` | waiting for eligible labels |
| fCO2 | Issue 64 CatBoost/F4 inner-OOF convex ensemble, CatBoost weight 0.3 | `88869d61f0f79d24e6c6f72b8eabe64a3e8de61efb9c2e6a4dcfb865bb2a13cb` | waiting for eligible labels |
| TA | excluded | n/a | Issue 45 untriggered |
| DIC | excluded | n/a | Issue 45 untriggered |

The frozen predictor cube hash is `0f4e9f37ddd65d5abb0ebd92715a8521703a7778375d05c1a6055254203d0401`; the coastal mask hash is `0e99ca53398a3f4c29d5058b3307857535c78195bcbabdadf06c59b0ee4957cf`. The evaluation protocol forbids mask, threshold, calibration, candidate, or metric changes after label access.

## Metadata-only eligibility audit

Only filenames, version metadata, and the Issue 61 exposure ledger were read. The local SOCAT inventory contains v2026 and no release newer than v2026. The legacy SOCAT ledger is ineligible: all 538 locked groups entered the global manifest and 417 groups/79,135 rows entered the P2 matchup; all nine legacy external groups entered the manifest and seven groups/1,474 rows entered the matchup. These identities cannot regain blind status through re-splitting.

The eligible design remains prospective: whole cruise/platform groups from a SOCAT release newer than v2026, selected without reading targets, normalized against every Issue 61 alias, and frozen with source and candidate-set hashes before target access. The exact set does not currently exist on this machine, so the frozen protocol does not unambiguously authorize a one-time evaluation.

## Independent-evaluation contract

The evaluation will report observation, grid-month, cruise, region, geographic-block, support-distance, time, and fixed-coverage metrics. It freezes RMSE, MAE, bias, R2, cruise-equal and region-macro RMSE, 50/90 coverage and width, WIS, grade-specific skill, boundary/front diagnostics, and common-support comparison against strict v1.x. The Issue 44 grade and abstention policy is applied without refitting. A regional pass cannot become a global claim; failed target-region pairs remain diagnostic or suppressed.

## Conditional Issue 46 release

`configs/frozen/p3_release_contract_v2.4.json` prepares the 1993-2025 SSS/fCO2 release schema, trusted/diagnostic/suppressed layers, required variables, common 0.25-degree comparisons, provenance, and separate 2026 provisional policy. Issue 46 may build a trusted release only for target-region pairs that pass Issue 5. TA/DIC remain suppressed.

## Why there are no result figures

This is a pre-access freeze and eligibility audit, not an evaluation result. Plotting prior development labels again would neither create independent evidence nor answer the P3 gate. Result figures, captions directly below each figure, and source tables will be generated exactly once after an eligible independent set is frozen and opened. The absence of figures here preserves the distinction between preparation and evidence.

## Required decision

P3 cannot proceed from the current files. The next concrete choice is to obtain a SOCAT release newer than v2026 and authorize its metadata-only cruise inventory as the candidate independent source, or nominate another genuinely independent SSS/fCO2 observation source with comparable global coastal coverage. No label should be opened until that source is present and the generated zero-overlap candidate list is reviewed and hashed.
