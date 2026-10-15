# P3 terminal evidence boundary: no eligible independent source

## Decision

Issues 5 and 46 close with **`terminal_no_eligible_independent_source`**. SOCAT v2026 is the latest available release, and no alternative genuinely independent global coastal SSS/fCO2 observation source is available. The P3 lock remains unopened: independent-label access was never authorized, access count remains zero, and no 2026 provisional label was read.

## Why the existing data cannot be reused

The Issue 61 exposure audit found that all 538 legacy SOCAT locked groups and all nine legacy external groups entered the P2 manifest. Of these, 417 locked groups and seven external groups contributed 79,135 and 1,474 matchup rows. Their labels influenced the development data chain and cannot regain blind status through a new split, a new filename, or a later evaluation script.

The Issue 25 SSS locked set was already opened and is permanently spent. The 2004-2005 benchmark has also been repeatedly examined. SOCAT v2026 cannot simultaneously serve as the development source and a new independent validation source.

## Product consequence

| Output | Final status | Permitted claim |
|---|---|---|
| Global coastal SSS 1993-2025 | `diagnostic_only` | development candidate with frozen A/B/C/D and uncertainty fields |
| Global coastal fCO2 1993-2025 | `diagnostic_only` | development candidate with frozen A/B/C/D and uncertainty fields |
| Trusted SSS/fCO2 release | not produced | no target-region pair passed an independent gate |
| North-American TA/DIC | suppressed | Issue 45 was untriggered |
| 2026 product | provisional only | excluded from formal validation and release claims |

The complete Issue 44 products remain scientifically useful for method development, hypothesis generation, support-aware mapping, and designing future sampling. Their A/B grades describe the demonstrated development support envelope. They do not prove independent accuracy at unsampled grid cells.

## Issue 46 closure

The release schema, provenance fields, common-grid comparison rules, and checksums were prepared in PR 73. The frozen release gate permits a trusted product only for target-region pairs that pass Issue 5. Since no eligible independent set exists, the gate cannot be evaluated and no trusted release is created. Publishing the existing files must use the phrase “development/diagnostic candidate product” and retain the complete grade, status, reason-bit, interval, and provenance fields.

## Preserved frozen assets

The candidate lock, evaluation protocol, and conditional release contract remain frozen for a future data release. The SSS and fCO2 candidate product SHA256 values are `3d3fbc01d8c63d2c8a955f601e4c4fead6c019d2108f93c47f538f70da12e0db` and `88869d61f0f79d24e6c6f72b8eabe64a3e8de61efb9c2e6a4dcfb865bb2a13cb`. Reopening requires a genuinely new source and a metadata-only zero-overlap audit before reading labels.

## Why there are no result figures

No independent evaluation was run, so there are no independent metrics or figures to plot. Reusing development figures in this archive would blur the evidence boundary. All development figures and captions remain in the Issue 44 archive; this terminal archive records only the availability decision and permitted claims.
