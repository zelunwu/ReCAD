# Issue 45 trigger audit: optional North-American TA/DIC rescue

## Decision

Issue 45 closes as **`untriggered_non_blocking`**. Its frozen entry condition requires qualified North-American SSS and fCO2 deployment inputs. That condition is not met, so no TA model was trained, no DIC was derived, no P2 training budget was consumed, and no locked, external-independent, or 2026 labels were opened.

## Evidence boundary

| Required evidence | Current evidence | Entry status |
|---|---|---|
| Qualified North-American SSS | Issue 44 A/B grades are a development support contract; global shifted 90% coverage reaches a minimum of 0.788 and fails the 0.85 gate | fail |
| Qualified North-American fCO2 | Issue 44 has the same failed shifted-interval qualification and does not promote any region to independent product status | fail |
| Deployable TA prerequisite | MAB TA remains `diagnostic_only`; SAB TA remains `fail` | unavailable |
| Deployable DIC chain | MAB derived DIC remains `diagnostic_only`, with every 2025 grid-month suppressed in the prior experiment | unavailable |

The complete-domain Issue 44 products define where a future independent audit should test the model. They do not convert development support grades into qualified upstream inputs. Treating A/B as sufficient here would weaken the frozen entry condition after observing the results.

## Frozen action matrix

| Action | Performed | Reason |
|---|---:|---|
| Train the bounded TA matrix | no | upstream entry condition failed |
| Derive DIC with PyCO2SYS | no | no qualified TA/fCO2 chain exists |
| Open prior locked TA/DIC labels | no | unnecessary and prohibited before P3 preparation |
| Open external-independent labels | no | one-time P3 access remains sealed |
| Use 2026 provisional labels | no | prohibited by the roadmap |
| Consume the 10% optional P2 budget | no | untriggered branch consumes zero training budget |

## Why there are no result figures

This archive records a preregistered trigger decision rather than a model experiment. Generating eight plots from unchanged prior results would duplicate evidence and could imply that Issue 45 trained or evaluated a new model. The reviewer-relevant evidence is the machine-readable action matrix and the hashed upstream reports below. All figures supporting the upstream scientific results remain in those immutable archives with captions directly below each figure.

## Reopening rule

The optional rescue may be reopened only under a new frozen roadmap that identifies North-American SSS and fCO2 inputs which have passed the applicable independent qualification gates. Reopening must not reinterpret Issue 44 development A/B grades as independent validation.

## Provenance

The decision and SHA256-pinned evidence are stored in `configs/frozen/p2_na_carbonate_trigger_audit_v2.4.json`. The four inputs are the Issue 44 reliability report, the SAB/MAB TA report, the MAB DIC report, and the frozen product-scope contract. No external artifact was created.
