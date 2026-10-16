# Issue 81 absolute-error flag contract

This preregistration archive freezes the model-agnostic error-tier schema and decision order. Run `python scripts/build_absolute_error_flag_contract_archive.py`, then verify the archive with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_absolute_error_flag_contract_v2.5`.

Physical A/B/C/D thresholds are intentionally unset until their intended-use derivation is reviewed. The production validator fails when `--require-thresholds` is used before that freeze.
