# Repository contribution rules

## Language

- Use English for every tracked repository file, including documentation, code comments, configuration descriptions, reports, tests, release notes, and contributor instructions.
- Use English for Git branch names, commit messages, GitHub issue titles and bodies, pull request titles and bodies, review comments, and issue or pull request comments.
- Do not add Chinese or other non-English prose to Git. Personal working notes may use another language only when they remain under the ignored `/out/` directory.
- Translate legacy non-English tracked content through dedicated reviewable documentation pull requests. Do not mix broad translation changes with scientific or model changes.

## Generated and personal artifacts

- Keep personal reports and generated working documents under `/out/`; this directory is ignored by Git.
- Keep scientific data, model checkpoints, large predictions, and generated products outside Git. Commit only reproducible code, compact configuration, registries, hashes, and reviewer archives permitted by the project workflow.

## Development workflow

- Work on a `feat/`, `bugfix/`, or `docs/` branch. Do not commit directly to `main`.
- Open a pull request and squash-merge it after required checks pass.
- Code changes must satisfy PEP 8 and the repository's L1, L2, and L3 test levels as applicable.
- Every scientific experiment must have a reviewer-ready report, source tables, archived figures, and a caption directly below each figure in the report.
