# ReCAD development workflow

This workflow is mandatory for every repository change.

## Branches and integration

- Never commit or push directly to `main`.
- Start from the current `main` and use `feat/<short-name>`, `bugfix/<short-name>`,
  `docs/<short-name>`, `refactor/<short-name>`, `test/<short-name>`, or
  `chore/<short-name>` as appropriate.
- Open a pull request, wait for all required checks, and use **Squash and merge**.
- Merge commits and rebase merges are disabled. Delete the source branch after merge.
- A scientific Issue closes only after its reviewer archive verifies and its PR is squash-merged.

## Python quality

Python code must satisfy PEP 8 as enforced by Ruff:

```powershell
ruff check src tests scripts
ruff format --check <changed-python-files>
```

The repository is adopting linting and formatting incrementally; CI checks every Python file
changed by the pull request. Run `ruff format` and `ruff check --fix` locally when needed. New or changed behavior
must have a regression test; tests that merely duplicate the implementation are insufficient.

## Test levels

| Level | Purpose | Required when | Command |
|---|---|---|---|
| L1 | Fast isolated unit and regression tests; no network, GPU, or project data | Every code change and commit | `pytest -m "not l2 and not l3 and not slow and not gpu" -q` |
| L2 | CPU integration across modules, CLI, serialization, and temporary-file interfaces | Every pull request that changes code or configuration | `pytest -m "l2 or slow" -q` |
| L3 | End-to-end experiment replay, real-data/hash validation, reviewer archive, or GPU validation | Scientific experiments, data pipelines, model/training/evaluation changes, and releases | `pytest -m "l3 and not gpu" -q`, plus the registered real-data/GPU command |

Use `@pytest.mark.l2` or `@pytest.mark.l3` at module or test level. Unmarked portable
tests are L1. A GPU-dependent L3 test carries both `l3` and `gpu`. The pull request
must record the exact non-portable L3 command, data manifest hash, outcome, and artifact path.

## Scientific experiment completion

Formal experiments also follow `docs/experiment_protocol.md`. Before merge they require:

1. a committed preregistration and frozen data/config hashes;
2. L1 and L2 tests plus the applicable L3 full experiment run;
3. a verified `docs/experiment_archive/<experiment_id>/` containing `REPORT.md`,
   `CAPTIONS.md`, figures, source tables, and `archive_manifest.json`;
4. a pull request description that states the result, allowed claim, failed gates,
   sealed-data status, and validation commands.
