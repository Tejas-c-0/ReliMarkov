# Phase 4 Implementation Report

## Status

- [IMPLEMENTED] Phase 4 retains the specified scoring helpers, D0/D1 treatments, candidate policy, deterministic tie-breaks, populations, author metrics, and user-cluster bootstrap.
- [IMPLEMENTED] The runner now accepts `--data-dir` and `--out-dir`, saves a run summary/console log with configuration, and records the maximum loaded event timestamp, elapsed seconds, and peak working set.
- [IMPLEMENTED] `phase4_validity_check.py` checks the emitted run outputs, Phase 3 hashes, data cutoff, candidate/query assertions, metric arithmetic, output-directory novelty, and repeated 1,000-query ranking hashes.
- [IMPLEMENTED] `phase4_extra_diagnostics.py` computes the requested descriptive diagnostics from training and validation only.
- [MEASURED] `python -m unittest -v test_phase3_validation.py test_phase4_validation.py test_phase4_validation_additional.py` passed: 24 tests.
- [MEASURED] `python -m py_compile kr_experiments_phase4.py phase4_validity_check.py phase4_extra_diagnostics.py test_phase4_validation_additional.py` completed successfully.
- [OPEN] The full experiment was not run in this session after the user asked to leave the long execution command to them. There are no completed Phase 4 results yet.

## Files changed or added

- [IMPLEMENTED] `kr_experiments_phase4.py` was backed up before editing to `backups/kr_experiments_phase4.20261104_182737.py`; changes are limited to timestamp guarding, configurable paths, and persisted run metadata/logging.
- [IMPLEMENTED] `test_phase4_validation_additional.py` adds coverage for deterministic tie-breaking, empty source candidates, unknown sources, out-of-catalogue targets, rank displacement, synthetic repeatability, event cutoff rejection, and bootstrap resampling reproducibility.
- [IMPLEMENTED] `phase4_validity_check.py` and `phase4_extra_diagnostics.py` are new command-line tools. Both accept `--data-dir` and `--out-dir`.
- [IMPLEMENTED] `outputs/phase4_validation/phase3_hashes_before.json` stores the pre-run hashes; `outputs/phase4_validation/phase4_output_inventory_before.json` stores the pre-run output inventory.
- [IMPLEMENTED] `outputs/phase4_validation/AGENT_LOG.md` records stages and test iterations.
- [OPEN] The three Phase 4 reports contain the audited implementation status and preserve unanswered measured-result questions until the user-run outputs are available.

## Run instructions

- [IMPLEMENTED] Run Phase 4 from the project directory with `python kr_experiments_phase4.py`; the default data and output locations are unchanged.
- [IMPLEMENTED] After a successful run, pass its timestamped run directory to the validity checker, for example: `python phase4_validity_check.py --data-dir data --out-dir outputs/phase4_validation/<run_id>`.
- [IMPLEMENTED] Descriptive diagnostics run separately per gap, for example: `python phase4_extra_diagnostics.py --data-dir data --out-dir outputs/phase4_validation/<run_id> --gap 30`.
- [OPEN] The validity checker and descriptive diagnostics have been syntax-checked but not run against a completed Phase 4 output directory.
