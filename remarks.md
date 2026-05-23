# hmm-kit codebase remarks

## Completion

The project is a compact Python package with a clear target: discrete Hidden Markov Model estimation and decoding, including a covariate-dependent variant. The core algorithms are present:

- forward/backward likelihood computation in log space
- gamma/xi sufficient statistics
- Baum-Welch training loop
- covariate-dependent forward/backward and Baum-Welch
- covariate-aware Viterbi decoding
- sampling and likelihood utilities

Current completion is good for a practice/research prototype, but not yet strong enough as a reliable package API. The implementation assumes valid inputs, contiguous integer-encoded observations/covariates, non-empty sequences, normalized probability matrices, and matching shapes. These assumptions are not enforced, so invalid data will usually fail with low-level NumPy/Numba errors or silently produce wrong updates.

## Organization

The codebase is small and easy to scan, but boundaries are currently loose:

- `hmm_kit/core/base.py` mixes public API, EM orchestration, sufficient-statistic accumulation, and Numba kernels.
- `hmm_kit/core/covariates.py` duplicates much of `base.py`, which makes the two variants easy to drift apart.
- `hmm_kit/utils.py` contains unrelated concerns: likelihood calculation, normalization, label switching, sequence sampling, and base-statistics initialization.
- `__init__.py` files are empty, so there is no intentional package-level API.
- Public return contracts are tuple-based and undocumented in a way users can rely on programmatically.

The most valuable organizational refactor is to introduce shared validation and normalization helpers, then make base/covariate training use consistent internal contracts. A heavier class-based model abstraction can wait until the behavior is more thoroughly tested.

## Function Design

Strong points:

- The probability recursions use log space, which is the right direction for numerical stability.
- Numba kernels are isolated in lower-level helper functions.
- The Baum-Welch loop is readable and follows the expected E-step/M-step structure.
- Test fixtures compare the base covariate case to the non-covariate algorithms, which is a useful invariant.

Design issues:

- `normalize()` divides blindly; all-zero rows produce NaN and no error. This can happen after masked updates or degenerate input.
- Function signatures accept shape-sensitive arrays but do not validate dimensions, lengths, integer labels, or probability rows.
- Random initialization and sampling use global `np.random`, making reproducible experiments awkward.
- `pretrains` is accepted without shape checks, so a wrong pretrained tuple can fail late or produce invalid results.
- `verbose=False` disables `tqdm`, but convergence still prints directly. Training should not print when verbosity is off.
- Covariate support assumes covariate values are `0..n_covariates-1`. If values are sparse, e.g. `[1, 3]`, matrix indexing and statistics allocation will be wrong.
- `calc_xi_with_covariates()` accepts `n_covariates` but never uses it after calculating a default.
- Type hints such as `tuple[np.ndarray]` are too vague for three-array parameter tuples.

## Functional Correctness

The existing tests pass in the project virtual environment:

```text
.venv/bin/python -m pytest -q
6 passed
```

Important correctness concerns remain:

- In `tests/test_algorithms.py`, `test_xi_statistics_with_covariates()` accidentally obtains `backward_probabilities` by calling `base_hmm.forward()` instead of `base_hmm.backward()`. This weakens the xi test and can mask errors.
- In `_do_estep_with_covariates()`, `stats["startprob"] += gamma[0]` is inside the loop over covariate values. This adds each sequence's start probability once per covariate instead of once per sequence. Because the vector is normalized later, the normalized result may look plausible when every sequence sees the same covariate count, but the sufficient statistics are wrong and fragile.
- `normalize()` is used for 1D, 2D, and 3D arrays, but there is no guard for zero denominators. EM can produce invalid probabilities in edge cases.
- No tests cover Baum-Welch convergence, pretrain shape handling, empty sequences, single-observation sequences, malformed probabilities, sparse covariate labels, zero-count covariate slices, or reproducibility.
- `sample_single_sequence()` advances the hidden state after the last emitted observation. This does not break returned values, but it does unnecessary work and can confuse readers.
- `switch_label()` uses `np.lexsort()` over potentially 3D emission matrices and then takes `res[0]`; this assumes consistent state ordering across covariates but does not verify it.

## Recommended Refactor Branch

Create a focused branch for stabilizing the current implementation before adding more features:

```text
refactor-hmm-core-validation
```

Suggested scope:

- Add internal input validation for sequence length, array dimensions, probability rows, and label ranges.
- Replace blind normalization with a safe helper that raises clear errors for zero-sum rows unless explicitly configured otherwise.
- Fix the covariate E-step start-probability accumulation bug.
- Fix the xi test so it compares against true backward probabilities.
- Centralize parameter tuple handling and shape checks.
- Keep public function names stable for now.

## Suggested Feature Branches

- `feature/random-state-control`: add optional `random_state` or `rng` support to training initialization and sampling.
- `feature/base-viterbi-api`: expose a non-covariate Viterbi wrapper so users do not need to pass a fake covariate dimension.
- `feature/model-result-types`: introduce lightweight dataclasses for training results and HMM parameters while preserving tuple compatibility during transition.
- `feature/covariate-label-encoding`: support arbitrary covariate labels by encoding them internally and returning the mapping.
- `feature/baum-welch-coverage`: add property-style or reference-package tests for EM monotonicity, shape invariants, and degenerate inputs.
