import numpy as np


def validate_base_inputs(
    data: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> None:
    _validate_integer_sequence("data", data)
    _validate_initial_probabilities(initial_probabilities)

    n_hiddens = len(initial_probabilities)
    n_outcomes = emission_matrix.shape[-1] if emission_matrix.ndim == 2 else None

    _validate_probability_array(
        "transition_matrix", transition_matrix, expected_shape=(n_hiddens, n_hiddens)
    )
    _validate_probability_array(
        "emission_matrix", emission_matrix, expected_shape=(n_hiddens, n_outcomes)
    )
    _validate_label_range("data", data, n_outcomes)


def validate_covariate_inputs(
    data: np.ndarray,
    covariates: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> None:
    _validate_integer_sequence("data", data)
    _validate_integer_sequence("covariates", covariates)
    if len(data) != len(covariates):
        raise ValueError(
            f"data and covariates must have the same length, got {len(data)} and {len(covariates)}"
        )

    _validate_initial_probabilities(initial_probabilities)
    n_hiddens = len(initial_probabilities)

    if transition_matrix.ndim != 3:
        raise ValueError(
            f"transition_matrix must be 3-dimensional, got shape {transition_matrix.shape}"
        )
    if emission_matrix.ndim != 3:
        raise ValueError(
            f"emission_matrix must be 3-dimensional, got shape {emission_matrix.shape}"
        )
    if transition_matrix.shape[0] != emission_matrix.shape[0]:
        raise ValueError(
            "transition_matrix and emission_matrix must have the same covariate dimension"
        )

    n_covariates = transition_matrix.shape[0]
    n_outcomes = emission_matrix.shape[-1]

    _validate_probability_array(
        "transition_matrix",
        transition_matrix,
        expected_shape=(n_covariates, n_hiddens, n_hiddens),
    )
    _validate_probability_array(
        "emission_matrix",
        emission_matrix,
        expected_shape=(n_covariates, n_hiddens, n_outcomes),
    )
    _validate_label_range("data", data, n_outcomes)
    _validate_label_range("covariates", covariates, n_covariates)


def validate_training_sequences(
    data_list: list[np.ndarray], n_outcomes: int, name: str = "data_list"
) -> None:
    if not data_list:
        raise ValueError(f"{name} must contain at least one sequence")
    for idx, data in enumerate(data_list):
        _validate_integer_sequence(f"{name}[{idx}]", data)
        _validate_label_range(f"{name}[{idx}]", data, n_outcomes)


def validate_covariate_training_sequences(
    data_list: list[np.ndarray],
    covariates_list: list[np.ndarray],
    n_outcomes: int,
    n_covariates: int,
) -> None:
    validate_training_sequences(data_list, n_outcomes)
    if len(data_list) != len(covariates_list):
        raise ValueError(
            "data_list and covariates_list must contain the same number of sequences"
        )
    for idx, (data, covariates) in enumerate(zip(data_list, covariates_list)):
        _validate_integer_sequence(f"covariates_list[{idx}]", covariates)
        if len(data) != len(covariates):
            raise ValueError(
                f"data_list[{idx}] and covariates_list[{idx}] must have the same length"
            )
        _validate_label_range(f"covariates_list[{idx}]", covariates, n_covariates)


def infer_n_covariates(covariates_list: list[np.ndarray]) -> int:
    if not covariates_list:
        raise ValueError("covariates_list must contain at least one sequence")
    for idx, covariates in enumerate(covariates_list):
        _validate_integer_sequence(f"covariates_list[{idx}]", covariates)

    observed = np.unique(np.concatenate(covariates_list).ravel())
    expected = np.arange(observed[-1] + 1)
    if not np.array_equal(observed, expected):
        raise ValueError("covariate labels must be contiguous and start at 0")
    return len(observed)


def validate_base_pretrains(
    pretrains: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_hiddens: int,
    n_outcomes: int,
) -> None:
    transition_matrix, emission_matrix, initial_probabilities = pretrains
    _validate_initial_probabilities(initial_probabilities, expected_shape=(n_hiddens,))
    _validate_probability_array(
        "transition_matrix", transition_matrix, expected_shape=(n_hiddens, n_hiddens)
    )
    _validate_probability_array(
        "emission_matrix", emission_matrix, expected_shape=(n_hiddens, n_outcomes)
    )


def validate_covariate_pretrains(
    pretrains: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_covariates: int,
    n_hiddens: int,
    n_outcomes: int,
) -> None:
    transition_matrix, emission_matrix, initial_probabilities = pretrains
    _validate_initial_probabilities(initial_probabilities, expected_shape=(n_hiddens,))
    _validate_probability_array(
        "transition_matrix",
        transition_matrix,
        expected_shape=(n_covariates, n_hiddens, n_hiddens),
    )
    _validate_probability_array(
        "emission_matrix",
        emission_matrix,
        expected_shape=(n_covariates, n_hiddens, n_outcomes),
    )


def _validate_initial_probabilities(
    initial_probabilities: np.ndarray, expected_shape: tuple[int, ...] | None = None
) -> None:
    if initial_probabilities.ndim != 1:
        raise ValueError(
            f"initial_probabilities must be 1-dimensional, got shape {initial_probabilities.shape}"
        )
    expected_shape = expected_shape or initial_probabilities.shape
    _validate_probability_array(
        "initial_probabilities",
        initial_probabilities,
        expected_shape=expected_shape,
        axis=0,
    )


def _validate_integer_sequence(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer labels")
    if np.any(values < 0):
        raise ValueError(f"{name} must not contain negative labels")


def _validate_label_range(name: str, values: np.ndarray, upper_bound: int) -> None:
    if upper_bound <= 0:
        raise ValueError(f"{name} upper bound must be positive")
    if np.any(values >= upper_bound):
        raise ValueError(
            f"{name} contains labels outside the valid range [0, {upper_bound})"
        )


def _validate_probability_array(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int | None, ...],
    axis: int = -1,
) -> None:
    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0):
        raise ValueError(f"{name} must not contain negative probabilities")
    if not np.allclose(values.sum(axis=axis), 1):
        raise ValueError(f"{name} rows must sum to 1")
