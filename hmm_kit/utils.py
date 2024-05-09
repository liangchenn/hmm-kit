import numpy as np

from hmm_kit.constant import SMALL_CONSTANT


def compute_log_likelihood(
    data: np.ndarray,
    covariate: np.ndarray,
    state: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> float:
    """
    Compute log likelihood given data, hidden states, and parameters.

    Args:
    ----
        data: np.array
            An 1D array of observed data points.
        covariate: np.array
            An 1D array of covariate indices that may affect the transition and emission probabilities.
        state: np.array
            An 1D array of hidden state indices corresponding to each data point.
        transition_matrix: np.ndarray
            A 3D array where element (i, j, k) represents the probability of transitioning from state j to state k given covariate i.
        emission_matrix: np.ndarray
            A 3D array where element (i, j, k) represents the probability of observing data point k from state j given covariate i.
        initial_probabilities: np.ndarray
            An array of probabilities for each state being the initial state.

    Returns:
    ------
        float:
            The computed log likelihood of the sequence under the given model parameters.
    """

    # log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # calculate log-likelihood
    log_likelihood = 0.0

    # inital point
    log_likelihood += log_pi[state[0]] + log_B[covariate[0]][state[0], data[0]]

    for t in range(1, len(data)):
        _llk = (
            log_A[covariate[t - 1]][state[t - 1], state[t]]
            + log_B[covariate[t]][state[t], data[t]]
        )
        log_likelihood += _llk

    return log_likelihood


def normalize(arr: list | np.ndarray, axis: int = -1):
    """
    Normalize array to make the axis sum to 1.
    Default to normalize along last axis.

    Parameters:
    -----------
    - arr: array-like numeric data
    - axis: dimension for normalization
    """
    return arr / arr.sum(axis=axis, keepdims=1)


def switch_label(
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Obtain consistent state with custom sorting algorithm.
    Default to use emission matrix values for sorting.

    Parameters:
    -----------
    - transition_matri (ndarray)
    - emission_matrix (ndarray)
    - initial_probabilities (ndarry)

    Return:
    -------
    - reordered_indexes (array):
        Reordered index by sorting algo. e.g. [0, 1, 2] to [2, 0, 1].
    - reordered_parameters (tuple):
        Reordered parameters with reordered indexes.
        Results will be (Transition, Emission, InitialProbs).
    """
    # TODO: consider how to accommodate other sorting algorithm
    # TODO: check the assumptions is true for the case

    n_outcomes = emission_matrix.shape[-1]  # emission could be 2- or 3-dimensional

    # sorting algo: sort by (col1, col2, col3, ..., coln)
    # np.lexsort() to sort the columns in emission matrix in order
    res = np.lexsort([emission_matrix[..., i] for i in range(n_outcomes)])

    # handle different dimensional cases
    # Assumption:
    # relation between emission probs and states is consistent under different covariates
    reordered_idx = res if len(res.shape) == 1 else res[0]

    # reorder the parameters with 2 or 3-dimensional cases
    reordered_transition = np.take(
        np.take(transition_matrix, reordered_idx, axis=-2), reordered_idx, axis=-1
    )
    reordered_emission = np.take(emission_matrix, reordered_idx, axis=-2)
    reordered_initial_prob = np.take(initial_probabilities, reordered_idx, axis=0)

    return reordered_idx, (
        reordered_transition,
        reordered_emission,
        reordered_initial_prob,
    )


def sample_single_sequence(
    transtion_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
    n_covariates: int,
    n_periods: int,
) -> tuple[np.ndarray]:
    """
    Generate a single sequence of outcomes, hidden states, and covariates.

    Parameters:
        transtion_matrix (np.ndarray): Transition matrix representing the probabilities of transitioning between hidden states.
        emission_matrix (np.ndarray): Emission matrix representing the probabilities of emitting outcomes given hidden states.
        initial_probabilities (np.ndarray): Initial probabilities representing the probabilities of starting in each hidden state.
        n_covariates (int): Number of possible covariate values.
        n_periods (int): Number of periods in the sequence.

    Returns:
        tuple[np.ndarray]: A tuple containing the generated outcome sequence, hidden state sequence, and covariate sequence.

    """
    # generate covariate sequence
    covariate_sequence = np.random.choice(n_covariates, size=n_periods)

    # generate outcome, and hidden state sequence
    outcome_sequence, hidden_sequence = _sample_single_sequence_from_parameters(
        transtion_matrix,
        emission_matrix,
        initial_probabilities,
        covariate_sequence,
        n_periods,
    )

    return outcome_sequence, hidden_sequence, covariate_sequence


def _sample_single_sequence_from_parameters(
    transtion_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
    covariate_sequence: np.ndarray,
    n_periods: int,
) -> tuple[np.ndarray]:
    """
    Samples a single sequence of hidden states and corresponding observed outcomes from the given HMM parameters.

    Parameters:
        transtion_matrix (np.ndarray): The transition matrix of the HMM.
        emission_matrix (np.ndarray): The emission matrix of the HMM.
        initial_probabilities (np.ndarray): The initial probabilities of the hidden states in the HMM.
        covariate_sequence (np.ndarray): The sequence of covariates associated with each time period.
        n_periods (int): The number of time periods in the sequence.

    Returns:
        tuple[np.ndarray]: A tuple containing the sequence of observed outcomes and the sequence of hidden states.

    """

    if len(covariate_sequence) != n_periods:
        raise ValueError(
            f"Length of covariate sequence should be equal to {n_periods=}, but get {len(covariate_sequence)}"
        )

    n_hiddens = len(initial_probabilities)
    n_outcomes = emission_matrix.shape[-1]

    hiddens = np.zeros(n_periods, dtype=int)
    outcomes = np.zeros(n_periods, dtype=int)

    curr_hidden = np.random.choice(n_hiddens, p=initial_probabilities)

    for t in range(n_periods):
        hiddens[t] = curr_hidden

        outcome = np.random.choice(
            n_outcomes, p=emission_matrix[covariate_sequence[t]][curr_hidden].ravel()
        )
        outcomes[t] = outcome

        curr_hidden = np.random.choice(
            n_hiddens, p=transtion_matrix[covariate_sequence[t]][curr_hidden].ravel()
        )

    return outcomes, hiddens


def _initialize_statistics(n_hiddens, n_outcomes):
    stats = {
        "startprob": np.zeros((n_hiddens,)),
        "transmat": np.zeros((n_hiddens, n_hiddens)),
        "emissmat": np.zeros((n_hiddens, n_outcomes)),
    }
    return stats
