import numpy as np
from numba import jit

from hmm_kit.constant import SMALL_CONSTANT


def decode(
    data: np.ndarray,
    covariate: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
):
    """
    Implements the Viterbi Algorithm to find the most likely sequence of hidden states
    given a sequence of observations (data, covariate),
    based on a model defined by transition probabilities, emission probabilities,
    and initial state probabilities. This function handles the model
    where transition and emission probabilities can depend on covariates.

    Args:
    ----
        data: np.ndarray
            An 1D array of observed data points, where each element corresponds to an observed symbol
            from the alphabet defined by the emission matrix.
        covariate: np.ndarray
            An 1D array of covariate indices that correspond to each data point, which are used to
            select the appropriate slice from the transition and emission matrices.
        transition_matrix: np.ndarray
            A 3D array where element (i, j, k) represents the probability of transitioning from
            state j to state k given covariate i.
        emission_matrix: np.ndarray
            A 3D array where element (i, j, k) represents the probability of emitting symbol k
            when in state j given covariate i.
        initial_probabilities: np.ndarray
            An array representing the initial probability of each state at the start of the
            sequence.

    Returns:
    -------
        tuple:
            - float: The log-likelihood of the most likely path.
            - np.array: An array representing the most likely sequence of states.
    """

    # log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # setups
    num_states = len(initial_probabilities)

    # decode algorithm
    log_likelihood, path = _decode(data, covariate, num_states, log_A, log_B, log_pi)

    return log_likelihood, path


@jit(nopython=True)
def _decode(data, covariate, num_states, log_A, log_B, log_pi):
    """
    Optimized Viterbi decoding Algorithm.
    """
    T = len(data)
    delta = np.zeros((T, num_states))
    psi = np.zeros((T, num_states), dtype=np.int64)

    delta[0] = log_pi + log_B[covariate[0]][:, data[0]]

    for t in range(1, T):
        for i in range(num_states):
            # from previous state to state i
            temp = delta[t - 1, :] + log_A[covariate[t - 1]][:, i]
            # find max, argmax
            delta[t, i] = np.max(temp) + log_B[covariate[t]][i, data[t]]
            psi[t, i] = np.argmax(temp)  # index for max delta

    log_likelihood = np.max(delta[-1])
    path = np.zeros(T, dtype=np.int64)
    path[-1] = np.argmax(delta[-1])

    # path backtracking
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return log_likelihood, path
