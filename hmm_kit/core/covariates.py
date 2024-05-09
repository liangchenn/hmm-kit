import numpy as np
from numba import jit
from tqdm import tqdm

from hmm_kit.core.base import _do_mstep, calc_gamma
from hmm_kit.constant import SMALL_CONSTANT
from hmm_kit.utils import normalize


def forward_with_covariates(
    data: np.ndarray,
    covariates: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Performs the log forward algorithm for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        covariates (np.ndarray): 1D Array of observed covariates.
        initial_probabilities (np.ndarray): Array of initial state probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.

    Returns:
        np.ndarray: Matrix of forward probabilities.
    """
    # Get the number of states and the length of the data
    num_states = len(initial_probabilities)

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # calculate the log forward probabilities
    forward_probabilities = _get_forward_probabilities(
        data, covariates, num_states, log_A, log_B, log_pi
    )

    # Obtain the log likelihood
    log_likelihood = np.logaddexp.reduce(forward_probabilities[-1])

    return log_likelihood, forward_probabilities


@jit(nopython=True)
def _get_forward_probabilities(
    data: np.ndarray,
    covariates: np.ndarray,
    num_states: int,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_pi: np.ndarray,
):
    T = len(data)

    forward_probabilities = np.zeros((T, num_states))
    forward_probabilities[0] = log_pi + log_B[covariates[0]][:, data[0]]

    for t in range(1, T):
        for i in range(num_states):
            log_sum = forward_probabilities[t - 1, 0] + log_A[covariates[t - 1]][0, i]
            for j in range(1, num_states):
                log_sum = np.logaddexp(
                    log_sum,
                    forward_probabilities[t - 1, j] + log_A[covariates[t - 1]][j, i],
                )
            forward_probabilities[t, i] = log_B[covariates[t]][i, data[t]] + log_sum

    return forward_probabilities


def backward_with_covariates(
    data: np.ndarray,
    covariates: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Performs the log backward algorithm for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        covariates (np.ndarray): 1D Array of observed covariates.
        initial_probabilities (np.ndarray): Array of initial state probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.
    Returns:
        np.ndarray: Matrix of backward probabilities.
    """
    # Get the number of states and the length of the data
    num_states = len(initial_probabilities)

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # calculate the log backward probabilities
    backward_probabilities = _get_backward_probabilities(
        data, covariates, num_states, log_A, log_B
    )

    # Obtain the log likelihood
    log_likelihood = np.logaddexp.reduce(
        backward_probabilities[0] + log_B[covariates[0]][:, data[0]] + log_pi
    )

    return log_likelihood, backward_probabilities


@jit(nopython=True)
def _get_backward_probabilities(
    data: np.ndarray,
    covariates: np.ndarray,
    num_states: int,
    log_A: np.ndarray,
    log_B: np.ndarray,
):
    T = len(data)

    # Initialize the log backward probabilities matrix
    backward_probabilities = np.zeros((T, num_states))
    backward_probabilities[-1] = 0.0

    # Perform the backward algorithm
    for t in range(T - 2, -1, -1):
        for i in range(num_states):
            log_sum = (
                backward_probabilities[t + 1, 0]
                + log_B[covariates[t + 1]][0, data[t + 1]]
                + log_A[covariates[t]][i, 0]
            )
            for j in range(1, num_states):
                log_sum = np.logaddexp(
                    log_sum,
                    backward_probabilities[t + 1, j]
                    + log_B[covariates[t + 1]][j, data[t + 1]]
                    + log_A[covariates[t]][i, j],
                )
            backward_probabilities[t, i] = log_sum

    return backward_probabilities


def calc_xi_with_covariates(
    data: np.ndarray,
    covariates: np.ndarray,
    forward_probabilities: np.ndarray,
    backward_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
    log_denominator: float,
    n_covariates: int = None,
) -> np.ndarray:
    """
    Calculates the xi probabilities for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        covariates (np.ndarray): 1D Array of observed covariates
        forward_probabilities (np.ndarray): Matrix of log forward probabilities.
        backward_probabilities (np.ndarray): Matrix of log backward probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.
        n_covariates (int):
            Number of covariates.
            Defaults to None, and using the number of unique covariates in the data.
    Returns:
        np.ndarray: Matrix of xi probabilities.
    """
    # Get the number of states and the length of the data
    num_states = len(initial_probabilities)
    T = len(data)
    if not n_covariates:
        n_covariates = len(set(covariates))

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)

    log_xi = np.zeros((T - 1, num_states, num_states))

    # calculate the xi probabilities
    log_xi = _get_log_xi(
        data,
        covariates,
        num_states,
        forward_probabilities,
        backward_probabilities,
        log_A,
        log_B,
        log_denominator,
    )

    xi = np.exp(log_xi)

    return xi


@jit(nopython=True)
def _get_log_xi(
    data: np.ndarray,
    covariates: np.ndarray,
    num_states: int,
    forward_probabilities: np.ndarray,
    backward_probabilities: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_denominator: float,
) -> np.ndarray:
    T = len(data)
    log_xi = np.zeros((T - 1, num_states, num_states))

    for t in range(T - 1):
        for i in range(num_states):
            for j in range(num_states):
                log_xi[t, i, j] = (
                    forward_probabilities[t, i]
                    + backward_probabilities[t + 1, j]
                    + log_B[covariates[t + 1]][j, data[t + 1]]
                    + log_A[covariates[t]][i, j]
                    - log_denominator
                )
    return log_xi


def _do_estep_with_covariates(
    data_list: list[np.ndarray],
    covariates_list: list[np.ndarray],
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> tuple[float, dict]:
    """
    Performs the E-step of the EM algorithm for Hidden Markov Models (HMM) with covariates.

    Args:
        data_list (list[np.ndarray]): A list of 1D numpy arrays containing the observed outcomes sequences.
        covariates_list (list[np.ndarray]): A list of 1D numpy arrays containing the observed covariate sequences.
        transition_matrix (np.ndarray): The transition matrix of the HMM.
        emission_matrix (np.ndarray): The emission matrix of the HMM.
        initial_probabilities (np.ndarray): The initial probabilities of the HMM.

    Returns:
        llks (float): The log-likelihood is a float value representing the log-likelihood of the model given the observed data and parameters.

        statistics (dict): The statistics is a dictionary containing the updated statistics of the model. The dictionary has the following keys:
        - 'startprob': A numpy array representing the updated initial probabilities of the HMM.
        - 'transmat': A numpy array representing the updated transition matrix of the HMM.
        - 'emissmat': A numpy array representing the updated emission matrix of the HMM.

    """

    n_hiddens = len(initial_probabilities)
    n_outcomes = emission_matrix.shape[-1]
    n_covariates = len(
        set(np.concatenate(covariates_list).ravel())
    )  # should consider some sequence only has incomplete covariate realizations

    stats = _initialize_statistics(n_covariates, n_hiddens, n_outcomes)
    log_likelihood = 0

    for sub_data, covariates in zip(data_list, covariates_list):
        llk, forward_probabilities = forward_with_covariates(
            sub_data,
            covariates,
            transition_matrix,
            emission_matrix,
            initial_probabilities,
        )
        _, backward_probabilities = backward_with_covariates(
            sub_data,
            covariates,
            transition_matrix,
            emission_matrix,
            initial_probabilities,
        )
        gamma = calc_gamma(forward_probabilities, backward_probabilities)
        xi = calc_xi_with_covariates(
            sub_data,
            covariates,
            forward_probabilities,
            backward_probabilities,
            transition_matrix,
            emission_matrix,
            initial_probabilities,
            log_denominator=llk,
            n_covariates=n_covariates,
            # provide n_covariates since some sequence only has
            # incomplete covariate realizations
        )

        log_likelihood += llk

        # update by each covariate value
        for cov in range(n_covariates):
            stats["startprob"] += gamma[0]
            stats["transmat"][cov] += xi[(covariates[:-1] == cov)].sum(axis=0)
            np.add.at(
                stats["emissmat"][cov].T,
                sub_data[(covariates == cov)],
                gamma[(covariates == cov)],
            )

    return log_likelihood, stats


def _initialize_statistics(n_covariates, n_hiddens, n_outcomes):
    stats = {
        "startprob": np.zeros((n_hiddens,)),
        "transmat": np.zeros((n_covariates, n_hiddens, n_hiddens)),
        "emissmat": np.zeros((n_covariates, n_hiddens, n_outcomes)),
    }
    return stats


def baum_welch_with_covariates(
    data_list: list[np.ndarray],
    covariates_list: list[np.ndarray],
    n_hiddens: int,
    n_outcomes: int,
    pretrains: tuple[np.ndarray] = None,
    n_iters: int = 500,
    tol=1e-3,
    verbose=True,
):
    """
    Baum-Welch algorithm for Hidden Markov Models with covariates.

    This function performs the Baum-Welch algorithm to estimate the parameters of a Hidden Markov Model (HMM)
    with covariates. The HMM is trained using the given data and covariates, and the estimated parameters are
    returned.

    Parameters:
        data_list (list[np.ndarray]): A list of 1D numpy arrays representing the observed outcomes sequences.
        covariates_list (list[np.ndarray]): A list of 1D numpy arrays representing the covariates for each data sequence.
        n_hiddens (int): The number of hidden states in the HMM.
        n_outcomes (int): The number of possible outcomes in the HMM.
        pretrains (tuple[np.ndarray]): A tuple contains (transition, emission, initial probs) np.ndarrys.
        n_iters (int, optional): The maximum number of iterations for the Baum-Welch algorithm. Default is 500.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-3.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        llks (list[float]): A list of log likelihoods at each iteration of the algorithm.
        parameters (tuple): A tuple containing the estimated parameters of the HMM:
            transition_matrix (np.ndarray): The estimated transition matrix of shape (n_covariates, n_hiddens, n_hiddens).
            emission_matrix (np.ndarray): The estimated emission matrix of shape (n_covariates, n_hiddens, n_outcomes).
            initial_probabilities (np.ndarray): The estimated initial probabilities of shape (n_hiddens).

    """
    n_covariates = len(set(np.concatenate(covariates_list).ravel()))

    # initialize parameters if pretrains are not given
    if pretrains is None:
        transition_matrix = normalize(
            np.random.rand(n_covariates, n_hiddens, n_hiddens)
        )
        emission_matrix = normalize(np.random.rand(n_covariates, n_hiddens, n_outcomes))
        initial_probabilities = normalize(np.random.rand(n_hiddens))
    else:
        (transition_matrix, emission_matrix, initial_probabilities) = pretrains

    llks = []
    # training loop
    prev_log_likelihood = None
    for n_iter in tqdm(range(n_iters), disable=not verbose):
        # E-step
        log_likelihood, stats = _do_estep_with_covariates(
            data_list,
            covariates_list,
            transition_matrix,
            emission_matrix,
            initial_probabilities,
        )
        llks.append(log_likelihood)
        # Check convergence
        if (prev_log_likelihood is not None) and (
            np.abs(log_likelihood - prev_log_likelihood) < tol
        ):
            print(f"Algorithm converged after {n_iter} iters.")
            break
        prev_log_likelihood = log_likelihood

        # M-step: update parameters
        (transition_matrix, emission_matrix, initial_probabilities) = _do_mstep(stats)

    return llks, (transition_matrix, emission_matrix, initial_probabilities)
