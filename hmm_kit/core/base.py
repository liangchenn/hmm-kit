import numpy as np
from numba import jit
from tqdm import tqdm

from hmm_kit.constant import SMALL_CONSTANT
from hmm_kit.utils import _initialize_statistics, normalize


def forward(
    data: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Performs the log forward algorithm for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        initial_probabilities (np.ndarray): Array of initial state probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.

    Returns:
        np.ndarray: Matrix of log forward probabilities.
    """
    # Get the number of states and the length of the data
    num_states = len(initial_probabilities)

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # calculate the log forward probabilities
    forward_probabilities = _get_forward_probabilities(
        data, num_states, log_A, log_B, log_pi
    )

    # Obtain the log likelihood
    log_likelihood = np.logaddexp.reduce(forward_probabilities[-1])

    return log_likelihood, forward_probabilities


def backward(
    data: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> np.ndarray:
    """
    Performs the log backward algorithm for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        initial_probabilities (np.ndarray): Array of initial state probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.

    Returns:
        np.ndarray: Matrix of log backward probabilities.
    """
    # Get the number of states and the length of the data
    num_states = len(initial_probabilities)

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)
    log_pi = np.log(initial_probabilities + SMALL_CONSTANT)

    # calculate the log backward probabilities
    backward_probabilities = _get_backward_probabilities(data, num_states, log_A, log_B)

    # Obtain the log likelihood
    log_likelihood = np.logaddexp.reduce(
        backward_probabilities[0] + log_B[:, data[0]] + log_pi
    )

    return log_likelihood, backward_probabilities


def calc_gamma(
    forward_probabilities: np.ndarray, backward_probabilities: np.ndarray
) -> np.ndarray:
    """
    Calculates the gamma statistics for Hidden Markov Models (HMM).

    Args:
        forward_probabilities (np.ndarray): Matrix of Log forward probabilities.
        backward_probabilities (np.ndarray): Matrix of Log backward probabilities.

    Returns:
        np.ndarray: Matrix of gamma statistics.
    """
    # obtain log_gamma by adding log forward and log backward probabilities
    log_gamma = forward_probabilities + backward_probabilities
    log_denominator = np.logaddexp.reduce(log_gamma[-1])
    # convert log_gamma to gamma to maintain in probability space
    gamma = np.exp(log_gamma - log_denominator)
    return gamma


def calc_xi(
    data: np.ndarray,
    forward_probabilities: np.ndarray,
    backward_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculates the xi statistics for Hidden Markov Models (HMM).

    Args:
        data (np.ndarray): 1D Array of observed outcomes.
        forward_probabilities (np.ndarray): Matrix of log forward probabilities.
        backward_probabilities (np.ndarray): Matrix of log backward probabilities.
        transition_matrix (np.ndarray): Matrix of state transition probabilities.
        emission_matrix (np.ndarray): Matrix of emission probabilities.

    Returns:
        np.ndarray: Matrix of xi statistics.
    """
    # Get the number of states and the length of the data
    num_states = transition_matrix.shape[0]

    # change parameters to log domain
    log_A = np.log(transition_matrix + SMALL_CONSTANT)
    log_B = np.log(emission_matrix + SMALL_CONSTANT)

    log_denominator = np.logaddexp.reduce(forward_probabilities[-1])

    log_xi = _get_log_xi(
        data,
        num_states,
        forward_probabilities,
        backward_probabilities,
        log_A,
        log_B,
        log_denominator,
    )

    xi = np.exp(log_xi)

    return xi


def _do_estep(
    data_list: list[np.ndarray],
    transition_matrix: np.ndarray,
    emission_matrix: np.ndarray,
    initial_probabilities: np.ndarray,
) -> tuple:
    """
    Performs the E-step in the EM algorithm for HMM.

    Args:
        data_list (list[np.ndarray]): A list of 1D numpy arrays representing the observed outcomes sequences.
        transition_matrix (np.ndarray): The transition matrix of the HMM.
        emission_matrix (np.ndarray): The emission matrix of the HMM.
        initial_probabilities (np.ndarray): The initial probabilities of the hidden states.

    Returns:
        llks (float): The log-likelihood is a float value representing the log-likelihood of the model given the observed data and parameters.

        statistics (dict): The statistics is a dictionary containing the updated statistics of the model. The dictionary has the following keys:
        - 'startprob': A numpy array representing the updated initial probabilities of the HMM.
        - 'transmat': A numpy array representing the updated transition matrix of the HMM.
        - 'emissmat': A numpy array representing the updated emission matrix of the HMM.

    """

    n_hiddens = len(initial_probabilities)
    n_outcomes = emission_matrix.shape[-1]

    stats = _initialize_statistics(n_hiddens, n_outcomes)
    log_likelihood = 0

    for sub_data in data_list:
        llk, forward_probabilities = forward(
            sub_data, transition_matrix, emission_matrix, initial_probabilities
        )
        _, backward_probabilities = backward(
            sub_data, transition_matrix, emission_matrix, initial_probabilities
        )
        gamma = calc_gamma(forward_probabilities, backward_probabilities)
        xi = calc_xi(
            sub_data,
            forward_probabilities,
            backward_probabilities,
            transition_matrix,
            emission_matrix,
        )

        log_likelihood += llk

        stats["startprob"] += gamma[0]
        stats["transmat"] += xi.sum(axis=0)

        for i in range(n_hiddens):
            for k in range(n_outcomes):
                stats["emissmat"][i, k] += gamma[sub_data == k, i].sum()

    return log_likelihood, stats


def _do_mstep(stats):
    """
    Perform the M-step of the EM algorithm for HMM.

    This function takes the sufficient statistics computed in the E-step and updates the model parameters.

    Parameters:
        stats (dict): A dictionary containing the statistics computed in the E-step. It should have the following keys:
            - 'startprob' (np.ndarray): The initial state probabilities.
            - 'transmat' (np.ndarray): The transition matrix.
            - 'emissmat' (np.ndarray): The emission matrix.

    Returns:
        updated_transition (np.ndarray): The updated transition matrix.
        updated_emission (np.ndarray): The updated emission matrix.
        updated_initial_probs (np.ndarray): The updated initial state probabilities.
    """
    # statistics
    startprob = stats["startprob"]
    transmat = stats["transmat"]
    emissmat = stats["emissmat"]

    # remove small values in probability matrices
    startprob = np.where(np.isclose(startprob, 0), 0, startprob)
    transmat = np.where(np.isclose(transmat, 0), 0, transmat)
    emissmat = np.where(np.isclose(emissmat, 0), 0, emissmat)

    # normalize the accumulated statistics
    updated_initial_probs = normalize(startprob)
    updated_transition = normalize(transmat)
    updated_emission = normalize(emissmat)

    return updated_transition, updated_emission, updated_initial_probs


def baum_welch(
    data_list: list[np.ndarray],
    n_hiddens: int,
    n_outcomes: int,
    pretrains: tuple[np.ndarray] = None,
    n_iters: int = 500,
    tol=1e-3,
    verbose=True,
):
    """
    Baum-Welch algorithm for training Hidden Markov Models (HMMs).

    Parameters:
        data_list (list[np.ndarray]): A list of 1D numpy arrays representing the observed outcomes sequences.
        n_hiddens (int): The number of hidden states in the HMM.
        n_outcomes (int): The number of possible outcomes in the HMM.
        pretrains (tuple[np.ndarray]): A tuple contains (transition, emission, initial probs) np.ndarrys.
        n_iters (int, optional): The maximum number of iterations for the algorithm. Defaults to 500.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-3.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        llks (list[float]): A list of log likelihoods at each iteration of the algorithm.
        parameters (tuple): A tuple containing the estimated parameters of the HMM:
            transition_matrix (np.ndarray): The estimated transition matrix of shape (n_covariates, n_hiddens, n_hiddens).
            emission_matrix (np.ndarray): The estimated emission matrix of shape (n_covariates, n_hiddens, n_outcomes).
            initial_probabilities (np.ndarray): The estimated initial probabilities of shape (n_hiddens).

    """
    # initialize parameters
    if pretrains is None:
        transition_matrix = normalize(np.random.rand(n_hiddens, n_hiddens))
        emission_matrix = normalize(np.random.rand(n_hiddens, n_outcomes))
        initial_probabilities = normalize(np.random.rand(n_hiddens))
    else:
        (transition_matrix, emission_matrix, initial_probabilities) = pretrains

    llks = []
    # training loop
    prev_log_likelihood = None
    for n_iter in tqdm(range(n_iters), disable=not verbose):
        # E-step: accumulate statistics from list of data (sequences)
        log_likelihood, stats = _do_estep(
            data_list, transition_matrix, emission_matrix, initial_probabilities
        )
        llks.append(log_likelihood)
        # check convergence
        if (prev_log_likelihood is not None) and (
            np.abs(log_likelihood - prev_log_likelihood) < tol
        ):
            print(f"Algorithm converged after {n_iter} iters.")
            break
        prev_log_likelihood = log_likelihood

        # M-step: update parameters
        transition_matrix, emission_matrix, initial_probabilities = _do_mstep(stats)

    return llks, (transition_matrix, emission_matrix, initial_probabilities)


@jit(nopython=True)
def _get_log_xi(
    data: np.ndarray,
    num_states: int,
    forward_probabilities: np.ndarray,
    backward_probabilities: np.ndarray,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_denominator: float,
) -> np.ndarray:
    """
    Optimized implementation of the xi statistics calculation for HMMs.
    """

    T = len(data)
    log_xi = np.zeros((T - 1, num_states, num_states))
    for t in range(T - 1):
        for i in range(num_states):
            for j in range(num_states):
                log_xi[t, i, j] = (
                    forward_probabilities[t, i]
                    + backward_probabilities[t + 1, j]
                    + log_B[j, data[t + 1]]
                    + log_A[i, j]
                    - log_denominator
                )
    return log_xi


@jit(nopython=True)
def _get_forward_probabilities(
    data: np.ndarray,
    num_states: int,
    log_A: np.ndarray,
    log_B: np.ndarray,
    log_pi: np.ndarray,
):
    """
    Optimized implementation of the forward algorithm for HMMs.
    """
    T = len(data)

    forward_probabilities = np.zeros((T, num_states))
    forward_probabilities[0] = log_pi + log_B[:, data[0]]

    for t in range(1, T):
        for i in range(num_states):
            log_sum = forward_probabilities[t - 1, 0] + log_A[0, i]
            for j in range(1, num_states):
                log_sum = np.logaddexp(
                    log_sum, forward_probabilities[t - 1, j] + log_A[j, i]
                )
            forward_probabilities[t, i] = log_B[i, data[t]] + log_sum

    return forward_probabilities


@jit(nopython=True)
def _get_backward_probabilities(
    data: np.ndarray, num_states: int, log_A: np.ndarray, log_B: np.ndarray
):
    """
    Optimized implementation of the backward algorithm for HMMs.
    """
    T = len(data)

    # Initialize the log backward probabilities matrix
    backward_probabilities = np.zeros((T, num_states))
    backward_probabilities[-1] = 0.0

    # Perform the backward algorithm
    for t in range(T - 2, -1, -1):
        for i in range(num_states):
            log_sum = (
                backward_probabilities[t + 1, 0] + log_B[0, data[t + 1]] + log_A[i, 0]
            )
            for j in range(1, num_states):
                log_sum = np.logaddexp(
                    log_sum,
                    backward_probabilities[t + 1, j]
                    + log_B[j, data[t + 1]]
                    + log_A[i, j],
                )
            backward_probabilities[t, i] = log_sum

    return backward_probabilities
