import unittest

import numpy as np

from hmm_kit.core.base import forward
from hmm_kit.core.base import backward
from hmm_kit.core.covariates import forward_with_covariates
from hmm_kit.core.covariates import backward_with_covariates
import hmm_kit.core.base as base_hmm
import hmm_kit.core.covariates as cov_hmm


class BaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # TODO: add test cases with mal-defined probabilities
        self.test_cases = [
            {  # normal base case
                "type": "base",
                "data": np.array([0, 1, 0, 1, 0]),
                "covariates": np.array([0, 0, 0, 0, 0]),
                "transition_matrix": np.array([[[0.7, 0.3], [0.4, 0.6]]]),
                "emission_matrix": np.array([[[0.5, 0.5], [0.4, 0.6]]]),
                "initial_probabilities": np.array([0.6, 0.4]),
            },
            {  # normal covariates dependent case
                "type": "covariates",
                "data": np.array([0, 1, 0, 1, 0, 0, 1]),
                "covariates": np.array([0, 1, 0, 1, 0, 0, 1]),
                "transition_matrix": np.array(
                    [
                        [[0.9, 0.1], [0.1, 0.9]],
                        [[0.7, 0.3], [0.3, 0.7]],
                    ]
                ),
                "emission_matrix": np.array(
                    [
                        [[0.5, 0.5], [0.4, 0.6]],
                        [[0.1, 0.9], [0.2, 0.8]],
                    ]
                ),
                "initial_probabilities": np.array([0.6, 0.4]),
            },
        ]

        return super().setUp()


class TestForwardBackwordAlgorithmWithCovariates(BaseTestCase):
    def test_foward_core_functionality(self):
        """Compare the results of the forward algorithm with and without covariates."""
        for test_case in self.test_cases:
            if test_case["type"] != "base":
                # complicated test cases are handled in other tests
                continue

            data = test_case["data"]
            covariates = test_case["covariates"]
            transition_matrix = test_case["transition_matrix"]
            emission_matrix = test_case["emission_matrix"]
            initial_probabilities = test_case["initial_probabilities"]

            # Calculate expected resutls using the base forward algorithm
            # remove the extra dimension for base algorithm inputs
            A = (
                transition_matrix.squeeze(0)
                if transition_matrix.ndim > 2
                else transition_matrix
            )
            B = (
                emission_matrix.squeeze(0)
                if emission_matrix.ndim > 2
                else emission_matrix
            )
            expected_log_likelihood, expected_forward_probabilities = forward(
                data, A, B, initial_probabilities
            )

            # Calculate the results using the forward algorithm with covariates
            log_likelihood, forward_probabilities = forward_with_covariates(
                data,
                covariates,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
            )

            # Check the log likelihood
            self.assertTrue(np.isclose(log_likelihood, expected_log_likelihood))

            # Check the shape of the forward probabilities
            self.assertEqual(
                forward_probabilities.shape, expected_forward_probabilities.shape
            )

            # Check the forward probabilities
            self.assertTrue(
                np.allclose(forward_probabilities, expected_forward_probabilities)
            )

    def test_backward_core_functionality(self):
        """Compare the results of the backward algorithm with and without covariates."""
        for test_case in self.test_cases:
            if test_case["type"] != "base":
                # complicated test cases are handled in other tests
                continue

            data = test_case["data"]
            covariates = test_case["covariates"]
            transition_matrix = test_case["transition_matrix"]
            emission_matrix = test_case["emission_matrix"]
            initial_probabilities = test_case["initial_probabilities"]

            # Calculate expected resutls using the base forward algorithm
            # remove the extra dimension for base algorithm inputs
            A = (
                transition_matrix.squeeze(0)
                if transition_matrix.ndim > 2
                else transition_matrix
            )
            B = (
                emission_matrix.squeeze(0)
                if emission_matrix.ndim > 2
                else emission_matrix
            )
            expected_log_likelihood, expected_backward_probabilities = backward(
                data, A, B, initial_probabilities
            )

            # Calculate the results using the forward algorithm with covariates
            log_likelihood, forward_probabilities = backward_with_covariates(
                data,
                covariates,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
            )

            # Check the log likelihood
            # Check the log likelihood
            self.assertTrue(np.isclose(log_likelihood, expected_log_likelihood))

            # Check the shape of the forward probabilities
            self.assertEqual(
                forward_probabilities.shape, expected_backward_probabilities.shape
            )

            # Check the forward probabilities
            self.assertTrue(
                np.allclose(forward_probabilities, expected_backward_probabilities)
            )

    def test_forward_backward_log_likelihood(self):
        """Compare the log likelihoods calculated by the forward and backward algorithms."""
        for test_case in self.test_cases:
            if test_case["type"] != "base":
                # complicated test cases are handled in other tests
                continue

            data = test_case["data"]
            transition_matrix = test_case["transition_matrix"]
            emission_matrix = test_case["emission_matrix"]
            initial_probabilities = test_case["initial_probabilities"]

            # adjuste the shape of the transition and emission matrices for base inputs
            A = (
                transition_matrix.squeeze(0)
                if transition_matrix.ndim > 2
                else transition_matrix
            )
            B = (
                emission_matrix.squeeze(0)
                if emission_matrix.ndim > 2
                else emission_matrix
            )
            # Calculate the results using the forward algorithm with covariates
            log_likelihood_forward, _ = forward(data, A, B, initial_probabilities)

            # Calculate the results using the backward algorithm with covariates
            log_likelihood_backward, _ = backward(data, A, B, initial_probabilities)

            # Check the log likelihood
            self.assertTrue(np.isclose(log_likelihood_forward, log_likelihood_backward))

    def test_forward_backward_log_likelihood_with_covariates(self):
        """Compare the log likelihoods calculated by the forward and backward algorithms with covariates."""
        for test_case in self.test_cases:
            if test_case["type"] != "covariates":
                # base test cases are handled in other tests
                continue

            data = test_case["data"]
            covariates = test_case["covariates"]
            transition_matrix = test_case["transition_matrix"]
            emission_matrix = test_case["emission_matrix"]
            initial_probabilities = test_case["initial_probabilities"]

            # Calculate the results using the forward algorithm with covariates
            log_likelihood_forward, _ = forward_with_covariates(
                data,
                covariates,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
            )

            # Calculate the results using the backward algorithm with covariates
            log_likelihood_backward, _ = backward_with_covariates(
                data,
                covariates,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
            )

            # Check the log likelihood
            self.assertTrue(np.isclose(log_likelihood_forward, log_likelihood_backward))


class TestGammXiStatistics(BaseTestCase):
    def test_xi_statistics_with_covariates(self):
        """Compare the results of the xi statistics with and without covariates."""
        for test_case in self.test_cases:
            if test_case["type"] != "base":
                # complicated test cases are handled in other tests
                continue

            data = test_case["data"]
            covariates = test_case["covariates"]
            transition_matrix = test_case["transition_matrix"]
            emission_matrix = test_case["emission_matrix"]
            initial_probabilities = test_case["initial_probabilities"]

            # Calculate expected resutls using the base forward algorithm
            # remove the extra dimension for base algorithm inputs
            A = (
                transition_matrix.squeeze(0)
                if transition_matrix.ndim > 2
                else transition_matrix
            )
            B = (
                emission_matrix.squeeze(0)
                if emission_matrix.ndim > 2
                else emission_matrix
            )

            log_likelihood, forward_probabilities = base_hmm.forward(
                data, A, B, initial_probabilities
            )
            _, backward_probabilities = base_hmm.forward(
                data, A, B, initial_probabilities
            )

            expected_xi = base_hmm.calc_xi(
                data, forward_probabilities, backward_probabilities, A, B
            )

            # Calculate the xi with covariates
            xi = cov_hmm.calc_xi_with_covariates(
                data,
                covariates,
                forward_probabilities,
                backward_probabilities,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
                log_likelihood,
                n_covariates=1,
            )

            # Check the log likelihood
            self.assertTrue(np.allclose(xi, expected_xi))


if __name__ == "__main__":
    unittest.main()
