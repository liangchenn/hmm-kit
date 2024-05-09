import unittest

import numpy as np
from hmmlearn import hmm

from hmm_kit.core.viterbi import decode


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


class TestViterbiAlgorithm(BaseTestCase):
    def test_viterbi_basic_functionality(self):
        """Compare the results of viterbi decoder to the hmmlearn package results."""
        for test_case in self.test_cases:
            if test_case["type"] != "base":
                # complicated test cases are handled in other tests
                continue

            data = test_case["data"]
            covariate = test_case["covariates"]
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

            # expected results
            model = hmm.CategoricalHMM(
                n_components=len(initial_probabilities),  # num of hidden states
                n_features=emission_matrix.shape[-1],  # num of outcomes
            )
            # provide the parameters
            model.transmat_ = A
            model.emissionprob_ = B
            model.startprob_ = initial_probabilities

            expected_log_likelihood, expected_path = model.decode(
                data.reshape(-1, 1), lengths=[len(data)]
            )

            # calculated results
            log_likelihood, path = decode(
                data,
                covariate,
                transition_matrix,
                emission_matrix,
                initial_probabilities,
            )

            # check log likelihood
            self.assertEqual(log_likelihood, expected_log_likelihood)

            # check path
            self.assertTrue((path == expected_path).all())
