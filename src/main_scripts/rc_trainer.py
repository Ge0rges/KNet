import torch
import numpy as np

from math import log
from collections import Counter
from src.main_scripts.pss_trainer import DENTrainer
from src.utils.misc import get_modules


class ResourceConstrainedTrainer:
    """
    A training algorithm for neural networks that optimizes structure based on resources used.
    Uses a DENTrainer as backend for most functions.
    """
    def __init__(self, den_trainer: DENTrainer, bits_available: int) -> None:
        raise NotImplementedError

        self._den_trainer = den_trainer
        self.max_entropy = self.calculate_max_entropy(bits_available)

    def calculate_max_entropy(self, bits_available: int) -> float:
        max_number = 2**(bits_available-1)
        prob_dist = np.arange(max_number)

        total_count = np.sum(prob_dist)
        discrete_dist = [float(x) / total_count for x in prob_dist.flatten()]

        return self.entropy(discrete_dist)

    def get_model_entropy(self) -> float:
        weights = self.get_model_weights()
        total_entropy = 0

        # For our cases entropy is additive as each matrix
        # will supposedly encode different information from previous
        # Break case: Singular values.
        for weights in weights.values():
            total_entropy += self.matrix_entropy(weights)

        return total_entropy

    def get_model_weights(self) -> dict:
        model = self._den_trainer.model
        modules = get_modules(model)
        weights = {}

        # For each module
        for dict_key, module in modules.items():
            weights[dict_key] = []

            # For each layer
            for param_name, param in module:
                # Skip biases params
                if "bias" in param_name:
                    continue

                weights[dict_key].append(param.clone().detach())

        return weights

    def matrix_entropy(self, matrix: torch.Tensor) -> float:
        """
        Calculates the "entropy" of a matrix by treating each element as
        independent and obtaining the histogram of element values
        @:param matrix the matrix to calculate entropy of
        @:returns the von neumann entropy of the matrix
        """
        counts = dict(Counter(matrix.flatten())).values()  # Singular values
        total_count = sum(counts)
        discrete_dist = [float(x) / total_count for x in counts]  # Normalize
        return self.entropy(discrete_dist)

    def entropy(self, probability_list: [float]) -> float:
        """
        Calculates the entropy of a specified discrete probability distribution
        @:param probability_list The discrete probability distribution
        @:returns the von neumann entropy of a probability list -[x1 * log(1) + .. + xn * log(xn)]
        """
        running_total = 0

        for item in probability_list:
            if item == 0:
                continue
            running_total += item * log(item, 2)

        return -running_total

    def matrix_rank(self, matrix: torch.Tensor) -> int:
        """
        :param matrix: The tensor to get rank of
        :type matrix: A tensor
        :return: The rank of the matrix
        :rtype: int
        """
        return np.linalg.matrix_rank(matrix)
