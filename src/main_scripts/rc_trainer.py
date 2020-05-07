import torch
import numpy as np

from math import log
from collections import Counter
from src.main_scripts.den_trainer import DENTrainer
from src.utils.misc import get_modules


class ResourceConstrainedTrainer:
    """
    A training algorithm for neural networks that optimizes structure based on resources used.
    Uses a DENTrainer as backend for most functions.
    """
    def __init__(self, den_trainer: DENTrainer, resources_available: int) -> None:
        raise NotImplementedError

        self._den_trainer = den_trainer
        self.max_entropy = None


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

    def matrix_entropy(self, matrix):
        """
        Calculates the "entropy" of a matrix by treating each element as
        independent and obtaining the histogram of element values
        @input matrix
        """
        counts = dict(Counter(matrix.flatten())).values()
        total_count = sum(counts)
        discrete_dist = [float(x) / total_count for x in counts]
        return self.entropy(discrete_dist)

    def entropy(self, probability_list):
        """
        Calculates the entropy of a specified discrete probability distribution
        @input probability_list The discrete probability distribution
        """
        running_total = 0

        for item in probability_list:
            running_total += item * log(item, 2)

        if running_total != 0:
            running_total *= -1

        return running_total

    def matrix_rank(self, matrix):
        return np.linalg.matrix_rank(matrix)
