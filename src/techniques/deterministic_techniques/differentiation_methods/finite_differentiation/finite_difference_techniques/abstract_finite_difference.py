import numpy as np
from abc import ABC, abstractmethod
from src.techniques.deterministic_techniques.essentials.base_case import BaseCase

"""
Further info about Abstract Class:
https://codefather.tech/blog/python-abstract-class/
"""


class AbstractFiniteDifferences(ABC):

    def __init__(self):
        self.dataset = None
        self._max_iterations = 100
        self._epsilon = np.finfo("float32").eps
        self._uniform_perturbation_size = np.sqrt(self._epsilon)
        self._uniform_perturbation_reduction_factor = np.float32(1)  # todo: play with this number and analyse the results !

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        self._max_iterations = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def uniform_perturbation_size(self):
        return self.uniform_perturbation_size

    @uniform_perturbation_size.setter
    def uniform_perturbation_size(self, value):
        self.uniform_perturbation_size = value

    @property
    def perturbation_reduction_factor(self):
        return self._uniform_perturbation_reduction_factor

    @perturbation_reduction_factor.setter
    def perturbation_reduction_factor(self, value):
        self.perturbation_reduction_factor = value

    def get_base_point(self):
        base_point_handler = BaseCase()
        base_point = base_point_handler.get_means(self.dataset.X)
        return base_point

    def get_perturbation_size(self, x_j, is_proportional_step, is_scaled=True):
        """
        There are two kinds of change: uniform or proportional
        """
        if not is_proportional_step:
            return np.sqrt(self._epsilon)
        if not is_scaled:
            return x_j * (np.sqrt(self._epsilon))
        return (1+x_j) * np.sqrt(self._epsilon)

    @abstractmethod
    def check_convergence(self, *args):
        pass

    @abstractmethod
    def calculate_difference_quotient(self, *args):
        pass

    @abstractmethod
    def compute_partial_derivative(self, *args):
        pass

    @abstractmethod
    def compute_derivatives(self, *args):
        pass





