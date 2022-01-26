from abc import ABC, abstractmethod

import numpy as np
from sklearn.neural_network import MLPRegressor  # type: ignore

# TODO, work in progress


class NeuralNetwork(ABC):
    @abstractmethod
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Fit the network to the data.

        Parameters
        ----------
        x_data : np.ndarray
            [description]
        y_data : np.ndarray
            [description]
        """


class SklearnNetwork(NeuralNetwork):
    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        pass
