from typing import Callable
import numpy as np


def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:

    """
    Derivative of the function 'func' for each element in the matrix input_
    """

    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)
