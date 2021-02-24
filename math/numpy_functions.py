import numpy as np


def square(x: np.ndarray) -> np.ndarray:
    """
    It will calculate the square value of each elements in the array

    :param x:
    :return:
    """

    return np.power(x, 2)


def leaky_rely(x: np.ndarray) -> np.ndarray:
    """
    It applies the "Leaky Relu" function to each elements in the array.
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU

    Leaky ReLUs allow a small, positive gradient when the unit is not active.

    :param x:
    :return:
    """

    return np.maximum(0.9 * x, x)


res_square = square(np.array([1, 2, 3]))
print(f"\r\nSquare results from each elements in the np.array\r\n{res_square}\r\n")

res_relu = leaky_rely(np.array([-2, 1, 2, 3]))
print(f"The ReLU values from each elements in the array\r\n{res_relu}\r\n")
