import numpy as np
from numba import njit


@njit(cache=True)  # type: ignore
def average2D(vector: np.ndarray) -> np.ndarray:
    result: np.ndarray = 0.5 * (vector[:, :-1] + vector[:, 1:])
    return result


@njit(cache=True)  # type: ignore
def pointwise_multiplication(
    vector_a: np.ndarray, vector_b: np.ndarray
) -> np.ndarray:
    result: np.ndarray = vector_a * vector_b
    return result
