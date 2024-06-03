import numpy as np
from numba import njit


@njit(cache=True)  # type: ignore
def average2D(vector: np.ndarray) -> np.ndarray:
    return 0.5 * (vector[:, :-1] + vector[:, 1:])


@njit(cache=True)  # type: ignore
def pointwise_multiplication(
    vector_a: np.ndarray, vector_b: np.ndarray
) -> np.ndarray:
    return vector_a * vector_b
