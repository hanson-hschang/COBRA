import numpy as np
from numba import njit


@njit(cache=True)  # type: ignore
def difference2D(vector_collection: np.ndarray) -> np.ndarray:
    return vector_collection[:, 1:] - vector_collection[:, :-1]


@njit(cache=True)  # type: ignore
def average2D(vector_collection: np.ndarray) -> np.ndarray:
    return (vector_collection[:, 1:] + vector_collection[:, :-1]) / 2
