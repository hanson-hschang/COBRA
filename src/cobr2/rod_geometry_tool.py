import numpy as np
from numba import njit


@njit(cache=True)  # type: ignore
def sigma_to_shear(sigma: np.ndarray) -> np.ndarray:
    shear = sigma.copy()
    shear[2, :] += 1
    return shear
