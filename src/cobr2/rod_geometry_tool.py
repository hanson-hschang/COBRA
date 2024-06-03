import numpy as np
from elastica._calculus import _difference, difference_kernel, quadrature_kernel
from elastica._linalg import _batch_cross, _batch_matvec
from numba import njit

from cobr2._calculus import average2D as _average


@njit(cache=True)  # type: ignore
def sigma_to_shear(sigma: np.ndarray) -> np.ndarray:
    shear = sigma.copy()
    shear[2, :] += 1
    return shear


@njit(cache=True)  # type: ignore
def compute_local_shear(
    local_position: np.ndarray,
    shear: np.ndarray,
    kappa: np.ndarray,
    delta_s: np.ndarray,
) -> np.ndarray:
    local_shear: np.ndarray = shear + quadrature_kernel(
        _batch_cross(kappa, _average(local_position))
        + _difference(local_position) / delta_s
    )
    return local_shear
