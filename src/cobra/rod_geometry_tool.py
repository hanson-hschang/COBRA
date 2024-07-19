import numpy as np
from elastica._calculus import _difference, quadrature_kernel
from elastica._linalg import _batch_cross
from numba import njit

from cobra.math_tool import average2D as _average


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


@njit(cache=True)  # type: ignore
def compute_local_tangent(local_shear: np.ndarray) -> np.ndarray:
    blocksize = local_shear.shape[1]
    local_tangent = np.empty((3, blocksize))
    for i in range(blocksize):
        local_tangent[:, i] = local_shear[:, i] / np.sqrt(
            (local_shear[0, i]) ** 2
            + (local_shear[1, i]) ** 2
            + (local_shear[2, i]) ** 2
        )
    return local_tangent
