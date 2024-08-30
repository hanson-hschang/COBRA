import numpy as np
from elastica._calculus import _difference, quadrature_kernel
from elastica._linalg import _batch_cross
from numba import njit

from cobra.math_tool import average2D as _average


# @njit(cache=True)
def pos_dir_to_input(pos: np.ndarray, dir: np.ndarray) -> np.ndarray:
    input_orien : np.ndarray = dir.reshape(len(dir), -1, dir.shape[-1])
    inputs : np.ndarray = np.hstack([pos, input_orien])
    return inputs

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

@njit(cache=True)
def forward_path(dl, shear, kappa, position_collection, director_collection):
    # _, voronoi_dilatation = calculate_dilatation(shear)
    # curvature = kappa_to_curvature(kappa, voronoi_dilatation)
    for i in range(dl.shape[0] - 1):
        next_position(
            director_collection[:, :, i],
            shear[:, i] * dl[i],
            position_collection[:, i : i + 2],
        )
        next_director(kappa[:, i] * dl[i], director_collection[:, :, i : i + 2])
    next_position(
        director_collection[:, :, -1],
        shear[:, -1] * dl[-1],
        position_collection[:, -2:],
    )


@njit(cache=True)
def next_position(director, delta, positions):
    positions[:, 1] = positions[:, 0]
    for index_i in range(3):
        for index_j in range(3):
            positions[index_i, 1] += director[index_j, index_i] * delta[index_j]
    return


@njit(cache=True)
def next_director(rotation, directors):
    Rotation = get_rotation_matrix(rotation)
    for index_i in range(3):
        for index_j in range(3):
            directors[index_i, index_j, 1] = 0
            for index_k in range(3):
                directors[index_i, index_j, 1] += (
                    Rotation[index_k, index_i] * directors[index_k, index_j, 0]
                )
    return


@njit(cache=True)
def get_rotation_matrix(axis):
    angle = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)

    axis = axis / (angle + 1e-8)
    K = np.zeros((3, 3))
    K[2, 1] = axis[0]
    K[1, 2] = -axis[0]
    K[0, 2] = axis[1]
    K[2, 0] = -axis[1]
    K[1, 0] = axis[2]
    K[0, 1] = -axis[2]

    K2 = np.zeros((3, 3))
    K2[0, 0] = -(axis[1] * axis[1] + axis[2] * axis[2])
    K2[1, 1] = -(axis[2] * axis[2] + axis[0] * axis[0])
    K2[2, 2] = -(axis[0] * axis[0] + axis[1] * axis[1])
    K2[0, 1] = axis[0] * axis[1]
    K2[1, 0] = axis[0] * axis[1]
    K2[0, 2] = axis[0] * axis[2]
    K2[2, 0] = axis[0] * axis[2]
    K2[1, 2] = axis[1] * axis[2]
    K2[2, 1] = axis[1] * axis[2]

    Rotation = np.sin(angle) * K + (1 - np.cos(angle)) * K2
    Rotation[0, 0] += 1
    Rotation[1, 1] += 1
    Rotation[2, 2] += 1

    return Rotation