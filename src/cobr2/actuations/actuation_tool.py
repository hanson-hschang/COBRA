import numpy as np
from elastica._calculus import difference_kernel, quadrature_kernel
from elastica._linalg import _batch_cross, _batch_matvec
from numba import njit

from cobr2._calculus import average2D as _average

# adding njit decorator strips away function type annotations, breaking mypy's analysis
# adding # type: ignore to the function signature suppresses the error
# link to numba issue: https://github.com/numba/numba/issues/7424


@njit(cache=True)  # type: ignore
def lab_to_material(
    directors: np.ndarray, lab_vectors: np.ndarray
) -> np.ndarray:
    material_vectors: np.ndarray = _batch_matvec(directors, lab_vectors)
    return material_vectors


@njit(cache=True)  # type: ignore
def material_to_lab(
    directors: np.ndarray, material_vectors: np.ndarray
) -> np.ndarray:
    blocksize = material_vectors.shape[1]
    lab_vectors = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                lab_vectors[i, n] += directors[j, i, n] * material_vectors[j, n]
    return lab_vectors


@njit(cache=True)  # type: ignore
def internal_load_to_equivalent_external_load(
    director_collection: np.ndarray,
    kappa: np.ndarray,
    tangents: np.ndarray,
    rest_lengths: np.ndarray,
    rest_voronoi_lengths: np.ndarray,
    dilatation: np.ndarray,
    voronoi_dilatation: np.ndarray,
    internal_force: np.ndarray,
    internal_couple: np.ndarray,
    external_force: np.ndarray,
    external_couple: np.ndarray,
) -> None:
    external_force[:, :] = difference_kernel(
        material_to_lab(director_collection, internal_force)
    )
    external_couple[:, :] = (
        difference_kernel(internal_couple)
        + quadrature_kernel(
            _batch_cross(kappa, internal_couple) * rest_voronoi_lengths
        )
        + _batch_cross(
            lab_to_material(director_collection, tangents * dilatation),
            internal_force,
        )
        * rest_lengths
    )


@njit(cache=True)  # type: ignore
def force_induced_couple(
    distance: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    couple: np.ndarray = _average(_batch_cross(distance, force))
    return couple


@njit(cache=True)  # type: ignore
def apply_load(
    system_load: np.ndarray,
    external_load: np.ndarray,
) -> None:
    system_load[:, :] += external_load
