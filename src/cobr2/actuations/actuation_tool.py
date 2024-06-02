import numpy as np
from elastica._calculus import difference_kernel, quadrature_kernel
from elastica._linalg import _batch_cross, _batch_matvec
from numba import njit


@njit(cache=True)
def lab_to_material(
    directors: np.ndarray, lab_vectors: np.ndarray
) -> np.ndarray:
    return _batch_matvec(directors, lab_vectors)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def average2D(vector_collection: np.ndarray) -> np.ndarray:
    blocksize = vector_collection.shape[1] - 1
    output_vector = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            output_vector[i, n] = (
                vector_collection[i, n] + vector_collection[i, n + 1]
            ) / 2
    return output_vector


@njit(cache=True)
def force_induced_couple(
    distance: np.ndarray,
    force: np.ndarray,
) -> np.ndarray:
    return average2D(_batch_cross(distance, force))
