import numpy as np
from elastica import CosseratRod

from cobra.actuations.actuation_tool import (
    apply_load,
    force_induced_couple,
    internal_load_to_equivalent_external_load,
    lab_to_material,
    material_to_lab,
)


class TestActuationTool:
    n_dim = 3
    n_elements = 10
    poisson_ratio = 0.5
    rod = CosseratRod.straight_rod(
        n_elements=n_elements,
        start=np.zeros((3,)),
        direction=np.array([0.0, 0.0, -1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        base_length=1,
        base_radius=0.01 * np.ones(n_elements),
        density=1000,
        youngs_modulus=1e7,
        shear_modulus=1e7 / (poisson_ratio + 1.0),
    )

    # TODO: the following tests only test the shape of the output, not the correctness of the output

    def test_lab_to_material(self) -> None:
        director_collection = np.random.rand(
            self.n_dim, self.n_dim, self.n_elements
        )
        force = np.random.rand(self.n_dim, self.n_elements)

        result = lab_to_material(director_collection, force)

        assert result.shape == (self.n_dim, self.n_elements)

    def test_material_to_lab(self) -> None:
        director_collection = np.random.rand(
            self.n_dim, self.n_dim, self.n_elements
        )
        force = np.random.rand(self.n_dim, self.n_elements)

        result = material_to_lab(director_collection, force)

        assert result.shape == (self.n_dim, self.n_elements)

    def test_internal_load_to_equivalent_external_load(self) -> None:
        internal_force = np.random.rand(self.n_dim, self.n_elements)
        internal_couple = np.random.rand(self.n_dim, self.n_elements - 1)
        equivalent_external_force = np.zeros((self.n_dim, self.n_elements + 1))
        equivalent_external_couple = np.zeros((self.n_dim, self.n_elements))

        internal_load_to_equivalent_external_load(
            self.rod.director_collection,
            self.rod.kappa,
            self.rod.tangents,
            self.rod.rest_lengths,
            self.rod.rest_voronoi_lengths,
            self.rod.dilatation,
            self.rod.voronoi_dilatation,
            internal_force,
            internal_couple,
            equivalent_external_force,
            equivalent_external_couple,
        )

        assert equivalent_external_force.shape == (
            self.n_dim,
            self.n_elements + 1,
        )
        assert equivalent_external_couple.shape == (self.n_dim, self.n_elements)

    def test_force_induced_couple(self) -> None:
        distance = np.random.rand(self.n_dim, self.n_elements)
        force = np.random.rand(self.n_dim, self.n_elements)

        couple = force_induced_couple(
            distance,
            force,
        )

        assert couple.shape == (self.n_dim, self.n_elements)

    def test_apply_load(self) -> None:
        system_load = np.zeros((self.n_dim, self.n_elements))
        external_load = np.random.rand(self.n_dim, self.n_elements)

        apply_load(
            system_load,
            external_load,
        )

        assert np.array_equal(system_load, external_load)
