import numpy as np
from elastica import CosseratRod

from cobr2.actuations.actuation import ContinuousActuation


class TestActuation:
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
    actuation = ContinuousActuation(n_elements=n_elements)

    def test_actuation_shape(self):
        assert self.actuation.internal_force.shape == (
            self.n_dim,
            self.n_elements,
        )
        assert self.actuation.internal_couple.shape == (
            self.n_dim,
            self.n_elements - 1,
        )
        assert self.actuation.equivalent_external_force.shape == (
            self.n_dim,
            self.n_elements + 1,
        )
        assert self.actuation.equivalent_external_couple.shape == (
            self.n_dim,
            self.n_elements,
        )

    def test_actuation_reset(self):
        self.actuation.internal_force[0, :] = 1
        self.actuation.internal_couple[0, :] = 1
        self.actuation.equivalent_external_force[0, :] = 1
        self.actuation.equivalent_external_couple[0, :] = 1

        self.actuation.reset_actuation()

        assert (self.actuation.internal_force == 0).all()
        assert (self.actuation.internal_couple == 0).all()
        assert (self.actuation.equivalent_external_force == 0).all()
        assert (self.actuation.equivalent_external_couple == 0).all()

    def test_actuation_call(self):
        self.actuation(self.rod)
        assert True
