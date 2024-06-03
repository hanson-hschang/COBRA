import numpy as np

from cobr2._calculus import average2D, difference2D


class TestCalculus:
    n_dim = 3
    n_elements = 10

    def test_difference2D(self) -> None:
        vector_collection = np.random.rand(self.n_dim, self.n_elements)
        difference = vector_collection[:, 1:] - vector_collection[:, :-1]
        assert np.allclose(difference, difference2D(vector_collection))

    def test_average2D(self) -> None:
        vector_collection = np.random.rand(self.n_dim, self.n_elements)
        average = (vector_collection[:, 1:] + vector_collection[:, :-1]) / 2
        assert np.allclose(average, average2D(vector_collection))
