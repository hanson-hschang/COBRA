import numpy as np

from cobra.math_tool import average2D, pointwise_multiplication


class TestMathTool:
    n_dim = 3
    n_elements = 10

    def test_average2D(self) -> None:
        vector = np.random.rand(self.n_dim, self.n_elements)
        result = average2D(vector)
        np.testing.assert_allclose(
            result, 0.5 * (vector[:, :-1] + vector[:, 1:])
        )

    def test_pointwise_multiplication(self) -> None:
        vector_a = np.random.rand(self.n_dim, self.n_elements)
        vector_b = np.random.rand(self.n_dim, self.n_elements)
        result = pointwise_multiplication(vector_a, vector_b)
        np.testing.assert_allclose(result, vector_a * vector_b)
