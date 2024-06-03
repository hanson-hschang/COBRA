import numpy as np

from cobr2.rod_geometry_tool import sigma_to_shear


class TestRodGeometryTool:
    n_dim = 3
    n_elements = 10

    def test_sigma_to_shear(self) -> None:
        sigma = np.random.rand(self.n_dim, self.n_elements)

        shear = sigma_to_shear(sigma)
        for n in range(self.n_dim):
            for i in range(self.n_elements):
                if n == 2:
                    assert shear[n, i] == sigma[n, i] + 1
                else:
                    assert shear[n, i] == sigma[n, i]
