from typing import Iterable

from dataclasses import dataclass

import elastica as ea
import numpy as np
from numba import njit

from cobr2.actuations import ApplyActuations, ContinuousActuation
from cobr2.actuations.actuation_tool import force_induced_couple
from cobr2.math_tool import average2D, pointwise_multiplication
from cobr2.rod_geometry_tool import (
    compute_local_shear,
    compute_local_tangent,
    sigma_to_shear,
)


@dataclass
class PressureCoefficients:
    """
    Dataclass containing pressure coefficients for force and couple.
    """

    force: np.ndarray
    couple: np.ndarray

    def set_n_elements(self, n_elements: int) -> None:
        self.ones = np.ones(n_elements)

    def get_force_value(self, pressure: float) -> np.ndarray:
        return np.polyval(self.force, pressure) * self.ones

    def get_couple_value(self, pressure: float) -> np.ndarray:
        return np.polyval(self.couple, pressure) * self.ones


class BaseFREE(ContinuousActuation):
    def __init__(
        self,
        position: np.ndarray,
        pressure_coefficients: PressureCoefficients,
        pressure_maximum: float = 40.0,
    ):
        """__init__ method for BaseFREE class.

        Parameters
        ----------
        position : np.ndarray
            2D (3, n_element) array array containing data with 'float' type.
            Array containing material frame position vectors.
        pressure_coefficients : PressureCoefficients
            Dataclass containing pressure coefficients for force and couple.
        pressure_maximum : float, optional
            Maximum pressure value with unit [psi], by default 40.0.
        """
        super().__init__(n_elements=position.shape[1])
        self.position = position
        self.pressure_coefficients = pressure_coefficients
        self.pressure_coefficients.set_n_elements(self.n_elements)
        self.pressure_maximum = pressure_maximum
        self.__pressure: float = 0.0
        self.tangent = np.zeros((3, self.n_elements))
        self.internal_force_value = self.pressure_coefficients.get_force_value(
            self.__pressure
        )
        self.internal_couple_value = (
            self.pressure_coefficients.get_couple_value(self.__pressure)
        )

    def __call__(self, system: ea.CosseratRod) -> None:
        self.tangent[:, :] = compute_local_tangent(
            compute_local_shear(
                self.position,
                sigma_to_shear(system.sigma),
                system.kappa,
                pointwise_multiplication(
                    system.rest_voronoi_lengths, system.voronoi_dilatation
                ),
            )
        )
        self.internal_force_value[:] = (
            self.pressure_coefficients.get_force_value(self.__pressure)
        )
        self.internal_couple_value[:] = (
            self.pressure_coefficients.get_couple_value(self.__pressure)
        )
        self.compute_internal_load(
            self.position,
            self.tangent,
            self.internal_force_value,
            self.internal_couple_value,
            self.internal_force,
            self.internal_couple,
        )
        super().__call__(system)

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_internal_load(
        position: np.ndarray,
        tangent: np.ndarray,
        internal_force_value: np.ndarray,
        internal_couple_value: np.ndarray,
        internal_force: np.ndarray,
        internal_couple: np.ndarray,
    ) -> None:
        blocksize = tangent.shape[1]
        temp_internal_couple = np.zeros((3, blocksize))
        for i in range(blocksize):
            internal_force[:, i] = internal_force_value[i] * tangent[:, i]
            temp_internal_couple[:, i] = (
                internal_couple_value[i] * tangent[:, i]
            )
        temp_internal_couple[:, :] += force_induced_couple(
            position, internal_force
        )
        internal_couple[:, :] = average2D(temp_internal_couple)

    @property
    def pressure(self) -> float:
        return self.__pressure

    @pressure.setter
    def pressure(self, pressure: float) -> None:
        self.__pressure = max(0.0, min(pressure, self.pressure_maximum))


class ApplyFREEs(ApplyActuations):
    def __init__(self, actuator_FREEs: Iterable[BaseFREE]):
        super().__init__(actuator_FREEs)
