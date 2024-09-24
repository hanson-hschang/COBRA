from typing import Iterable, Protocol

from dataclasses import dataclass

import elastica as ea
import numpy as np
from numba import njit

from cobra.actuations import ApplyActuations, ContinuousActuation
from cobra.actuations.actuation_tool import force_induced_couple
from cobra.math_tool import average2D, pointwise_multiplication
from cobra.rod_geometry_tool import (
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


class RequiredMaxPressure(Protocol):
    """
    Protocol class for pressure_maximum attribute.
    """

    pressure_maximum: float


class Pressure:
    """
    Descriptor class for pressure attribute.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.private_name = "__" + name
        setattr(owner, self.private_name, 0.0)

    def __get__(self, instance: object, owner: type) -> float:
        value: float = getattr(instance, self.private_name)
        return value

    def __set__(self, instance: RequiredMaxPressure, value: float) -> None:
        value = max(0.0, min(value, instance.pressure_maximum))
        setattr(instance, self.private_name, value)


class BaseFREE(ContinuousActuation):
    """
    Base class for FREE actuation.

    Parameters
    ----------
    position : np.ndarray
        2D (3, n_element) array array containing data with 'float' type.
        Array containing material frame position vectors.
    pressure_coefficients : PressureCoefficients
        Dataclass containing pressure coefficients for force and couple.
    pressure_maximum : float, optional
        Maximum pressure value with unit [psi], by default 30.0.
    """

    pressure = Pressure()

    def __init__(
        self,
        position: np.ndarray,
        pressure_coefficients: PressureCoefficients,
        pressure_maximum: float = 30.0,
    ):
        super().__init__(n_elements=position.shape[1])
        self.position = position
        self.pressure_coefficients = pressure_coefficients
        self.pressure_coefficients.set_n_elements(self.n_elements)
        self.pressure_maximum = pressure_maximum
        self.tangent = np.zeros((3, self.n_elements))
        self.internal_force_value = self.pressure_coefficients.get_force_value(
            self.pressure
        )
        self.internal_couple_value = (
            self.pressure_coefficients.get_couple_value(self.pressure)
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
            self.pressure_coefficients.get_force_value(self.pressure)
        )
        self.internal_couple_value[:] = (
            self.pressure_coefficients.get_couple_value(self.pressure)
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


class ApplyFREEs(ApplyActuations):
    def __init__(self, actuator_FREEs: Iterable[BaseFREE]):
        super().__init__(actuator_FREEs)
