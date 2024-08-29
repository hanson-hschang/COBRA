"""
Created on Jun 01, 2024
@author: Heng-Sheng (Hanson) Chang
"""

from typing import Self

from abc import ABC, abstractmethod

import elastica as ea
import numpy as np
from callbacks import BR2Property, RodCallBack
from packaging.version import Version
from tqdm import tqdm

from cobra.actuations.FREE import ApplyFREEs, BaseFREE, PressureCoefficients

BSR_AVAILABLE = True
try:
    import bsr
    from callbacks import BlenderBR2CallBack

    if Version(bsr.version) < Version("0.1.1"):
        raise ImportError("BSR version should be at least 0.1.1")
except ImportError:
    BSR_AVAILABLE = False


class Axis:
    def __init__(self, vector: list[float] | np.ndarray):
        if isinstance(vector, list):
            vector = np.array(vector)
        vector = vector / np.linalg.norm(vector)
        self.__vector = vector

    def rotate(self, angle: float, axis: np.ndarray | Self) -> Self:
        """
        Rotate itself around an axis by a given angle using Rodrigues' rotation formula.

        Parameters:
        angle (float): The angle of rotation in radians
        axis (np.array): The unit vector representing the axis of rotation

        Returns:
        Axis: The rotated axis
        """
        if isinstance(axis, Axis):
            axis = axis.to_numpy()
        # Ensure the axis is a unit vector
        axis = axis / np.linalg.norm(axis)

        # Rodrigues' rotation formula
        rotated_vector = (
            self.__vector * np.cos(angle)
            + np.cross(axis, self.__vector) * np.sin(angle)
            + axis * np.dot(axis, self.__vector) * (1 - np.cos(angle))
        )
        return Axis(rotated_vector)

    def to_numpy(self) -> np.ndarray:
        return self.__vector


class BaseSimulator(
    ea.BaseSystemCollection,
    ea.Damping,
    ea.CallBacks,
    ea.Constraints,
    ea.Forcing,
):
    pass


class BaseEnvironment(ABC):
    def __init__(
        self,
        final_time: float,
        time_step: float = 1.0e-5,
        recording_fps: int = 30,
    ) -> None:
        self.StatefulStepper = ea.PositionVerlet()  # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (self.recording_fps * self.time_step))
        self.reset()

    def reset(
        self,
    ) -> None:
        # Initialize the simulator
        self.simulator = BaseSimulator()

        self.setup()

        # Finalize the simulator and create time stepper
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

    def step(self, time: float) -> float:
        # Run the simulation for one step
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        # Return current simulation time
        return time

    @abstractmethod
    def setup(
        self,
    ) -> None:
        pass


class BR2Environment(BaseEnvironment):
    def __init__(self, *args, **kwargs) -> None:
        if BSR_AVAILABLE:
            bsr.clear_mesh_objects()
        super().__init__(*args, **kwargs)

    def setup(
        self,
    ) -> None:
        self.setup_BR2()

    def setup_BR2(
        self,
    ) -> None:

        # BR2 arm parameters
        bending_actuation_direction: Axis = Axis([-1.0, 0.0, 0.0])

        direction: Axis = Axis(
            [0.0, 0.0, -1.0]
        )  # direction of the BR2 arm (z-axis pointing down)
        normal: Axis = Axis(
            -bending_actuation_direction.to_numpy()
        )  # negative of bending FREE direction of the BR2 arm
        n_elements = 100  # number of discretized elements of the BR2 arm
        rest_length = 0.18  # rest length of the BR2 arm (used to be 0.16)
        rest_radius = 0.015  # rest radius of the BR2 arm
        thickness = 0.002  # thickness of the BR2 arm
        density = 700  # density of the BR2 arm
        youngs_modulus = 3 * 1e6  # Young's modulus of the BR2 arm
        poisson_ratio = 0.5  # Poisson's ratio of the BR2 arm
        damping_constant = 0.05  # damping constant of the BR2 arm

        # Adjust for the hollow rod
        FREE_radius_ratio = np.sqrt(3) / (2 + np.sqrt(3))
        FREE_outer_radius = FREE_radius_ratio * rest_radius
        FREE_inner_radius = FREE_outer_radius - thickness

        FREE_cross_section_area = np.pi * (
            FREE_outer_radius**2 - FREE_inner_radius**2
        )
        equivalent_ratio = np.sqrt(
            FREE_cross_section_area / (np.pi * rest_radius**2)
        )

        # Setup a rod
        self.rod = ea.CosseratRod.straight_rod(
            n_elements=n_elements,
            start=np.zeros((3,)),
            direction=direction.to_numpy(),
            normal=normal.to_numpy(),
            base_length=rest_length,
            base_radius=equivalent_ratio * rest_radius * np.ones(n_elements),
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
        )

        # Adjust for the inextensibility and unshearability
        # self.rod.shear_matrix = 100 * self.rod.shear_matrix

        self.simulator.append(self.rod)

        # Setup viscous damping
        self.simulator.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        # Setup rod callback
        self.rod_callback_params = ea.defaultdict(list)
        self.simulator.collect_diagnostics(self.rod).using(
            RodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_callback_params,
        )

        # Setup boundary conditions
        self.simulator.constrain(self.rod).using(
            ea.OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        # Setup gravity force
        acc_gravity = np.array([0.0, 0.0, -9.80665])
        self.simulator.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=acc_gravity
        )

        # Setup the BR2 arm presure actuation model
        offset_position_ratio = 2 / (2 + np.sqrt(3))

        rotation_CW_actuation_rotate_angle = 120 / 180 * np.pi
        rotation_CW_actuation_direction = bending_actuation_direction.rotate(
            angle=rotation_CW_actuation_rotate_angle,
            axis=direction,
        )

        rotation_CCW_actuation_rotate_angle = -120 / 180 * np.pi
        rotation_CCW_actuation_direction = bending_actuation_direction.rotate(
            angle=rotation_CCW_actuation_rotate_angle,
            axis=direction,
        )

        br2_property = BR2Property(
            radii=(FREE_radius_ratio * rest_radius * np.ones(n_elements - 1)),
            bending_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * bending_actuation_direction.to_numpy(),
                (n_elements, 1),
            ).T,
            rotation_CW_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * rotation_CW_actuation_direction.to_numpy(),
                (n_elements, 1),
            ).T,
            rotation_CCW_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * rotation_CCW_actuation_direction.to_numpy(),
                (n_elements, 1),
            ).T,
        )

        bending_actuation_force_coefficients = np.array([-0.1, 0.0])
        rotation_CW_actuation_couple_coefficients = np.array([0.0005, 0.0])
        rotation_CCW_actuation_couple_coefficients = np.array([-0.0005, 0.0])

        self.bending_actuation = BaseFREE(
            position=br2_property.bending_actuation_position,
            pressure_coefficients=PressureCoefficients(
                force=bending_actuation_force_coefficients,
                couple=np.array([0.0, 0.0, 0.0]),
            ),
        )
        self.rotation_CW_actuation = BaseFREE(
            position=br2_property.rotation_CW_actuation_position,
            pressure_coefficients=PressureCoefficients(
                force=np.array([0.0, 0.0, 0.0]),
                couple=rotation_CW_actuation_couple_coefficients,
            ),
        )
        self.rotation_CCW_actuation = BaseFREE(
            position=br2_property.rotation_CCW_actuation_position,
            pressure_coefficients=PressureCoefficients(
                force=np.array([0.0, 0.0, 0.0]),
                couple=rotation_CCW_actuation_couple_coefficients,
            ),
        )
        self.simulator.add_forcing_to(self.rod).using(
            ApplyFREEs,
            actuator_FREEs=[
                self.bending_actuation,
                self.rotation_CW_actuation,
                self.rotation_CCW_actuation,
            ],
        )

        if BSR_AVAILABLE:
            # Setup blender rod callback
            self.simulator.collect_diagnostics(self.rod).using(
                BlenderBR2CallBack,
                step_skip=self.step_skip,
                property=br2_property,
                system=self.rod,
            )

    def step(self, time: float, pressures: np.ndarray = np.zeros(3)) -> float:
        # Apply pressures to the BR2 arm
        self.bending_actuation.pressure = pressures[0]
        self.rotation_CW_actuation.pressure = pressures[1]
        self.rotation_CCW_actuation.pressure = pressures[2]

        return super().step(time)

    def save(self, filename: str) -> None:
        while filename.endswith(".npz") or filename.endswith(".blend"):
            if filename.endswith(".npz"):
                filename = filename[:-4]
            if filename.endswith(".blend"):
                filename = filename[:-6]

        # Save as .npz file
        np.savez(filename + ".npz", **self.rod_callback_params)

        if BSR_AVAILABLE:
            # Set the final keyframe number
            bsr.frame_manager.set_frame_end()

            # Save as .blend file
            bsr.save(filename + ".blend")


def main(
    final_time: float = 5.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 30,
):
    # Initialize the environment
    env = BR2Environment(
        final_time=final_time,
        time_step=time_step,
        recording_fps=recording_fps,
    )

    # Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for step in tqdm(range(env.total_steps)):
        time = env.step(
            time=time, pressures=np.array([30 * time, 30 * time, 0.0])
        )
    print("Simulation finished!")

    if BSR_AVAILABLE:
        # Set the frame rate
        bsr.frame_manager.set_frame_rate(fps=recording_fps)

        # Set the view distance
        bsr.set_view_distance(distance=0.5)

        # Deslect all objects
        bsr.deselect_all()

        # Select the camera object
        bsr.select_camera()

    # Save the simulation
    env.save("BR2_simulation")


if __name__ == "__main__":
    main()
