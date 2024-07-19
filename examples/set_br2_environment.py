"""
Created on Jun 01, 2024
@author: Heng-Sheng (Hanson) Chang
"""

from abc import ABC, abstractmethod

import elastica as ea
import numpy as np
from callbacks import RodCallBack
from tqdm import tqdm

from cobra.actuations.FREE import ApplyFREEs, BaseFREE, PressureCoefficients


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
    def setup(
        self,
    ) -> None:
        self.setup_BR2()

    def setup_BR2(
        self,
    ) -> None:
        # BR2 arm parameters
        direction = np.array(
            [0.0, 0.0, -1.0]
        )  # direction of the BR2 arm (z-axis pointing down)
        normal = np.array(
            [1.0, 0.0, 0.0]
        )  # bending FREE direction of the BR2 arm (x-axis pointing forward)
        n_elements = 100  # number of discretized elements of the BR2 arm
        rest_length = 0.16  # rest length of the BR2 arm
        rest_radius = 0.0075  # rest radius of the BR2 arm
        density = 700  # density of the BR2 arm
        youngs_modulus = 1e7  # Young's modulus of the BR2 arm
        poisson_ratio = 0.5  # Poisson's ratio of the BR2 arm
        damping_constant = 0.05  # damping constant of the BR2 arm

        # Setup a rod
        self.rod = ea.CosseratRod.straight_rod(
            n_elements=n_elements,
            start=np.zeros((3,)),
            direction=direction,
            normal=normal,
            base_length=rest_length,
            base_radius=rest_radius * np.ones(n_elements),
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
        )
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
        # TODO: set the right position and pressure coefficients for the BR2 actuations
        self.bending_actuation = BaseFREE(
            position=np.zeros((3, n_elements)),
            pressure_coefficients=PressureCoefficients(
                force=np.array([0.0, 0.0, 0.0]),
                couple=np.array([0.0, 0.0, 0.0]),
            ),
        )
        self.rotation_CW_actuation = BaseFREE(
            position=np.zeros((3, n_elements)),
            pressure_coefficients=PressureCoefficients(
                force=np.array([0.0, 0.0, 0.0]),
                couple=np.array([0.0, 0.0, 0.0]),
            ),
        )
        self.rotation_CCW_actuation = BaseFREE(
            position=np.zeros((3, n_elements)),
            pressure_coefficients=PressureCoefficients(
                force=np.array([0.0, 0.0, 0.0]),
                couple=np.array([0.0, 0.0, 0.0]),
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

    def step(self, time: float, pressures: np.ndarray = np.zeros(3)):
        # Apply pressures to the BR2 arm
        self.bending_actuation.pressure = pressures[0]
        self.rotation_CW_actuation.pressure = pressures[1]
        self.rotation_CCW_actuation.pressure = pressures[2]

        return super().step(time)


def main(
    final_time: float = 1.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 30,
):
    # Initialize the environment
    env = BR2Environment(
        final_time=final_time, time_step=time_step, recording_fps=recording_fps
    )

    # Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for step in tqdm(range(env.total_steps)):
        time = env.step(time)
    print("Simulation finished!")


if __name__ == "__main__":
    main()
