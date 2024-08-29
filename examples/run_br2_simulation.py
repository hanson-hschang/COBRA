"""
Created on Aug 03, 2024
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from packaging.version import Version
from set_br2_environment import BR2Environment
from tqdm import tqdm

BSR_AVAILABLE = True
try:
    import bsr

    if Version(bsr.version) < Version("0.1.1"):
        raise ImportError("BSR version should be at least 0.1.1")
except ImportError:
    BSR_AVAILABLE = False


def pressure_profile(time):
    period = 15 / 4
    if time < period:
        pressures = np.array([30 * time / period, 0.0, 0.0])
    elif time < 2 * period:
        pressures = np.array([30.0, 30 * (time - period) / period, 0.0])
    elif time < 3 * period:
        pressures = np.array([30.0 * (3 * period - time) / period, 30.0, 0.0])
    else:
        pressures = np.array([0.0, 30.0 * (4 * period - time) / period, 0.0])
    return pressures


def main(
    final_time: float = 15.0,
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

        pressures = pressure_profile(time)

        time = env.step(
            time=time,
            pressures=pressures,
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
