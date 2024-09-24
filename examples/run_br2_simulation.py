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
    period = [2.2, 3.75, 3.75, 3.75]
    if time < period[0]:
        pressures = np.array([30 * time / period[0], 0.0, 0.0])
    elif time < np.sum(period[:2]):
        pressures = np.array([30.0, 30 * (time - period[0]) / period[1], 0.0])
    elif time < np.sum(period[:3]):
        pressures = np.array(
            [
                30.0 * (period[0] + period[1] + period[2] - time) / period[2],
                30.0,
                0.0,
            ]
        )
    else:
        pressures = np.array(
            [0.0, 30.0 * (np.sum(period) - time) / period[3], 0.0]
        )
    return pressures


def main(
    final_time: float = 15.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 60,
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
