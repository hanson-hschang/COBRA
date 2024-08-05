"""
Created on Aug 03, 2024
@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from set_br2_environment import BR2Environment
from tqdm import tqdm

BSR_AVAILABLE = True
try:
    import bsr
except ImportError:
    BSR_AVAILABLE = False


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
        time = env.step(time=time, pressures=np.array([40 * time, 0.0, 0.0]))
    print("Simulation finished!")

    if BSR_AVAILABLE:
        bsr.set_view_distance(distance=0.5)

    # Save the simulation
    env.save("BR2_simulation")


if __name__ == "__main__":
    main()
