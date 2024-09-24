"""
Created on Jun 01, 2024
@author: Heng-Sheng (Hanson) Chang
"""

import sys

import numpy as np
from set_br2_environment import BR2Environment
from tqdm import tqdm

BSR_AVAILABLE = True
try:
    import bsr
except ImportError:
    BSR_AVAILABLE = False


def main(
    idx,
    final_time: float = 3.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 30,
):
    # Initialize the environment
    env = BR2Environment(
        final_time=final_time, time_step=time_step, recording_fps=recording_fps
    )

    idx = int(idx)
    bend_max = 30
    twist_max = 25
    factor = 5
    bends = np.arange(0, bend_max + factor, factor)
    twists = np.arange(0, twist_max + factor, factor)
    bend_twist_pair = np.hstack(
        [
            np.vstack([np.ones(len(twists)) * bend_max, twists]),
            np.vstack([bends, np.ones(len(bends)) * twist_max]),
        ]
    )[:, :-1]
    bend = bend_twist_pair[0, idx]
    CWtwist = bend_twist_pair[1, idx]
    print("bend:", bend, "twist:", CWtwist)
    # Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for step in tqdm(range(env.total_steps)):
        bending = min(bend * time, bend)
        CWtwisting = min(CWtwist * time, CWtwist)
        time = env.step(
            time=time,
            pressures=np.array(
                [bending, 0.0, CWtwisting]
            ),  # [bending, CWtwisting, 0.0]
        )
        # if (step+1) % 10000 == 0:
        #     print(np.linalg.norm(env.rod.velocity_collection))
    print("Simulation finished!")

    # Save the simulation
    import os

    folder_name = "dataset/Data"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    print("saving data...")
    env.save(
        folder_name + "/BR2_simulation%02d" % (idx)
    )  # + bend_twist_pair.shape[1]


if __name__ == "__main__":
    # main()
    main(sys.argv[1])
