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


def pressure_profile_0(time):
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


def pressure_profile_1(time):
    period = [3.0, 3.0, 2.0, 2.0]
    if time < period[0]:
        pressures = np.array(
            [30 * time / period[0], 0.0, 30 * time / period[0]]
        )
    elif time < np.sum(period[:2]):
        pressures = np.array(
            [
                30.0,
                30 * (time - period[0] - period[1] / 2) / (period[1] / 2),
                30 * (period[0] + period[1] / 2 - time) / (period[1] / 2),
            ]
        )
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


def pressure_profile_2(time):
    period = [2.0, 3.0, 2.0, 3.0]
    if time < period[0]:
        pressures = np.array(
            [30 * time / period[0], 10 * time / period[0], 0.0]
        )
    elif time < np.sum(period[:2]):
        pressures = np.array(
            [
                30.0 * (period[0] + period[1] - time) / period[1],
                10 * (period[0] + period[1] / 4 - time) / (period[1] / 4),
                30 * (time - period[0] - period[1] / 4) / (period[1] / 4 * 3),
            ]
        )
    elif time < np.sum(period[:3]):
        pressures = np.array(
            [
                30.0 * (time - period[0] - period[1]) / period[2],
                0.0,
                30.0,
            ]
        )
    else:
        pressures = np.array(
            [
                30.0 * (np.sum(period) - time) / period[3],
                0.0,
                30.0 * (np.sum(period) - time) / period[3],
            ]
        )
    return pressures


def pressure_profile_3(time):
    period = [2.0, 2.0, 3.0, 3.0]
    if time < period[0]:
        pressures = np.array(
            [
                0.0,
                30.0 * time / period[0],
                0.0,
            ]
        )
    elif time < np.sum(period[:2]):
        pressures = np.array(
            [
                15.0 * (time - period[0]) / period[1],
                30.0,
                0.0,
            ]
        )
    elif time < np.sum(period[:3]):
        pressures = np.array(
            [
                15.0,
                30.0
                * (period[0] + period[1] + period[2] / 3 * 2 - time)
                / (period[2] / 3 * 2),
                15.0
                * (time - period[0] - period[1] - period[2] / 3 * 2)
                / (period[2] / 3),
            ]
        )
    else:
        pressures = np.array(
            [
                15.0 * (np.sum(period) - time) / period[3],
                0.0,
                15.0 * (np.sum(period) - time) / period[3],
            ]
        )
    return pressures


def main(
    final_time: float = 10.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 60,
):
    # Initialize the environment
    env = BR2Environment(
        final_time=final_time,
        time_step=time_step,
        recording_fps=recording_fps,
    )

    if BSR_AVAILABLE:
        camera_direction_angle = 0 / 180 * np.pi
        camera_distance = 0.9
        camera_height = 0.5
        camera_direction = np.array(
            [
                np.cos(camera_direction_angle),
                np.sin(camera_direction_angle),
                0.0,
            ]
        )

        bsr.camera.location = camera_distance * camera_direction + np.array(
            [0.0, 0.0, camera_height]
        )
        bsr.camera.look_at = np.array([0.0, 0.0, -0.15])
        bsr.camera.set_film_transparent()
        bsr.camera.set_resolution(1920, 1080)
        bsr.camera.set_file_path(
            file_name="br2",
            folder_path="video/frames",
        )

        # Set the current frame number
        bsr.frame_manager.frame_current = 0

        # Set the initial keyframe number
        bsr.frame_manager.set_frame_start()

    # Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for step in tqdm(range(env.total_steps)):

        pressures = pressure_profile_3(time)

        time = env.step(
            time=time,
            pressures=pressures,
        )
    print("Simulation finished!")

    if BSR_AVAILABLE:
        # Set the frame end
        bsr.frame_manager.set_frame_end()

        # Set the frame rate
        bsr.frame_manager.set_frame_rate(fps=recording_fps)

        # Set the view distance
        bsr.set_view_distance(distance=0.5)

        # Deselect all objects
        bsr.deselect_all()

        # Select the camera object
        bsr.camera.select()

        # Render the camera frames
        bsr.camera.render(np.arange(bsr.frame_manager.frame_end))

    # Save the simulation
    env.save("BR2_simulation")


if __name__ == "__main__":
    main()

# ffmpeg -threads 8 -r 60 -i video/frames/br2_%03d.png -b:v 90M -c:v prores -pix_fmt yuva444p10le video/br2.mov
