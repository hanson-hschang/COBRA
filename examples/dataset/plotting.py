import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import forward_path, sigma_to_shear

color = ["C" + str(i) for i in range(10)]

folder = "Data/"
idx = 12
file_name = "BR2_simulation%02d" % (idx)
data = np.load(folder + file_name + ".npz")

# print(data.files)
t = data["time"]
position = data["position"]
orientation = data["director"]
kappa = data["kappa"]
sigma = data["sigma"]
n_elem = orientation.shape[-1]
L = np.linalg.norm(position[0, :, -1])
s = np.linspace(0, L, n_elem + 1)
s_mean = 0.5 * (s[1:] + s[:-1])
# print(t.shape, position.shape, orientation.shape, kappa.shape, sigma.shape)
# print('\n', orientation[-1,...,0], '\n', orientation[-1,...,-1])
# print(position[-1,...,-1])

dl = np.linalg.norm(position[0, :, 1:] - position[0, :, :-1], axis=0)
pos_estimate = np.zeros_like(position)
orien_estimate = orientation.copy()
shear = sigma_to_shear(sigma)
for i in range(len(t)):
    forward_path(dl, shear[i], kappa[i], pos_estimate[i], orien_estimate[i])

plotting_flag = False
video_flag = True
video_save_flag = False

if plotting_flag:
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection="3d")
    for i in range(len(t)):
        ax1.plot(position[i, 0, :], position[i, 1, :], position[i, 2, :])
        # ax1.set_xlim(-L,0)
        # ax1.set_ylim(-L,0)
        # ax1.set_zlim(-L,0)
        ax1.set_aspect("equal")

    fig2, axes = plt.subplots(ncols=3, sharex=True, figsize=(16, 5))
    for i in range(len(t)):
        for j in range(3):
            axes[j].plot(s[1:-1], kappa[i, j, :])

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111, projection="3d")
    idx_list = np.arange(40)[::8]  # np.random.randint(len(t), size=5)
    for ii in range(len(idx_list)):
        i = idx_list[ii]
        ax3.plot(
            position[i, 0, :],
            position[i, 1, :],
            position[i, 2, :],
            ls=":",
            color="k",
        )  # color[ii])
        ax3.plot(
            pos_estimate[i, 0, :],
            pos_estimate[i, 1, :],
            pos_estimate[i, 2, :],
            ls="--",
            color=color[ii],
        )
        # ax3.set_xlim(-L,0)
        # ax3.set_ylim(-L,0)
        # ax3.set_zlim(-L,0)
        ax3.set_aspect("equal")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")

if video_flag:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fps = 50
    factor = 1
    video_name = "Videos/" + file_name + ".mov"
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(
        title="Movie Test", artist="Matplotlib", comment="Movie support!"
    )
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    with writer.saving(fig, video_name, 100):
        for jj in range(int((len(t) - 1) / factor + 1)):
            i = jj * factor
            time = i / (len(t) - 1) * t[-1]
            ax.cla()
            ax.plot(position[i, 0, :], position[i, 1, :], position[i, 2, :])
            ax.text(
                0.05,
                1.05,
                1.05,
                "t: %.3f s" % (time),
                transform=ax.transAxes,
                fontsize=15,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_zlim(-L, 0)
            ax.set_aspect("equal")
            if not video_save_flag:
                plt.pause(0.01)
            else:
                writer.grab_frame()


plt.show()
