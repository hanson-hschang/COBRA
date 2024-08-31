import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import forward_path, sigma_to_shear

color = ["C" + str(i) for i in range(10)]

folder = "Data/"
idx = 10
file_name = "BR2_simulation%02d" % (idx)  # 'BR2_simulation' #
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

fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
for i in range(len(t)):
    ax.plot(position[i, 0, :], position[i, 1, :], position[i, 2, :])
    # ax.set_xlim(-L,0)
    # ax.set_ylim(-L,0)
    # ax.set_zlim(-L,0)
    ax.set_aspect("equal")

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
plt.show()
