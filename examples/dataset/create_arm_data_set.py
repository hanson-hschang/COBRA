"""
Created on Jul 18, 2024
@author: Tixian Wang
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import pos_dir_to_input, sigma_to_shear

color = ["C" + str(i) for i in range(10)]


folder_name = "Data/"

## simulation data
n_cases = 22
step_skip = 1

## data point setup
n_data_pts = 2  # exlude the initial point at base
idx_data_pts = np.array(
    [int(100 / (n_data_pts)) * i for i in range(1, n_data_pts)] + [-1]
)

# print(idx_data_pts)

input_data = []
true_pos = []
true_dir = []
true_kappa = []
true_shear = []

for i in tqdm(range(1, n_cases)):
    file_name = "BR2_simulation%02d" % (i)
    data = np.load(folder_name + file_name + ".npz")
    if i == 11:
        t = data["time"]
        position = data["position"]
        orientation = data["director"]
        kappa = data["kappa"]
        sigma = data["sigma"]
        n_elem = orientation.shape[-1]
        L = np.linalg.norm(position[0, :, -1])
        print("rest length:", L)
        s = np.linspace(0, L, n_elem + 1)
        s_mean = 0.5 * (s[1:] + s[:-1])
        radius = data["radius"]
        dl = np.linalg.norm(position[0, :, 1:] - position[0, :, :-1], axis=0)
        nominal_shear = sigma_to_shear(sigma[0])

    position = data["position"]
    orientation = data["director"]
    kappa = data["kappa"]
    sigma = data["sigma"]
    shear = sigma.copy()
    shear[:, 2, :] += 1

    input_pos = position[::step_skip, :, idx_data_pts]
    input_dir = orientation[::step_skip, ..., idx_data_pts]
    inputs = pos_dir_to_input(input_pos, input_dir)

    input_data.append(inputs)
    true_pos.append(position[::step_skip, ...])
    true_dir.append(orientation[::step_skip, ...])
    true_kappa.append(kappa[::step_skip, ...])
    true_shear.append(shear[::step_skip, ...])

input_data = np.vstack(input_data)
true_pos = np.vstack(true_pos)
true_dir = np.vstack(true_dir)
true_kappa = np.vstack(true_kappa)
true_shear = np.vstack(true_shear)
# print(input_data.shape, true_pos.shape, true_dir.shape, true_kappa.shape, true_shear.shape)

idx_list = np.random.randint(
    len(true_kappa), size=10
)  # [i*250 for i in range(10)]
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
fig2, axes = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 5))
for ii in range(len(idx_list)):
    i = idx_list[ii]
    ax.plot(
        true_pos[i, 0, :],
        true_pos[i, 1, :],
        true_pos[i, 2, :],
        ls="-",
        color=color[ii],
    )
    ax.scatter(
        input_data[i, 0, :],
        input_data[i, 1, :],
        input_data[i, 2, :],
        s=50,
        marker="o",
        color=color[ii],
    )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(-L, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    for j in range(3):
        axes[0][j].plot(s[1:-1], true_kappa[i, j, :])
        axes[1][j].plot(s_mean, true_shear[i, j, :])

# plt.show()
# quit()

model_data = {
    "n_elem": n_elem,
    "L": L,
    "radius": radius,
    "s": s,
    "dl": dl,
    "nominal_shear": nominal_shear,
}

data = {
    "model": model_data,
    "n_data_pts": n_data_pts,
    "idx_data_pts": idx_data_pts,
    "input_data": input_data,
    "true_pos": true_pos,
    "true_dir": true_dir,
    "true_kappa": true_kappa,
    "true_shear": true_shear,
}

arm_data_name = "BR2_arm_data.npy"  # 'octopus_arm_data.npy' #
np.save(folder_name + arm_data_name, data)

plt.show()
