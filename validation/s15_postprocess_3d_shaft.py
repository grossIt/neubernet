"""
s15_postprocess_3d_shaft.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Import data related to the 3d shaft, and runs neubernet on it
"""


import os
import numpy as np
import pandas as pd
import sys
import torch
import yaml

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay, ConvexHull
import alphashape
from shapely import Point, Polygon

from utils.helper_funcs import import_path_file, import_delimited_file

### USER INPUTS ##
target_variable = 2
save_plot = False
### END USER INPUTS ###

sys.path.append("../model")

with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

input_var_dict = config["input_var_dict"]
target_var_dict = config["target_var_dict"]
analysis_data_dict = config["analysis_data_dict"]
RL = config["RL"]
alpha_shape = config["alpha_shape"]

base_folder = "../fem/shaft_3d"
models_path = "../model"

# Load data from Validation_SHaft_Input_3D.txt
input_data = pd.read_csv(
    os.path.join(base_folder, "Validation_Shaft_Input_3D.txt"), sep=" ", header=None
)
# Remove all columns with NaN values
input_data = input_data.dropna(axis=1)
input_data = input_data.to_numpy()

R = input_data[0, 0]
Y_midnotch = input_data[0, 1]
R_notch = input_data[0, 2]
alpha = input_data[0, 3]
beta = input_data[0, 4]
ni = input_data[0, 5]
E = input_data[0, 6]
Sy = input_data[0, 7]
Et = input_data[0, 8]

beta_radians = np.radians(beta)

# Load data from Validation_Shaft_Circ_3D.txt
circ_data = import_path_file(
    os.path.join(base_folder, "Validation_Shaft_Circ_3D.txt"), expected_num_columns=8
)

x = circ_data[:, 0]
y = circ_data[:, 1]
theta = np.arctan2(y - Y_midnotch, x - R) - beta_radians
# theta = circ_data[:, 3] / (RL * R_notch)  # Apparently more precise
# ux = circ_data[:, 4]  # We extract ux from ezz
uy = circ_data[:, 5]
roty = -circ_data[:, 6] / x
ezz = circ_data[:, 7]
ux = ezz * x

# Now, set theta to be in the range [-np.pi, np.pi]
theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
theta = np.where(theta < -np.pi, theta + 2 * np.pi, theta)

# Sort data by theta
sorting_indices = np.argsort(theta)
theta = theta[sorting_indices]
ux = ux[sorting_indices]
uy = uy[sorting_indices]
roty = roty[sorting_indices]

# Get points where ux, uy, and roty are not all zero
nonzero_indices = np.where(np.logical_or(np.logical_or(ux != 0, uy != 0), roty != 0))[0]

# Exclude all elements outside the nonzero_indices
theta = theta[nonzero_indices]
ux = ux[nonzero_indices]
uy = uy[nonzero_indices]
roty = roty[nonzero_indices]
theta_deg = np.degrees(theta)

# Interpolate the displacements and rotations at equispaced angles (10 degrees)
angles = np.arange(-180, 180, 10)
ux_interp = np.interp(angles, theta_deg, ux)
uy_interp = np.interp(angles, theta_deg, uy)
roty_interp = np.interp(angles, theta_deg, roty)

# Compute the average of uy_interp and roty_interp by integrating them over angles
uy_avg = np.trapz(uy_interp, angles) / (angles[-1] - angles[0])
roty_avg = np.trapz(roty_interp, angles) / (angles[-1] - angles[0])

# Subtract the average from uy_interp and roty_interp
uy_interp -= uy_avg
roty_interp -= roty_avg

# Transform the displacements and rotations into (UR, tangential) displacements
ur_interp = ux_interp * np.cos(np.radians(angles + beta)) + uy_interp * np.sin(
    np.radians(angles + beta)
)
ut_interp = -ux_interp * np.sin(np.radians(angles + beta)) + uy_interp * np.cos(
    np.radians(angles + beta)
)

interpolated_data = np.zeros((len(angles), 4))
interpolated_data[:, 0] = angles
interpolated_data[:, 1] = ur_interp / R_notch
interpolated_data[:, 2] = ut_interp / R_notch
interpolated_data[:, 3] = roty_interp  # No need to divide by R_notch, it's an angle...

# Now, we can generate the baseline branch_data array
branch_data = np.array(
    [
        *interpolated_data[:, 1] * E / Sy,
        *interpolated_data[:, 2] * E / Sy,
        *interpolated_data[:, 3] * E / Sy,
    ]
)

parameter_data = np.array(
    [
        R / R_notch,
        alpha,
        beta,
        Sy / E,
        Et / E,
        ni,
    ]
)

# Concatenate the branch_data and parameter_data arrays and append to the analysis_data_list
analysis_data = np.expand_dims(
    np.concatenate((branch_data, parameter_data)).astype(np.float32), 0
)

# Import mesh data
mesh_data = import_delimited_file(
    os.path.join(base_folder, "Validation_Shaft_Mesh_3D.txt"), expected_num_columns=4
)

# Import results
results_data_en = import_delimited_file(
    os.path.join(base_folder, "Validation_Shaft_Send_3D.txt"),
    expected_num_columns=9,
)
results_data_stress = import_delimited_file(
    os.path.join(base_folder, "Validation_Shaft_Stress_3D.txt"),
    expected_num_columns=7,
)
results_data_strain = import_delimited_file(
    os.path.join(base_folder, "Validation_Shaft_ePlas_3D.txt"),
    expected_num_columns=7,
)

# Make sure that all the results files have the same number of rows and that the first column is the same
# If not, include only the rows so that the first column is the same
rows_set = (
    set(results_data_en[:, 0].astype(int))
    & set(results_data_stress[:, 0].astype(int))
    & set(results_data_strain[:, 0].astype(int))
)
rows_list = list(rows_set)  # Convert the sets to lists for indexing

# Filter the results data
results_data_en = results_data_en[np.isin(results_data_en[:, 0].astype(int), rows_list)]
results_data_stress = results_data_stress[
    np.isin(results_data_stress[:, 0].astype(int), rows_list)
]
results_data_strain = results_data_strain[
    np.isin(results_data_strain[:, 0].astype(int), rows_list)
]

# Filter mesh_data, extracting only rows whose first element appears in the first column of every result_data
mesh_data = mesh_data[
    np.logical_and.reduce(
        [
            np.isin(mesh_data[:, 0].astype(int), results_data_en[:, 0].astype(int)),
            np.isin(
                mesh_data[:, 0].astype(int),
                results_data_stress[:, 0].astype(int),
            ),
            np.isin(
                mesh_data[:, 0].astype(int),
                results_data_strain[:, 0].astype(int),
            ),
        ]
    )
]

# Extract only meaningful columns and normalize the results
mesh_data = mesh_data[:, 1:3]
mesh_data[:, 0] /= R_notch  # Normalize the radial coordinate
results_data_en_diff = (results_data_en[:, 1:3]) / Sy
results_data_stress_diff = (results_data_stress[:, 1:]) / Sy
results_data_strain_diff = results_data_strain[:, 1:]

# Extract only the nodes that have normalized R coordinate <= RL, both from mesh_data and results_data
results_data_en_diff = results_data_en_diff[mesh_data[:, 0] <= RL]
results_data_stress_diff = results_data_stress_diff[mesh_data[:, 0] <= RL]
results_data_strain_diff = results_data_strain_diff[mesh_data[:, 0] <= RL]
mesh_data = mesh_data[mesh_data[:, 0] <= RL]

# Transform mesh_data from polar to Cartesian coordinates (with same orientation as the notched circle)
mesh_data = np.array(
    [
        [
            r * np.cos(np.radians(theta)),
            r * np.sin(np.radians(theta)),
        ]
        for r, theta in mesh_data
    ]
)

# Concatenate mesh_data and results_data into a single array
true_targets = np.concatenate(
    (
        results_data_en_diff,
        results_data_stress_diff,
        results_data_strain_diff,
    ),
    axis=1,
)[..., target_variable]

# Broadcast the analysis_data to the same shape as mesh_result_pair and prepend it to the mesh_result_pair
analysis_data_broadcasted = np.repeat(analysis_data, len(mesh_data), axis=0)
inputs = np.concatenate((analysis_data_broadcasted, mesh_data), axis=1)

# Convert inputs and targets to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# Load the model
model = torch.load(os.path.join(models_path, "neubernet.pth")).cpu()

# Run the model
model.eval()
with torch.no_grad():
    predicted_targets_tensor = model(inputs_tensor)

predicted_targets = predicted_targets_tensor.numpy()

# Rotate the stresses (positions 2 to 7) and the strains (positions 8 to 13) by -beta, in Voigt notation
a_mat = np.array(
    [
        [np.cos(beta_radians), -np.sin(beta_radians), 0],
        [np.sin(beta_radians), np.cos(beta_radians), 0],
        [0, 0, 1],
    ]
)

rotation_voigt_stress = np.array(
    [
        [
            a_mat[0, 0] ** 2,
            a_mat[0, 1] ** 2,
            a_mat[0, 2] ** 2,
            2 * a_mat[0, 0] * a_mat[0, 1],
            2 * a_mat[0, 1] * a_mat[0, 2],
            2 * a_mat[0, 0] * a_mat[0, 2],
        ],
        [
            a_mat[1, 0] ** 2,
            a_mat[1, 1] ** 2,
            a_mat[1, 2] ** 2,
            2 * a_mat[1, 0] * a_mat[1, 1],
            2 * a_mat[1, 1] * a_mat[1, 2],
            2 * a_mat[1, 0] * a_mat[1, 2],
        ],
        [
            a_mat[2, 0] ** 2,
            a_mat[2, 1] ** 2,
            a_mat[2, 2] ** 2,
            2 * a_mat[2, 0] * a_mat[2, 1],
            2 * a_mat[2, 1] * a_mat[2, 2],
            2 * a_mat[2, 0] * a_mat[2, 2],
        ],
        [
            a_mat[0, 0] * a_mat[1, 0],
            a_mat[0, 1] * a_mat[1, 1],
            a_mat[0, 2] * a_mat[1, 2],
            a_mat[0, 0] * a_mat[1, 1] + a_mat[0, 1] * a_mat[1, 0],
            a_mat[0, 1] * a_mat[1, 2] + a_mat[0, 2] * a_mat[1, 1],
            a_mat[0, 0] * a_mat[1, 2] + a_mat[0, 2] * a_mat[1, 0],
        ],
        [
            a_mat[1, 0] * a_mat[2, 0],
            a_mat[1, 1] * a_mat[2, 1],
            a_mat[1, 2] * a_mat[2, 2],
            a_mat[1, 0] * a_mat[2, 1] + a_mat[1, 1] * a_mat[2, 0],
            a_mat[1, 1] * a_mat[2, 2] + a_mat[1, 2] * a_mat[2, 1],
            a_mat[1, 0] * a_mat[2, 2] + a_mat[1, 2] * a_mat[2, 0],
        ],
        [
            a_mat[0, 0] * a_mat[2, 0],
            a_mat[0, 1] * a_mat[2, 1],
            a_mat[0, 2] * a_mat[2, 2],
            a_mat[0, 0] * a_mat[2, 1] + a_mat[0, 1] * a_mat[2, 0],
            a_mat[0, 1] * a_mat[2, 2] + a_mat[0, 2] * a_mat[2, 1],
            a_mat[0, 0] * a_mat[2, 2] + a_mat[0, 2] * a_mat[2, 0],
        ],
    ]
)

rotation_voigt_strain = np.array(
    [
        [
            a_mat[0, 0] ** 2,
            a_mat[0, 1] ** 2,
            a_mat[0, 2] ** 2,
            a_mat[0, 0] * a_mat[0, 1],
            a_mat[0, 1] * a_mat[0, 2],
            a_mat[0, 0] * a_mat[0, 2],
        ],
        [
            a_mat[1, 0] ** 2,
            a_mat[1, 1] ** 2,
            a_mat[1, 2] ** 2,
            a_mat[1, 0] * a_mat[1, 1],
            a_mat[1, 1] * a_mat[1, 2],
            a_mat[1, 0] * a_mat[1, 2],
        ],
        [
            a_mat[2, 0] ** 2,
            a_mat[2, 1] ** 2,
            a_mat[2, 2] ** 2,
            a_mat[2, 0] * a_mat[2, 1],
            a_mat[2, 1] * a_mat[2, 2],
            a_mat[2, 0] * a_mat[2, 2],
        ],
        [
            2 * a_mat[0, 0] * a_mat[1, 0],
            2 * a_mat[0, 1] * a_mat[1, 1],
            2 * a_mat[0, 2] * a_mat[1, 2],
            a_mat[0, 0] * a_mat[1, 1] + a_mat[0, 1] * a_mat[1, 0],
            a_mat[0, 1] * a_mat[1, 2] + a_mat[0, 2] * a_mat[1, 1],
            a_mat[0, 0] * a_mat[1, 2] + a_mat[0, 2] * a_mat[1, 0],
        ],
        [
            2 * a_mat[1, 0] * a_mat[2, 0],
            2 * a_mat[1, 1] * a_mat[2, 1],
            2 * a_mat[1, 2] * a_mat[2, 2],
            a_mat[1, 0] * a_mat[2, 1] + a_mat[1, 1] * a_mat[2, 0],
            a_mat[1, 1] * a_mat[2, 2] + a_mat[1, 2] * a_mat[2, 1],
            a_mat[1, 0] * a_mat[2, 2] + a_mat[1, 2] * a_mat[2, 0],
        ],
        [
            2 * a_mat[0, 0] * a_mat[2, 0],
            2 * a_mat[0, 1] * a_mat[2, 1],
            2 * a_mat[0, 2] * a_mat[2, 2],
            a_mat[0, 0] * a_mat[2, 1] + a_mat[0, 1] * a_mat[2, 0],
            a_mat[0, 1] * a_mat[2, 2] + a_mat[0, 2] * a_mat[2, 1],
            a_mat[0, 0] * a_mat[2, 2] + a_mat[0, 2] * a_mat[2, 0],
        ],
    ]
)

predicted_targets[:, 2:8] = np.dot(predicted_targets[:, 2:8], rotation_voigt_stress.T)
predicted_targets[:, 8:14] = np.dot(predicted_targets[:, 8:14], rotation_voigt_strain.T)

predicted_targets = predicted_targets[..., target_variable]

if target_variable == 1:
    predicted_targets = np.abs(predicted_targets)

coordinates = inputs[:, -2:]
errors = predicted_targets - true_targets

# Find in input_var_dict the key for which the value has "X" and "Y" in it
X_key = [key for key, value in input_var_dict.items() if "X" in value][0]
Y_key = [key for key, value in input_var_dict.items() if "Y" in value][0]

# Rotate the coordinates by beta
rotation_matrix = np.array(
    [
        [np.cos(beta_radians), -np.sin(beta_radians)],
        [np.sin(beta_radians), np.cos(beta_radians)],
    ]
)
coordinates = np.dot(coordinates, rotation_matrix.T)

# Generate triangulation
triang = Delaunay(coordinates)
convex_hull = alphashape.alphashape(coordinates, alpha_shape)

# Mask out triangles outside the concave domain
mask = np.zeros(triang.simplices.shape[0], dtype=bool)
polygon = Polygon(convex_hull.exterior.coords)

for i, simplex in enumerate(triang.simplices):
    centroid = np.mean(coordinates[simplex], axis=0)
    if not polygon.contains(Point(centroid)):
        mask[i] = True

# Create a Triangulation object with the mask
triang_masked = tri.Triangulation(
    coordinates[:, 0], coordinates[:, 1], triang.simplices[~mask]
)

# Plot the results as tricontours
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Plot the true values
contourf1 = ax[0].tricontourf(triang_masked, true_targets, levels=20)
ax[0].set_title("True values")
ax[0].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
ax[0].set_title(target_var_dict[target_variable] + " Ground Truth")
ax[0].set_xlabel(input_var_dict[X_key])
ax[0].set_ylabel(input_var_dict[Y_key])
ax[0].set_xlim(-RL, RL)
ax[0].set_ylim(-RL, RL)
fig.colorbar(contourf1, ax=ax[0])

# Plot the predicted values
contourf2 = ax[1].tricontourf(triang_masked, predicted_targets, levels=20)
ax[1].set_title("Predicted values")
ax[1].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
ax[1].set_title(target_var_dict[target_variable] + " Predicted")
ax[1].set_xlabel(input_var_dict[X_key])
ax[1].set_ylabel(input_var_dict[Y_key])
ax[1].set_xlim(-RL, RL)
ax[1].set_ylim(-RL, RL)
fig.colorbar(contourf2, ax=ax[1])

# Plot the error
contourf3 = ax[2].tricontourf(triang_masked, errors, levels=20)
ax[2].set_title("Error")
ax[2].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
ax[2].set_title(target_var_dict[target_variable] + " Error")
ax[2].set_xlabel(input_var_dict[X_key])
ax[2].set_ylabel(input_var_dict[Y_key])
ax[2].set_xlim(-RL, RL)
ax[2].set_ylim(-RL, RL)
fig.colorbar(contourf3, ax=ax[2])

# Add the set of analysis input parameters as text
# Find the names in the analysis_data_dict from keys 108 to 113
analysis_data_names = [analysis_data_dict[key] for key in range(108, 114)]
analysis_data_values = analysis_data[0, 108:114]
analysis_data_text = ", ".join(
    [
        f"{name} = {value:.3g}"
        for name, value in zip(analysis_data_names, analysis_data_values)
    ]
)
fig.text(0.5, 0.03, analysis_data_text, ha="center", va="center")

fig.tight_layout()
fig.subplots_adjust(bottom=0.2)
fig.patch.set_alpha(0)
plt.show()

if save_plot:
    # Save the figure
    fig.savefig(f"Shaft_3D_plot_target_{target_variable}.pdf")
    fig.savefig(f"Shaft_3D_plot_target_{target_variable}.svg")