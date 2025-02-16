"""
s15_postprocess_3d_shaft.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that imports data related to the 3D shaft and runs neubernet on it
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

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
#     }
# )

### USER INPUTS ##
target_variable = 4
von_mises_stress = False
save_plot = True
labels = False
### END USER INPUTS ###

eps = 1e-3  # Tolerance for node locations
tol = 0.05  # Plotting tol, to avoid cropping the boundary lines

sys.path.append("../model")

with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

input_var_dict = config["input_var_dict"]
target_var_dict = config["target_var_dict"]
analysis_data_dict = config["analysis_data_dict"]
RL = config["RL"]
alpha_shape = config["alpha_shape"]

base_folder = "../fem/shaft_3d"
models_path = "../model/trained"

# Load data from Validation_Shaft_Input_3D.txt
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
# theta = circ_data[:, 3] / (RL * R_notch) - np.pi  # Apparently more precise?
ux = circ_data[:, 4]  # We extract ux from ezz
uy = circ_data[:, 5]
roty = -circ_data[:, 6] / x
ezz = circ_data[:, 7]
ux_derived = ezz * x

# Now, set theta to be in the range [-np.pi, np.pi]
theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
theta = np.where(theta < -np.pi, theta + 2 * np.pi, theta)

# Sort data by theta
sorting_indices = np.argsort(theta)
theta = theta[sorting_indices]
ux = ux[sorting_indices]
uy = uy[sorting_indices]
roty = roty[sorting_indices]
ezz = ezz[sorting_indices]
ux_derived = ux_derived[sorting_indices]

# Get points where ux, uy, and roty are not all zero
nonzero_indices = np.where(np.logical_or(np.logical_or(ux != 0, uy != 0), roty != 0))[0]

# Exclude all elements outside the nonzero_indices
x = x[nonzero_indices]
theta = theta[nonzero_indices]
ux = ux[nonzero_indices]
uy = uy[nonzero_indices]
roty = roty[nonzero_indices]
ezz = ezz[nonzero_indices]
ux_derived = ux_derived[nonzero_indices]

theta_deg = np.degrees(theta)

# Interpolate the displacements and rotations at equispaced angles (10 degrees)
angles = np.arange(-180, 180, 10)
x_interp = np.interp(angles, theta_deg, x)
ux_interp = np.interp(angles, theta_deg, ux)
uy_interp = np.interp(angles, theta_deg, uy)
roty_interp = np.interp(angles, theta_deg, roty)
ezz_interp = np.interp(angles, theta_deg, ezz)
ux_derived_interp = np.interp(angles, theta_deg, ux_derived)

# Compute the average of uy_interp and roty_interp by integrating them over angles
ux_derived_avg = np.trapezoid(ux_derived_interp, angles) / (angles[-1] - angles[0])
ux_avg = np.trapezoid(ux_interp, angles) / (angles[-1] - angles[0])
uy_avg = np.trapezoid(uy_interp, angles) / (angles[-1] - angles[0])
roty_avg = np.trapezoid(roty_interp, angles) / (angles[-1] - angles[0])

# Which ux implementation?
# ux_interp = ux_interp
ux_interp = ux_interp - ux_avg + ux_derived_avg
# ux_interp = ezz_interp * x_interp
# ux_interp = ux_derived_interp

# Subtract the average from uy_interp and roty_interp
uy_interp -= uy_avg
roty_interp -= roty_avg

# UX-UY formulation (UR-UT formulation has been discarded)
# Transform the displacements and rotations into local (UX, UY) displacements
ux_local_interp = ux_interp * np.cos(np.radians(beta)) + uy_interp * np.sin(
    np.radians(beta)
)
uy_local_interp = -ux_interp * np.sin(np.radians(beta)) + uy_interp * np.cos(
    np.radians(beta)
)

interpolated_data = np.zeros((len(angles), 4))
interpolated_data[:, 0] = angles
interpolated_data[:, 1] = ux_local_interp / R_notch
interpolated_data[:, 2] = uy_local_interp / R_notch
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
results_data_en_diff = results_data_en_diff[mesh_data[:, 0] <= RL + eps]
results_data_stress_diff = results_data_stress_diff[mesh_data[:, 0] <= RL + eps]
results_data_strain_diff = results_data_strain_diff[mesh_data[:, 0] <= RL + eps]
mesh_data = mesh_data[mesh_data[:, 0] <= RL + eps]

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
)

# Broadcast the analysis_data to the same shape as mesh_result_pair and prepend it to the mesh_result_pair
analysis_data_broadcasted = np.repeat(analysis_data, len(mesh_data), axis=0)
inputs = np.concatenate((analysis_data_broadcasted, mesh_data), axis=1)

# Convert inputs and targets to torch tensors
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

# Load the model
model = torch.load(os.path.join(models_path, "neubernet.pt"), weights_only=False).cpu()

# Run the model
model.eval()
with torch.no_grad():
    predicted_targets_tensor = model(inputs_tensor)

predicted_targets = predicted_targets_tensor.numpy()

# Rotate the stresses (positions 2 to 7) and the strains (positions 8 to 13) by -beta, in Voigt notation
a_mat = np.array(
    [
        [np.cos(beta_radians), -np.sin(beta_radians), 0],
        [0, 0, -1],
        [np.sin(beta_radians), np.cos(beta_radians), 0],
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

if von_mises_stress:
    # Compute the von Mises stress
    predicted_targets = np.sqrt(
        0.5
        * (
            (predicted_targets[:, 2] - predicted_targets[:, 3]) ** 2
            + (predicted_targets[:, 3] - predicted_targets[:, 4]) ** 2
            + (predicted_targets[:, 4] - predicted_targets[:, 2]) ** 2
            + 6
            * (
                predicted_targets[:, 5] ** 2
                + predicted_targets[:, 6] ** 2
                + predicted_targets[:, 7] ** 2
            )
        )
    )

    true_targets = np.sqrt(
        0.5
        * (
            (true_targets[:, 2] - true_targets[:, 3]) ** 2
            + (true_targets[:, 3] - true_targets[:, 4]) ** 2
            + (true_targets[:, 4] - true_targets[:, 2]) ** 2
            + 6
            * (
                true_targets[:, 5] ** 2
                + true_targets[:, 6] ** 2
                + true_targets[:, 7] ** 2
            )
        )
    )

else:
    predicted_targets = predicted_targets[..., target_variable]
    true_targets = true_targets[..., target_variable]

if target_variable == 1:
    predicted_targets = np.abs(predicted_targets)

coordinates = inputs[:, -2:]
errors = predicted_targets - true_targets
vmin, vmax = min(true_targets.min(), predicted_targets.min()), max(
    true_targets.max(), predicted_targets.max()
)
# vmin = -vmax
error_abs = np.abs(errors).max()

# Find in input_var_dict the key for which the value has "x /" and "y /" in it
X_key = [key for key, value in input_var_dict.items() if "x /" in value][0]
Y_key = [key for key, value in input_var_dict.items() if "y /" in value][0]

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
fig, ax = plt.subplots(1, 3, figsize=(4.5, 1.5))

# Define colormap
cmap = "viridis"

# Plot the true values
contourf1 = ax[0].tricontourf(
    triang_masked, true_targets, levels=np.linspace(vmin, vmax, 20 + 2), cmap=cmap
)
ax[0].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
if labels:
    if von_mises_stress:
        ax[0].set_title(r"$\sigma_\mathrm{vm}  / \sigma_\mathrm{y}$ Ground Truth")
    else:
        ax[0].set_title(target_var_dict[target_variable] + " Ground Truth")
    ax[0].set_xlabel(input_var_dict[X_key])
    ax[0].set_ylabel(input_var_dict[Y_key])
ax[0].set_axis_off()
ax[0].set_xlim(-RL - tol, RL + tol)
ax[0].set_ylim(-RL - tol, RL + tol)
ax[0].set_aspect("equal", "box")

# Plot the predicted values
contourf2 = ax[1].tricontourf(
    triang_masked, predicted_targets, levels=np.linspace(vmin, vmax, 20 + 2), cmap=cmap
)
ax[1].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
if labels:
    if von_mises_stress:
        ax[1].set_title(r"$\sigma_\mathrm{vm} / \sigma_\mathrm{y}$ Predicted")
    else:
        ax[1].set_title(target_var_dict[target_variable] + " Predicted")
    ax[1].set_xlabel(input_var_dict[X_key])
ax[1].set_axis_off()
ax[1].set_xlim(-RL - tol, RL + tol)
ax[1].set_ylim(-RL - tol, RL + tol)
ax[1].set_aspect("equal", "box")

# Plot the error
contourf3 = ax[2].tricontourf(
    triang_masked,
    errors,
    levels=np.linspace(-error_abs, error_abs, 20 + 2),
    vmin=-error_abs,
    vmax=error_abs,
    cmap="coolwarm",
)
ax[2].plot(convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5)
if labels:
    if von_mises_stress:
        ax[2].set_title(r"$\sigma_\mathrm{vm} / \sigma_\mathrm{y}$ Error")
    else:
        ax[2].set_title(target_var_dict[target_variable] + " Error")
    ax[2].set_xlabel(input_var_dict[X_key])
ax[2].set_axis_off()
ax[2].set_xlim(-RL - tol, RL + tol)
ax[2].set_ylim(-RL - tol, RL + tol)
ax[2].set_aspect("equal", "box")

# Adjust colorbars
cbar1 = fig.colorbar(
    contourf1,
    ax=[ax[0], ax[1]],
    fraction=0.021,
    pad=0.04,
    format="%.2f",
    location="left",
)
cbar1.ax.tick_params(labelsize=8)
cbar1.set_ticks(np.linspace(vmin, vmax, 3))

cbar2 = fig.colorbar(contourf3, ax=ax[2], fraction=0.046, pad=0.04, format="%.2e")
cbar2.ax.tick_params(labelsize=8)
cbar2.set_ticks(np.linspace(-error_abs, error_abs, 3))

# fig.tight_layout()
# fig.subplots_adjust(bottom=0.2)
# fig.patch.set_alpha(0)
plt.show()

if save_plot:
    # Save the figure
    fig.savefig(f"Shaft_3D_plot_target_{target_variable}.pdf")
    fig.savefig(f"Shaft_3D_plot_target_{target_variable}.svg")
