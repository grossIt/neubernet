"""
s7_plot_analysis.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Plot a given analysis, comparing the true and predicted target values
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay, ConvexHull
import alphashape
from shapely import Point, Polygon

### USER INPUTS ##
analysis_ID = 9983
analysis_factor = 5
target_variable = 3
save_plot = False
freemesh_size = None
### END USER INPUTS ###

# Load configuration file
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the dictionaries
input_var_dict = config["input_var_dict"]
target_var_dict = config["target_var_dict"]
analysis_data_dict = config["analysis_data_dict"]
RL = config["RL"]
alpha_shape = config["alpha_shape"]

print("Analysis ID:", analysis_ID)
print("Analysis load step:", analysis_factor)
print("Target variable:", target_var_dict[target_variable])

# Load training dataset
print("Loading training data...")
training_data = np.load("../database/preprocessed/training_data.npz")
inputs = training_data["inputs"]
analysis_data = training_data["analysis_data"]
true_targets = training_data["targets"]
print("Loading completed")
if freemesh_size is None:
    # Load data from validation/neubernet_predicted_data.npz
    print("Loading predicted data...")
    predicted_data = np.load("metrics/neubernet_predicted_data.npz")
    predicted_targets = predicted_data["predicted_targets"]
    print("Loading completed")
else:
    # Load the FreeMesh data
    print("Loading FreeMesh predicted data...")
    freemesh_data = np.load(
        os.path.join(
            "./freemesh_data",
            f"neubernet_predicted_data_freemesh_bcs_size_{freemesh_size}.npz",
        )
    )
    predicted_targets = freemesh_data["predicted_targets"]
    print("Loading completed")

# Get the target values
analysis_index = analysis_ID - 1
analysis_data_indexes = inputs[:, 0].astype(int)
analysis_data_factors = inputs[:, 1]
indexes = np.where(
    np.logical_and(
        analysis_data_indexes == analysis_index,  # Indexes start from 0
        analysis_data_factors == analysis_factor,
    )
)[0]
true_targets = true_targets[indexes, target_variable]
predicted_targets = predicted_targets[indexes, target_variable]
coordinates = inputs[indexes, 2:]
errors = predicted_targets - true_targets

# Find in analysis_data_dict the key for which the value is "beta"
beta_key = [key for key, value in analysis_data_dict.items() if value == r"$\beta$"][0]

# Find in input_var_dict the key for which the value has "X" and "Y" in it
X_key = [key for key, value in input_var_dict.items() if "X" in value][0]
Y_key = [key for key, value in input_var_dict.items() if "Y" in value][0]

# Rotate the coordinates by beta
beta = np.radians(analysis_data[analysis_index, beta_key])
rotation_matrix = np.array(
    [[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]
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
analysis_data_values = analysis_data[analysis_index, 108:114]
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
    fig.savefig(
        f"analysis_{analysis_ID}_load_step_{analysis_factor}_target_{target_variable}.pdf"
    )
    fig.savefig(
        f"analysis_{analysis_ID}_load_step_{analysis_factor}_target_{target_variable}.svg"
    )
