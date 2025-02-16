"""
s9_plot_analysis.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that plots a given analysis, comparing the true and predicted target values
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.helper_funcs import plot_analysis

### USER INPUTS ##
analysis_ID = 9999
analysis_factor = 5
target_variable = 6
save_plot = False
freemesh_size = None
### END USER INPUTS ###

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
#     }
# )

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

# Plot the analysis
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
plot_analysis(
    analysis_ID,
    analysis_factor,
    target_variable,
    inputs,
    true_targets,
    predicted_targets,
    analysis_data,
    analysis_data_dict,
    input_var_dict,
    target_var_dict,
    RL,
    fig=fig,
    ax=ax,
    alpha_shape=alpha_shape,
)

plt.show()
