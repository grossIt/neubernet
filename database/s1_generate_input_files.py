"""
s1_generate_input_files.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that generates N input files to be read by ANSYS
"""

import numpy as np
import os
import yaml

from scipy.stats.qmc import Sobol

# Read the configuration file
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define the number of input files to generate
N = config["N"]

# Notch path radius
RL = config["RL"]

# Save input files in the fem folder
path = "../fem/inputs"

# Create the directory if it does not exist
if not os.path.exists(path):
    os.makedirs(path)

    # In addition, create "bcs", "freemesh_bcs", "inputs", "meshes", and "results" folders
    os.makedirs("../fem/bcs")
    os.makedirs("../fem/freemesh_bcs")
    os.makedirs("../fem/meshes")
    os.makedirs("../fem/results")

# File base name
base_name = "Analysis_"

# Geometric bounds
theta_lim = config[
    "theta_lim"
]  # max angle between a notch side and the horizontal axis
gamma_lim = config[
    "gamma_lim"
]  # min angle between a notch side and the horizontal axis

# Mesh parameters
mean_radial_divisions = config["mean_radial_divisions"]
mean_circumferential_divisions = config["mean_circumferential_divisions"]
radial_variability = config["radial_variability"]
circumferential_variability = config["circumferential_variability"]

# Lower bounds for the random variables
R_min = config["lower_bounds"]["R"]
alpha_min = config["lower_bounds"]["alpha"]
sL_factor_min = config["lower_bounds"]["sL_factor"]
Fy_My_interp_min = config["lower_bounds"]["Fy_My_interp"]
Sy_min = config["lower_bounds"]["Sy"]
Et_min = config["lower_bounds"]["Et"]
ni_min = config["lower_bounds"]["ni"]

# Upper bounds for the random variables
R_max = config["upper_bounds"]["R"]
alpha_max = config["upper_bounds"]["alpha"]
R_ratio_1_max = config["upper_bounds"]["R_ratio_1"]
R_ratio_2_max = config["upper_bounds"]["R_ratio_2"]
sL_factor_max = config["upper_bounds"]["sL_factor"]
Fy_My_interp_max = config["upper_bounds"]["Fy_My_interp"]
Sy_max = config["upper_bounds"]["Sy"]
Et_max = config["upper_bounds"]["Et"]
ni_max = config["upper_bounds"]["ni"]

# Create a Sobol sequence generator
dimensions = 10
sobol = Sobol(d=dimensions, scramble=True)
sobol_samples = sobol.random(N)

# Generate the input files
for i in range(N):
    # Generate random values for the random variables (with Sobol sequences)
    R = sobol_samples[i, 0] * (R_max - R_min) + R_min
    alpha = sobol_samples[i, 1] * (alpha_max - alpha_min) + alpha_min

    # Note: beta_lim depends on alpha, so we handle it separately
    beta_lim = np.min([theta_lim - alpha, gamma_lim + alpha])
    beta = sobol_samples[i, 2] * (2 * beta_lim) - beta_lim

    R_ratio_1 = sobol_samples[i, 3] * (R_ratio_1_max - (1 + (RL / R))) + (1 + (RL / R))
    R_ratio_2 = sobol_samples[i, 4] * (R_ratio_2_max - (1 + (RL / R))) + (1 + (RL / R))
    sL_factor = sobol_samples[i, 5] * (sL_factor_max - sL_factor_min) + sL_factor_min
    Fy_My_interp = (
        sobol_samples[i, 6] * (Fy_My_interp_max - Fy_My_interp_min) + Fy_My_interp_min
    )
    Sy = sobol_samples[i, 7] * (Sy_max - Sy_min) + Sy_min
    Et = sobol_samples[i, 8] * (Et_max - Et_min) + Et_min
    ni = sobol_samples[i, 9] * (ni_max - ni_min) + ni_min

    radial_divisions = mean_radial_divisions + 2 * np.random.randint(
        -radial_variability, radial_variability + 1
    )  # High value to be included, so +1
    circumferential_divisions = mean_circumferential_divisions + 2 * np.random.randint(
        -circumferential_variability, circumferential_variability + 1
    )  # High value to be included, so +1

    # alpha is the semi-angle of notch aperture
    # beta is the angle between the notch bisector and the horizontal axis
    # theta_1 is the angle between the first notch side and the horizontal axis
    # theta_2 is the angle between the second notch side and the horizontal axis
    theta_1 = beta + alpha
    theta_2 = beta - alpha

    Fy_max = np.pi * R**2 * Sy
    My_max = (np.pi * (R**3) / 2) * (Sy / np.sqrt(3))

    # Interpolate the normal load and torsional moment, according to Fy_My_interp
    Fy_applied = Fy_max * Fy_My_interp
    My_applied = My_max * np.sqrt(
        1 - (Fy_My_interp**2)
    )  # We keep the same nominal equivalent stress

    # Create the input file content in FORTRAN
    content = f"R_midnotch = {R}\ntheta_1 = {theta_1}\ntheta_2 = {theta_2}\n"
    content += f"R_ratio_1 = {R_ratio_1}\nR_ratio_2 = {R_ratio_2}\n"
    content += f"sL_factor = {sL_factor}\nFy_applied = {Fy_applied}\nMy_applied = {My_applied}\n"
    content += f"Sy = {Sy}\nEt = {Et}\n"
    content += f"ni = {ni}\n"
    content += f"radial_divisions = {radial_divisions}\n"
    content += f"circumferential_divisions = {circumferential_divisions}"

    # Save the input file
    with open(os.path.join(path, f"{base_name}{i + 1}.txt"), "w") as f:
        f.write(content)

print(f"Generated {N} input files in the directory: {path}")
