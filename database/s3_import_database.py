"""
s3_import_database.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that imports the corresponding mesh and results from ANSYS, for each analysis in the database
"""

import os
import numpy as np
import yaml
from utils.file_io import import_delimited_file

# faulthandler logs
import faulthandler

faulthandler.enable()

# Read the configuration file
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Extract parameters from the config file
E = config["E"]
R_notch = config["R_notch"]
RL = config["RL"]
load_step_size = config["load_step_size"]

eps = 1e-3  # Tolerance for node locations

# Define the path to the fem folder
path = "../fem"

# Check the number of input files
N = len(os.listdir(os.path.join(path, "inputs")))

# Check whether N is consistent with the number of simulations specified in the config file
if N != config["N"]:
    raise ValueError(
        f"The number of input files ({N}) does not match the number of simulations specified "
        f"in the config file ({config['N']})\nCheck the input files in the 'Inputs' folder"
    )

analysis_data_list = []
input_target_pairs_list = []
fully_elastic_analyses = 0

# Loop over the analysis IDs
for i in range(1, N + 1):
    # Import parameters from the input file
    input_file = os.path.join(path, "inputs", f"Analysis_{i}.txt")
    with open(input_file, "r") as f:
        lines = f.readlines()
        R = float(lines[0].split("=")[1])
        theta_1 = float(lines[1].split("=")[1])
        theta_2 = float(lines[2].split("=")[1])
        R_ratio_1 = float(lines[3].split("=")[1])
        R_ratio_2 = float(lines[4].split("=")[1])
        sL_factor = float(lines[5].split("=")[1])
        Fy_applied = float(lines[6].split("=")[1])
        My_applied = float(lines[7].split("=")[1])
        Sy = float(lines[8].split("=")[1])
        Et = float(lines[9].split("=")[1])
        ni = float(lines[10].split("=")[1])

    # Calculate the alpha and beta angles
    alpha = (theta_1 - theta_2) / 2
    beta = (theta_1 + theta_2) / 2

    # Import the mesh
    mesh_file = os.path.join(path, "meshes", f"Mesh_{i}.txt")

    # Read the file by using TAB as a separator; put a NaN where there are strings
    mesh_data_master = import_delimited_file(mesh_file, expected_num_columns=4)

    # Raise an error if the mesh data is empty
    if len(mesh_data_master) == 0:
        raise ValueError(f"Mesh data is empty for Analysis ID: {i}")

    # Import the input displacements and rotations *AT YIELD* at R=5
    BC_file = os.path.join(path, "bcs", f"BC_{i}.txt")

    # Read the file by using TAB as a separator; put a NaN where there are strings
    BC_data_import = import_delimited_file(BC_file, expected_num_columns=2)

    # Group the data by the first column
    BC_data_dict = {}
    for row in BC_data_import:
        if int(row[0]) not in BC_data_dict:
            BC_data_dict[int(row[0])] = []
        BC_data_dict[int(row[0])].append(row[1])

    # Create a NumPy array, having:
    # - the first column as the node number, taken from the keys of BC_data_dict
    # - the second and third columns as their polar coordinates (in mesh_data)
    # - the fourth to sixth columns as their displacements and rotations (in BC_data)
    BC_data = np.array(
        [
            [
                node,
                *mesh_data_master[mesh_data_master[:, 0].astype(int) == node][0][1:3],
                *BC_data_dict[node],
            ]
            for node in BC_data_dict
        ]
    )

    # Sort the BC_data array by the third column (theta)
    BC_data = BC_data[BC_data[:, 2].argsort()]

    # Interpolate the displacements and rotations at equispaced angles (10 degrees)
    angles = np.arange(-180, 180, 10)
    ux_interp = np.interp(angles, BC_data[:, 2], BC_data[:, 3])
    uy_interp = np.interp(angles, BC_data[:, 2], BC_data[:, 4])
    roty_interp = np.interp(angles, BC_data[:, 2], BC_data[:, 5])

    # Compute the average of uy_interp and roty_interp by integrating them over angles
    uy_avg = np.trapezoid(uy_interp, angles) / (angles[-1] - angles[0])
    roty_avg = np.trapezoid(roty_interp, angles) / (angles[-1] - angles[0])

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

    # Build the normalized displacements array
    interpolated_data = np.zeros((len(angles), 4))
    interpolated_data[:, 0] = angles
    interpolated_data[:, 1] = ux_local_interp / R_notch
    interpolated_data[:, 2] = uy_local_interp / R_notch
    interpolated_data[:, 3] = (
        roty_interp  # No need to divide by R_notch, it's an angle...
    )
    ##

    # Now, we can generate the baseline branch_data array
    branch_data = np.array(
        [
            *interpolated_data[:, 1] * E / Sy,
            *interpolated_data[:, 2] * E / Sy,
            *interpolated_data[:, 3] * E / Sy,
        ]  # Additional normalization (see paper)
    )

    # Import the parameters for the analysis
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
    analysis_data_list.append(
        np.expand_dims(
            np.concatenate((branch_data, parameter_data)).astype(np.float32), 0
        )
    )

    # Import the base results for the elastic analysis
    results_file_en_el = os.path.join(path, "results", f"El_Send_{i}.txt")
    results_file_stress_el = os.path.join(path, "results", f"El_Stress_{i}.txt")

    # Read the file by using TAB as a separator; put a NaN where there are strings
    results_data_en_el_master = import_delimited_file(
        results_file_en_el, expected_num_columns=9
    )
    results_data_stress_el_master = import_delimited_file(
        results_file_stress_el, expected_num_columns=7
    )

    # Raise an error if the results data is empty
    if len(results_data_en_el_master) == 0 or len(results_data_stress_el_master) == 0:
        raise ValueError(f"Results data is empty for Analysis ID: {i}")

    # Now, for the {i} analysis, M analyses are available, depending on the number of load steps performed
    # before violating small-scale plasticity. For each of these analyses, we need to import the results.

    # Get the number of load steps, by looking at files Send_{i}_{j}.txt and taking the maximum value of j
    M = (
        max(
            [
                int(file.split("_")[-1].split(".")[0])
                for file in os.listdir(os.path.join(path, "results"))
                if file.startswith(f"Send_{i}_")
            ]
        )
        + 1
    )  # Files starts at j=0, so we need to add 1

    # Loop over the load steps
    for j in range(0, M):
        mesh_data = mesh_data_master.copy()

        # The branch data is scaled, depending on the load step ID
        branch_data_j = branch_data * (1 + load_step_size * j)

        # Scale the master elastic analyses according to the load step ID
        results_data_en_el = results_data_en_el_master.copy()
        results_data_stress_el = results_data_stress_el_master.copy()
        results_data_en_el[:, 1:3] *= (1 + load_step_size * j) ** 2
        results_data_stress_el[:, 1:] *= 1 + load_step_size * j

        # Import the results
        results_file_en = os.path.join(path, "results", f"Send_{i}_{j}.txt")
        results_file_stress = os.path.join(path, "results", f"Stress_{i}_{j}.txt")
        results_file_strain = os.path.join(path, "results", f"ePlas_{i}_{j}.txt")

        # Read the file by using TAB as a separator; put a NaN where there are strings
        results_data_en = import_delimited_file(results_file_en, expected_num_columns=9)
        results_data_stress = import_delimited_file(
            results_file_stress, expected_num_columns=7
        )
        results_data_strain = import_delimited_file(
            results_file_strain, expected_num_columns=7
        )

        # Raise an error if the results data is empty
        if (
            len(results_data_en) == 0
            or len(results_data_stress) == 0
            or len(results_data_strain) == 0
        ):
            raise ValueError(
                f"Results data is empty for Analysis ID: {i}, Load step: {j}"
            )

        # Make sure that all the results files have the same number of rows and that the first column is the same
        # If not, include only the rows so that the first column is the same
        rows_set = (
            set(results_data_en[:, 0].astype(int))
            & set(results_data_en_el[:, 0].astype(int))
            & set(results_data_stress[:, 0].astype(int))
            & set(results_data_stress_el[:, 0].astype(int))
            & set(results_data_strain[:, 0].astype(int))
        )
        rows_list = list(rows_set)  # Convert the sets to lists for indexing
        # Filter the results data
        results_data_en = results_data_en[
            np.isin(results_data_en[:, 0].astype(int), rows_list)
        ]
        results_data_en_el = results_data_en_el[
            np.isin(results_data_en_el[:, 0].astype(int), rows_list)
        ]
        results_data_stress = results_data_stress[
            np.isin(results_data_stress[:, 0].astype(int), rows_list)
        ]
        results_data_stress_el = results_data_stress_el[
            np.isin(results_data_stress_el[:, 0].astype(int), rows_list)
        ]
        results_data_strain = results_data_strain[
            np.isin(results_data_strain[:, 0].astype(int), rows_list)
        ]

        # Filter mesh_data, extracting only rows whose first element appears in the first column of every result_data
        mesh_data = mesh_data[
            np.logical_and.reduce(
                [
                    np.isin(
                        mesh_data[:, 0].astype(int), results_data_en[:, 0].astype(int)
                    ),
                    np.isin(
                        mesh_data[:, 0].astype(int),
                        results_data_en_el[:, 0].astype(int),
                    ),
                    np.isin(
                        mesh_data[:, 0].astype(int),
                        results_data_stress[:, 0].astype(int),
                    ),
                    np.isin(
                        mesh_data[:, 0].astype(int),
                        results_data_stress_el[:, 0].astype(int),
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

        # If the second column of results_data is all zero, then there's no plasticity
        if not np.any(results_data_en_diff[:, 1]):
            fully_elastic_analyses += 1

        # Concatenate mesh_data and results_data into a single array
        mesh_result_pair = np.concatenate(
            (
                mesh_data,
                results_data_en_diff,
                results_data_stress_diff,
                results_data_strain_diff,
            ),
            axis=1,
        )

        # Prepend to mesh_result_pair the *index* of the analysis, so that one can identify the analysis without
        # having to broadcast the branch_data and parameter_data arrays
        input_target_pairs = np.concatenate(
            (
                np.full((mesh_result_pair.shape[0], 1), i - 1),  # Python indexing!
                np.full(
                    (mesh_result_pair.shape[0], 1), 1 + load_step_size * j
                ),  # Just a multiplicative factor
                mesh_result_pair,
            ),
            axis=1,
        )

        # Append the combined array to the list, converting it to float32 for RAM efficiency
        input_target_pairs_list.append(input_target_pairs.astype(np.float32))

        # Print a message
        print(f"Imported data for Analysis ID: {i}")

        # Optional plots
        # if i == 10 and j == 1:
        #     # Make a plot of the first imported data, with equal axes
        #     plt.figure()
        #     plt.scatter(input_target_pairs[:, -4], input_target_pairs[:, -3])
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #     plt.title("Nodal coordinates")
        #     plt.gca().set_aspect("equal", adjustable="box")
        #
        #     # Print alpha, beta, R_ratio_1, R_ratio_2
        #     print(f"Alpha: {alpha}")
        #     print(f"Beta: {beta}")
        #     print(f"R_ratio_1: {R_ratio_1}")
        #     print(f"R_ratio_2: {R_ratio_2}")
        #
        #     # Plot the displacements as arrows
        #     # Polar coordinates are in the 2nd and 3rd columns of BC_data
        #     # (UX, UY) displacements are in the 4th and 5th columns of BC_data
        #     for row in BC_data:
        #         plt.arrow(
        #             row[1] * np.cos(np.radians(row[2])),
        #             row[1] * np.sin(np.radians(row[2])),
        #             row[3] * 100,
        #             row[4] * 100,
        #             head_width=0.05,
        #             head_length=0.1,
        #             fc="r",
        #             ec="r",
        #         )
        #
        #     plt.show()


# Combine all the data into a single NumPy array
analysis_data = np.concatenate(analysis_data_list)
training_array = np.concatenate(input_target_pairs_list)

# Divide the training data into inputs and targets
training_inputs = training_array[:, :-14]
training_targets = training_array[:, -14:]

print(training_inputs.shape)

print(f"{fully_elastic_analyses} analyses did not show plastic behavior")

# Save the training data to a file in the database folder
np.savez_compressed(
    "./preprocessed/training_data.npz",
    inputs=training_inputs,
    targets=training_targets,
    analysis_data=analysis_data,
)

print("Training data saved successfully")
