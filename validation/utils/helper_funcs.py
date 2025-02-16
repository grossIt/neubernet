"""
helper_funcs.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Helper functions for the validation scripts
"""

import os
import sys

import numpy as np
import pandas as pd
import pickle
import torch
import yaml

import alphashape
import matplotlib.colors as mcolors
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon


def import_path_file(filename, expected_num_columns=3):
    data = []
    with open(filename, "r") as file:
        for line in file:
            # Split the line by tabs
            parts_temp = line.strip().split(" ")
            # Remove empty strings
            parts = [x for x in list(parts_temp) if x]
            # Check if the number of columns is consistent and if all elements are numbers
            if (
                len(parts) == expected_num_columns
                and (parts[0][0].isdigit() or parts[0][0] == "-")
                and (parts[1][0].isdigit() or parts[1][0] == "-")
            ):
                # Convert the line into a NumPy array
                row = np.array(parts, dtype=float)
                data.append(row)
            # else:
            #     print("Skipping line:", line.strip())

    return np.array(data)


def import_delimited_file(filename, expected_num_columns=3):
    data = []
    with open(filename, "r") as file:
        for line in file:
            # Split the line by tabs
            parts = line.strip().split(" ")
            # Remove empty strings
            parts = [x for x in parts if x]
            # Check if the number of columns is consistent and if all elements are numbers
            if (
                len(parts) == expected_num_columns
                and parts[0].isdigit()
                and (parts[1][0].isdigit() or parts[1][0] == "-")
            ):
                # Convert the line into a NumPy array
                row = np.array(parts, dtype=float)
                data.append(row)
            # else:
            #     print("Skipping line:", line.strip())

    return np.array(data)


def evaluate_neubernet_metrics(
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    freemesh_data_path=None,
):
    # Load configuration file
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the target_var_dict
    target_var_dict = config["target_var_dict"]

    # Load the database split fraction
    N = config["N"]
    split_fraction = config["split_fraction"]

    # Load NeuberNet hyperparameters
    branch_input_dim = config["branch_input_dim"]
    load_step_size = config["load_step_size"]

    models_path = "../model/trained"

    # Load the trained model
    neubernet = torch.load(
        os.path.join(models_path, "neubernet.pt"), weights_only=False
    ).to(device)

    # Load data from database/training_data.npz
    print("Loading training data...")
    data = np.load("../database/preprocessed/training_data.npz")
    inputs = data["inputs"]
    targets = data["targets"]
    analysis_data = data["analysis_data"]
    print("Loading completed")

    if freemesh_data_path is not None:
        print("Loading custom analysis data...")
        custom_analysis_data = np.load(freemesh_data_path)
        analysis_data = custom_analysis_data["analysis_data"]
        print("Custom analysis data loaded")

    # Convert data to tensors
    input_tensor = torch.tensor(inputs[:, 2:], dtype=torch.float32).to(device)
    target_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    predicted_target_tensor = torch.zeros_like(target_tensor).to(device)
    analysis_data_tensor = torch.tensor(analysis_data, dtype=torch.float32).to(device)
    analysis_data_indexes = torch.tensor(inputs[:, 0], dtype=torch.int).to(device)
    analysis_data_factors = torch.tensor(inputs[:, 1], dtype=torch.float32).to(device)
    print("Training tensors created")

    # Make sure that there are actually N analyses
    assert analysis_data.shape[0] == N, "The number of analyses is not equal to N"
    assert (
        torch.max(analysis_data_indexes) == N - 1
    ), "The number of analyses referenced by the database is not equal to N"

    # Predict the target values
    neubernet.eval()
    print("--- GLOBAL ERRORS EVALUATION ---")
    # Cycle over the analyses
    with torch.no_grad():
        # Get the number of unique combinations of analysis_data_indexes and analysis_data_factors
        analysis_combos = torch.unique_consecutive(
            torch.stack((analysis_data_indexes, analysis_data_factors), dim=1),
            dim=0,
        )

        # Initialize storing tensors
        max_target_values_tensor = torch.zeros(
            (analysis_combos.shape[0], target_tensor.shape[1]), dtype=torch.float32
        ).to(device)
        max_predicted_values_tensor = torch.zeros_like(max_target_values_tensor)
        max_abs_error_tensor = torch.zeros_like(max_target_values_tensor)
        abs_error_on_max_values_tensor = torch.zeros_like(max_target_values_tensor)
        squared_error_on_max_values_tensor = torch.zeros_like(max_target_values_tensor)
        summed_squared_error_tensor = torch.zeros_like(max_target_values_tensor)
        mse_error_tensor = torch.zeros_like(max_target_values_tensor)

        bad_analyses = []

        bc_index = 0
        for i, analysis in enumerate(analysis_data_tensor):
            print(f"Computing errors for analysis {i+1}")

            # Select the data points corresponding to the current analysis
            indexes_i = torch.where(analysis_data_indexes == i)[0]

            # Get the load factors for the current analysis
            factors = torch.unique(analysis_data_factors[indexes_i])

            # Check that no two consecutive load factors are the same up to a slight tolerance
            if torch.any(torch.abs(factors[1:] - factors[:-1]) < 1e-6):
                raise ValueError(
                    f"Two different load factors closer than 1e-6 has been found in analysis {i+1}. Please check!"
                )

            # Cycle over the different load factors
            for j in range(len(factors)):
                # Restrict indexes_i to the current load factor
                sub_indexes = torch.where(
                    analysis_data_factors[indexes_i] == factors[j]
                )[0]
                indexes = indexes_i[sub_indexes]
                coord_inputs = input_tensor[indexes]
                current_input = torch.cat(
                    (
                        analysis.unsqueeze(0).repeat(len(indexes), 1),
                        coord_inputs,
                    ),
                    dim=1,
                )
                current_input[:, :branch_input_dim] *= factors[j]
                ground_truth_targets = target_tensor[indexes]
                predicted_targets = neubernet(current_input)
                predicted_target_tensor[indexes] = predicted_targets

                # Compute the maximum absolute error between the true and predicted target values
                max_abs_error, _ = torch.max(
                    torch.abs(ground_truth_targets - predicted_targets), dim=0
                )

                max_target_values, _ = torch.max(torch.abs(ground_truth_targets), dim=0)
                max_predicted_values, _ = torch.max(torch.abs(predicted_targets), dim=0)

                # Compute the absolute error on the maximum values
                abs_error_on_max_values = torch.abs(
                    max_target_values - max_predicted_values
                )

                # Compute the squared error on the maximum values
                squared_error_on_max_values = (
                    max_target_values - max_predicted_values
                ) ** 2

                # Print (i,j) if the maximum relative error is greater 0.5
                if torch.any(max_abs_error > 0.5):
                    # Store (i,j) in a list
                    bad_analyses.append((i + 1, j))
                    print(
                        f"A maximum absolute error > 0.5 has been found in analysis {i+1},{j}"
                    )

                # Compute the sum of squared errors between the true and predicted target values
                summed_squared_error_tensor[bc_index, :] = torch.sum(
                    (ground_truth_targets - predicted_targets) ** 2, dim=0
                )

                # Store the maximum absolute and relative errors
                max_target_values_tensor[bc_index, :] = max_target_values
                max_predicted_values_tensor[bc_index, :] = max_predicted_values
                max_abs_error_tensor[bc_index, :] = max_abs_error
                abs_error_on_max_values_tensor[bc_index, :] = abs_error_on_max_values
                squared_error_on_max_values_tensor[bc_index, :] = (
                    squared_error_on_max_values
                )
                mse_error_tensor[bc_index, :] = summed_squared_error_tensor[
                    bc_index, :
                ] / len(indexes)

                bc_index += 1

    # Get the indexes of the test data
    # first_test_index = torch.where(analysis_data_indexes >= split_fraction * N)[0][0]
    first_test_index = int(
        split_fraction * predicted_target_tensor.shape[0]
    )  # More correct
    first_test_analysis_index = analysis_data_indexes[first_test_index]
    first_test_analysis_factor = analysis_data_factors[first_test_index]

    # Find the corresponding position in analysis_combos
    first_test_analysis_combo_index = torch.where(
        (analysis_combos[:, 0] == first_test_analysis_index)
        & (analysis_combos[:, 1] == first_test_analysis_factor)
    )[0][0]
    test_indexes = torch.arange(
        first_test_analysis_combo_index, analysis_combos.shape[0]
    )
    # test_indexes = torch.arange(0, analysis_combos.shape[0])  # To test over the entire database

    rms_error_tensor = torch.sqrt(mse_error_tensor)

    # Save the analysis-wise metrics to pickle
    if freemesh_data_path is not None:
        # Extract only the file name from the custom_analysis_data_path, removing the extension
        custom_analysis_data_file = os.path.splitext(
            os.path.basename(freemesh_data_path)
        )[0]

        # Create a dictionary to hold the 2D arrays for each metric
        analysis_metrics_dict = {
            "Max abs error": max_abs_error_tensor[test_indexes].cpu().numpy(),
            "Max target value": max_target_values_tensor[test_indexes].cpu().numpy(),
            "Max predicted value": max_predicted_values_tensor[test_indexes]
            .cpu()
            .numpy(),
            "Abs error on max value": abs_error_on_max_values_tensor[test_indexes]
            .cpu()
            .numpy(),
            "RMS error": rms_error_tensor[test_indexes].cpu().numpy(),
            "MSE error": mse_error_tensor[test_indexes].cpu().numpy(),
        }

        # Save each metric as a separate DataFrame for 2D arrays
        # Convert each element of analysis_combos to a tuple
        analysis_metrics_dfs = {
            metric_name: pd.DataFrame(
                data,
                columns=target_var_dict.values(),
                index=[
                    tuple(
                        [
                            analysis_combos[i, 0].cpu().numpy() + 1,  # ID
                            analysis_combos[i, 1].cpu().numpy(),
                        ]
                    )
                    for i in test_indexes
                ],
            )
            for metric_name, data in analysis_metrics_dict.items()
        }

        # Save the dictionary to a pickle file
        with open(
            os.path.join(
                "./freemesh_data",
                f"neubernet_analysis_metrics_{custom_analysis_data_file}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(analysis_metrics_dfs, f)
    else:
        # Create a dictionary to hold the 2D arrays for each metric
        analysis_metrics_dict = {
            "Max abs error": max_abs_error_tensor[test_indexes].cpu().numpy(),
            "Max target value": max_target_values_tensor[test_indexes].cpu().numpy(),
            "Max predicted value": max_predicted_values_tensor[test_indexes]
            .cpu()
            .numpy(),
            "Abs error on max value": abs_error_on_max_values_tensor[test_indexes]
            .cpu()
            .numpy(),
            "RMS error": rms_error_tensor[test_indexes].cpu().numpy(),
            "MSE error": mse_error_tensor[test_indexes].cpu().numpy(),
        }

        # Save each metric as a separate DataFrame for 2D arrays
        # Convert each element of analysis_combos to a tuple
        analysis_metrics_dfs = {
            metric_name: pd.DataFrame(
                data,
                columns=target_var_dict.values(),
                index=[
                    tuple(
                        [
                            analysis_combos[i, 0].cpu().numpy() + 1,  # ID
                            analysis_combos[i, 1].cpu().numpy(),
                        ]
                    )
                    for i in test_indexes
                ],
            )
            for metric_name, data in analysis_metrics_dict.items()
        }

        # Save the dictionary to a pickle file
        with open("metrics/neubernet_analysis_metrics.pkl", "wb") as f:
            pickle.dump(analysis_metrics_dfs, f)

    # Compute the global metrics
    max_target_values_global, argmax_target_values_global = torch.max(
        max_target_values_tensor[test_indexes], dim=0
    )
    max_predicted_values_global, argmax_predicted_values_global = torch.max(
        max_predicted_values_tensor[test_indexes], dim=0
    )
    max_abs_error_global, argmax_abs_error_global = torch.max(
        max_abs_error_tensor[test_indexes], dim=0
    )
    abs_error_on_max_values_global, argmax_abs_error_on_max_values_global = torch.max(
        abs_error_on_max_values_tensor[test_indexes], dim=0
    )
    squared_error_on_max_values_global = torch.sum(
        squared_error_on_max_values_tensor[test_indexes], dim=0
    ) / len(test_indexes)
    squared_error_global = torch.sum(
        summed_squared_error_tensor[test_indexes], dim=0
    ) / ((1 - split_fraction) * predicted_target_tensor.shape[0])
    max_mse_error_global, argmax_mse_error_global = torch.max(
        mse_error_tensor[test_indexes], dim=0
    )

    # Extract actual indexes in the database
    argmax_target_values_global = (
        analysis_combos[test_indexes][argmax_target_values_global].cpu().numpy()
    )
    argmax_predicted_values_global = (
        analysis_combos[test_indexes][argmax_predicted_values_global].cpu().numpy()
    )
    argmax_abs_error_global = (
        analysis_combos[test_indexes][argmax_abs_error_global].cpu().numpy()
    )
    argmax_abs_error_on_max_values_global = (
        analysis_combos[test_indexes][argmax_abs_error_on_max_values_global]
        .cpu()
        .numpy()
    )
    argmax_mse_error_global = (
        analysis_combos[test_indexes][argmax_mse_error_global].cpu().numpy()
    )

    # Create a pandas dataframe to store the global metrics
    # The rows are the values in the target_var_dict dictionary
    # The headers are the global metrics
    global_test_metrics = pd.DataFrame(
        {
            "Max abs error": max_abs_error_global.cpu().numpy(),
            "Max abs analysis ID": argmax_abs_error_global[:, 0] + 1,
            "Max abs yield factor": argmax_abs_error_global[:, 1],
            # "Max target value": max_target_values_global.cpu().numpy(),
            # "Max target value analysis ID": argmax_target_values_global[:, 0] + 1,
            # "Max target value yield factor": argmax_target_values_global[:, 1],
            # "Max predicted value": max_predicted_values_global.cpu().numpy(),
            # "Max predicted value analysis ID": argmax_predicted_values_global[:, 0] + 1,
            # "Max predicted value yield factor": argmax_predicted_values_global[:, 1],
            "Max abs error on max values": abs_error_on_max_values_global.cpu().numpy(),
            "Max abs error on max values analysis ID": argmax_abs_error_on_max_values_global[
                :, 0
            ]
            + 1,
            "Max abs error on max values yield factor": argmax_abs_error_on_max_values_global[
                :, 1
            ],
            "Max RMS error": np.sqrt(max_mse_error_global.cpu().numpy()),
            "Max RMS error analysis ID": argmax_mse_error_global[:, 0] + 1,
            "Max RMS error yield factor": argmax_mse_error_global[:, 1],
            "RMS error on max values": np.sqrt(
                squared_error_on_max_values_global.cpu().numpy()
            ),
            "RMS error": np.sqrt(squared_error_global.cpu().numpy()),
        },
        index=target_var_dict.values(),
    )

    # Print the global metrics dataframe
    print("--- Global test metrics ---")
    print(global_test_metrics)

    # Save the dataframe to pickle
    if freemesh_data_path is not None:
        # Extract only the file name from the custom_analysis_data_path, removing the extension
        custom_analysis_data_file = os.path.splitext(
            os.path.basename(freemesh_data_path)
        )[0]

        global_test_metrics.to_pickle(
            os.path.join(
                "./freemesh_data",
                f"neubernet_global_metrics_{custom_analysis_data_file}.pkl",
            )
        )

        # Save the inputs, analysis_data, and predicted targets to npz
        np.savez(
            os.path.join(
                "./freemesh_data",
                f"neubernet_predicted_data_{custom_analysis_data_file}.npz",
            ),
            predicted_targets=predicted_target_tensor.cpu().numpy(),
        )
    else:
        global_test_metrics.to_pickle("metrics/neubernet_global_metrics.pkl")

        # Save the inputs, analysis_data, and predicted targets to npz
        np.savez(
            "metrics/neubernet_predicted_data.npz",
            predicted_targets=predicted_target_tensor.cpu().numpy(),
        )


def import_freemesh_bcs(freemesh_data_folder):
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    E = config["E"]
    R_notch = config["R_notch"]
    RL = config["RL"]
    load_step_size = config["load_step_size"]

    # Define the path to the fem folder
    database_path = "../fem"

    # Check the number of input files
    n_freemesh = len(os.listdir(freemesh_data_folder))
    n_database = len(os.listdir(os.path.join(database_path, "inputs")))

    # Check whether N is consistent with the number of simulations specified in the config file
    if n_freemesh != n_database:
        raise ValueError(
            f"The number of freemesh data files ({n_freemesh}) does not match the number of simulations "
            f"in the database ({n_database})\nCheck the input files in the 'Inputs' folder"
        )

    analysis_data_list = []

    # Loop over the analysis IDs
    for i in range(1, n_freemesh + 1):
        # Import parameters from the input file
        input_file = os.path.join(database_path, "inputs", f"Analysis_{i}.txt")
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

        # Import the input displacements and rotations *AT YIELD* at R=5
        BC_file = os.path.join(freemesh_data_folder, f"FreeMesh_BC_{i}.txt")

        # Read the file by using TAB as a separator; put a NaN where there are strings
        BC_data_import = import_path_file(BC_file, expected_num_columns=7)

        # Import the ratio between the simulated loads and the corresponding elastic limit
        el_ratio_file = os.path.join(database_path, "bcs", f"El_Ratio_{i}.txt")

        # Read the file by using TAB as a separator; put a NaN where there are strings
        el_ratio_import = import_path_file(el_ratio_file, expected_num_columns=3)
        el_ratio = el_ratio_import[0, 2]

        x = BC_data_import[:, 0]
        y = BC_data_import[:, 1]
        theta = np.arctan2(y, x) - np.radians(beta)
        # theta = BC_data_import[:, 3] / (RL * R_notch)  - np.pi  # Old
        ux = BC_data_import[:, 4] / el_ratio
        uy = BC_data_import[:, 5] / el_ratio
        roty = BC_data_import[:, 6] / el_ratio

        # Now, set theta to be in the range [-np.pi, np.pi]
        theta = np.where(theta > np.pi, theta - 2 * np.pi, theta)
        theta = np.where(theta < -np.pi, theta + 2 * np.pi, theta)

        # Sort data by theta
        sorting_indices = np.argsort(theta)
        theta = theta[sorting_indices]
        ux = ux[sorting_indices]
        uy = uy[sorting_indices]
        roty = roty[sorting_indices]

        # Apply a very small thresholding tolerance to ux, uy, and roty
        ux = np.where(np.abs(ux) < 1e-12, 0, ux)
        uy = np.where(np.abs(uy) < 1e-12, 0, uy)
        roty = np.where(np.abs(roty) < 1e-12, 0, roty)

        # Get points where ux, uy, and roty are not all zero
        nonzero_indices = np.where(
            np.logical_or(np.logical_or(ux != 0, uy != 0), roty != 0)
        )[0]

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

        # UX-UY formulation (UR-UT formulation has been discarded)
        # Transform the displacements and rotations into local (UX, UY) displacements
        ux_local_interp = ux_interp * np.cos(np.radians(beta)) + uy_interp * np.sin(
            np.radians(beta)
        )
        uy_local_interp = -ux_interp * np.sin(np.radians(beta)) + uy_interp * np.cos(
            np.radians(beta)
        )

        # Build the data array
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
            ]  # Additional normalization step, see the paper
        )

        # Now, we can generate the parameter_data array
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

        # Print a message
        print(f"Imported data for Analysis ID: {i}")

    # Combine all the data into a single NumPy array
    analysis_data = np.concatenate(analysis_data_list)

    print(analysis_data.shape)

    # Get the last subfolder in freemesh_data_folder
    freemesh_data_name = os.path.basename(freemesh_data_folder)

    # Save the analysis data to npz
    np.savez_compressed(
        os.path.join("./freemesh_data", f"{freemesh_data_name}.npz"),
        analysis_data=analysis_data,
    )

    print("Freemesh data saved successfully")


def plot_analysis(
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
    fig=None,
    ax=None,
    alpha_shape=2,
    sigma_y=None,
    predicted_only=False,
    labels=True,
):
    """
    Generate comparative plots of true values, predicted values, and errors for a NeuberNet prediction.
    If 'fig' and `ax` are provided, plots on that axis. Otherwise, creates a new figure.
    If `sigma_y` is provided, modifies predicted targets accordingly.
    If `predicted_only` is True, only plots predicted values.
    """
    tol = 0.05  # Plotting tol, to avoid cropping the boundary lines

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    # Get the target values
    analysis_index = analysis_ID - 1
    analysis_data_indexes = inputs[:, 0].astype(int)
    analysis_data_factors = inputs[:, 1]
    indexes = np.where(
        np.logical_and(
            analysis_data_indexes == analysis_index,
            analysis_data_factors == analysis_factor,
        )
    )[0]
    analysis = analysis_data[analysis_index]
    true_targets = true_targets[indexes, target_variable]
    predicted_targets = predicted_targets[indexes, target_variable]
    coordinates = inputs[indexes, 2:]

    # Modify predicted targets if sigma_y is provided
    if sigma_y is not None:
        # Load configuration file
        with open("../../config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Load NeuberNet hyperparameters
        branch_input_dim = config["branch_input_dim"]

        models_path = "../model/trained"

        # Load the trained model
        neubernet = torch.load(
            os.path.join(models_path, "neubernet.pt"), weights_only=False
        )

        analysis[111] = sigma_y  # Modify the yield stress

        # Convert data to tensors
        analysis_tensor = torch.tensor(analysis, dtype=torch.float32)
        coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

        current_input = torch.cat(
            (
                analysis_tensor.unsqueeze(0).repeat(len(indexes), 1),
                coordinates_tensor,
            ),
            dim=1,
        )
        current_input[:, :branch_input_dim] *= analysis_factor
        predicted_targets = np.array(neubernet(current_input))

    errors = predicted_targets - true_targets
    vmin, vmax = min(true_targets.min(), predicted_targets.min()), max(
        true_targets.max(), predicted_targets.max()
    )
    error_abs = np.abs(errors).max()

    # For colormap purposes
    if vmax != 0 and np.abs(vmin / vmax) < 1e-6:
        vmin -= np.abs(vmax) * 1e-6
    if vmin != 0 and np.abs(vmax / vmin) < 1e-6:
        vmax += np.abs(vmin) * 1e-6

    # Find in analysis_data_dict the key for which the value is "beta"
    beta_key = [
        key for key, value in analysis_data_dict.items() if value == r"$\beta$"
    ][0]

    # Find in input_var_dict the key for which the value has "x /" and "y /" in it
    X_key = [key for key, value in input_var_dict.items() if "x /" in value][0]
    Y_key = [key for key, value in input_var_dict.items() if "y /" in value][0]

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

    if predicted_only:
        contourf = ax.tricontourf(
            triang_masked,
            predicted_targets,
            levels=np.linspace(vmin, vmax, 20 + 2),
        )
        ax.plot(
            convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5
        )
        if labels:
            ax.set_title(target_var_dict[target_variable] + " Predicted")
            ax.set_xlabel(input_var_dict[X_key])
            ax.set_ylabel(input_var_dict[Y_key])
        ax.set_axis_off()
        ax.set_xlim(-RL - tol, RL + tol)
        ax.set_ylim(-RL - tol, RL + tol)
        ax.set_aspect("equal", "box")
        fig.colorbar(contourf, ax=ax)
        return fig, ax

    # Define colormap
    cmap = "viridis"

    # Plot the true values
    contourf1 = ax[0].tricontourf(
        triang_masked,
        true_targets,
        levels=np.linspace(vmin, vmax, 20 + 2),
        cmap=cmap,
    )
    ax[0].plot(
        convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5
    )
    if labels:
        ax[0].set_title(target_var_dict[target_variable] + " Ground Truth")
        ax[0].set_xlabel(input_var_dict[X_key])
        ax[0].set_ylabel(input_var_dict[Y_key])
    ax[0].set_axis_off()
    ax[0].set_xlim(-RL - tol, RL + tol)
    ax[0].set_ylim(-RL - tol, RL + tol)
    ax[0].set_aspect("equal", "box")

    # Plot the predicted values
    contourf2 = ax[1].tricontourf(
        triang_masked,
        predicted_targets,
        levels=np.linspace(vmin, vmax, 20 + 2),
        cmap=cmap,
    )
    ax[1].plot(
        convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5
    )
    if labels:
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
    ax[2].plot(
        convex_hull.exterior.xy[0], convex_hull.exterior.xy[1], "k-", linewidth=0.5
    )
    if labels:
        ax[2].set_title(target_var_dict[target_variable] + " Error")
        ax[2].set_xlabel(input_var_dict[X_key])
    ax[2].set_axis_off()
    ax[2].set_xlim(-RL - tol, RL + tol)
    ax[2].set_ylim(-RL - tol, RL + tol)
    ax[2].set_aspect("equal", "box")

    if np.abs(vmax) < 0.2 and np.abs(vmin) < 0.2:
        format = "%.2e"
    else:
        format = "%.2f"

    # Adjust colorbars
    cbar1 = fig.colorbar(
        contourf1,
        ax=[ax[0], ax[1]],
        fraction=0.021,
        pad=0.04,
        format=format,
        location="left",
    )
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_ticks(np.linspace(vmin, vmax, 3))

    cbar2 = fig.colorbar(contourf3, ax=ax[2], fraction=0.046, pad=0.04, format="%.2e")
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_ticks(np.linspace(-error_abs, error_abs, 3))

    return fig, ax
