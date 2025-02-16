"""
s5_neubernet_training.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that creates and trains NeuberNet on FE training data
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
from utils.definitions import NeuberNetComponent, NeuberNet
from utils.training import train_model
import yaml

# faulthandler logs
import faulthandler

faulthandler.enable()


if __name__ == "__main__":
    # Load configuration file
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Mesh parameter
    RL = config["RL"]  # Size of simulated domain

    # Analysis parameters
    load_step_size = config["load_step_size"]

    # Load the target_var_dict
    target_var_dict = config["target_var_dict"]

    # Load training parameters
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    cosine_annealing = config["cosine_annealing"]
    decimation_factor = config["decimation_factor"]
    leave_out_fraction = config["leave_out_fraction"]
    split_fraction = config["split_fraction"]
    whole_dataset_on_GPU = config["whole_dataset_on_GPU"]
    half_precision_database = config["half_precision_database"]
    min_epoch_save = config["min_epoch_save"]
    checkpoint_freq = config["checkpoint_freq"]
    random_seed = config["random_seed"]
    num_workers = config["num_workers"]

    # Load NeuberNet hyperparameters
    branch_input_dim = config["branch_input_dim"]
    nomad_secondary_input_dim = config["nomad_secondary_input_dim"]
    branch_hidden_dim = config["branch_hidden_dim"]
    nomad_hidden_dim = config["nomad_hidden_dim"]
    branch_hidden_layers = config["branch_hidden_layers"]
    nomad_hidden_layers = config["nomad_hidden_layers"]
    n_terms = config["n_terms"]
    activation_type = config["activation_type"]
    optimizer_algo = config["optimizer_algo"]
    loss_fun = config["loss_fun"]
    activation = nn.Tanh() if activation_type == "Tanh" else nn.ReLU()

    # Random seeds
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"  # Force CPU

    # Partial results path
    partial_results_path = "./partial_results/"

    # Create the directory if it does not exist
    if not os.path.exists(partial_results_path):
        os.makedirs(partial_results_path)

    # Trained results path
    trained_results_path = "./trained/"

    # Create the directory if it does not exist
    if not os.path.exists(trained_results_path):
        os.makedirs(trained_results_path)

    # Load data from database/training_data.npz
    print("Loading training data...")
    data = np.load("../database/preprocessed/training_data.npz")
    inputs = data["inputs"]
    targets = data["targets"]
    analysis_data = data["analysis_data"]
    print("Loading completed")

    # Normalize the input and target arrays (before the conversion to tensors)
    analysis_mean_input = analysis_data.mean(axis=0)
    analysis_mean_input[:branch_input_dim] = (
        0  # No mean normalization for the branch data
    )
    analysis_std_input = np.sqrt(
        np.mean((analysis_data - analysis_mean_input) ** 2, axis=0)
    )
    analysis_std_input[:branch_input_dim] = (
        np.sum(analysis_std_input[:branch_input_dim] ** 2) / branch_input_dim
    )  # Branch data is just scaled by a factor, so that their global variance is 1

    # Input coordinates are already normalized by R_notch, so we do not need to normalize them
    mean_coord_input = np.array([0.0, 0.0])
    std_coord_input = np.array([1.0, 1.0])

    # Compute the mean and std of the target variables
    mean_target = targets.mean(axis=0) * 0  # No mean normalization
    std_target = np.sqrt(np.mean((targets - mean_target) ** 2, axis=0))
    # std_target[:2] = np.sqrt(np.sum(std_target[:2] ** 2) / 2)  # Not needed
    std_target[2:8] = np.sqrt(
        np.sum(std_target[2:8] ** 2) / 6
    )  # Aggregated stress stds
    std_target[8:] = np.sqrt(np.sum(std_target[8:] ** 2) / 6)  # Aggregated strain stds

    # Normalize the data
    analysis_data -= analysis_mean_input
    analysis_data /= analysis_std_input
    inputs[:, 2:] -= mean_coord_input
    inputs[:, 2:] /= std_coord_input
    targets -= mean_target
    targets /= std_target
    mean_input = np.concatenate(
        (analysis_mean_input, mean_coord_input)
    )  # Mean of assembled inputs
    std_input = np.concatenate(
        (analysis_std_input, std_coord_input)
    )  # Std of assembled inputs
    print("Dataset normalized")

    # Create tensors
    if half_precision_database:
        input_tensor = torch.tensor(inputs[:, 2:], dtype=torch.float16)
        target_tensor = torch.tensor(targets, dtype=torch.float16)
        analysis_data_tensor = torch.tensor(analysis_data, dtype=torch.float16)
        analysis_data_indexes = torch.tensor(inputs[:, 0], dtype=torch.int)
        analysis_data_factor = torch.tensor(inputs[:, 1], dtype=torch.float16)
    else:
        input_tensor = torch.tensor(inputs[:, 2:], dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        analysis_data_tensor = torch.tensor(analysis_data, dtype=torch.float32)
        analysis_data_indexes = torch.tensor(inputs[:, 0], dtype=torch.int)
        analysis_data_factor = torch.tensor(inputs[:, 1], dtype=torch.float32)
    inputs = None
    targets = None
    analysis_data = None
    print("Training tensors created")

    # Save normalization parameters
    normalization_path = os.path.join(
        partial_results_path,
        f"neubernet_normalization.npz",
    )
    normalization_params = {
        "mean_input": mean_input,
        "std_input": std_input,
        "mean_target": mean_target,
        "std_target": std_target,
    }
    torch.save(normalization_params, normalization_path)
    print("Normalization parameters saved successfully")

    # Instantiate NeuberNet
    neubernet_components = [
        NeuberNetComponent(
            branch_input_dim,
            nomad_secondary_input_dim,
            1,
            branch_hidden_dim,
            nomad_hidden_dim,
            branch_hidden_layers,
            nomad_hidden_layers,
            n_terms,
            activation,
        ).to(device)
        for _ in range(target_tensor.shape[1])
    ]

    neubernet = NeuberNet(neubernet_components).to(device)

    # Train the model
    train_losses, test_losses = train_model(
        neubernet,
        input_tensor,
        target_tensor,
        analysis_data_tensor,
        analysis_data_indexes,
        analysis_data_factor,
        branch_input_dim,
        split_fraction,
        normalization_params,
        optimizer_algo,
        partial_results_path,
        trained_results_path,
        decimation_factor,
        leave_out_fraction,
        whole_dataset_on_GPU,
        half_precision_database,
        device,
        batch_size,
        num_epochs,
        learning_rate,
        weight_decay,
        cosine_annealing,
        min_epoch_save,
        checkpoint_freq,
        name="neubernet",
    )
    train_losses = np.stack(train_losses)
    test_losses = np.stack(test_losses)

    # Plot the training and test loss curves
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), train_losses, color="k", alpha=0.5)
    ax.plot(range(1, num_epochs + 1), test_losses, color="r", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Test Loss Curves")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend()

    plt.show()
