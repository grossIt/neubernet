"""
s7_signnet_training.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that creates and trains SignNet on FE training data
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
from utils.definitions import NeuberNetComponent, SignNet
from utils.training import train_model_bce
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

    # Load SignNet-specific training parameters
    num_epochs = config["sign_num_epochs"]
    batch_size = config["sign_batch_size"]
    learning_rate = float(config["sign_learning_rate"])
    weight_decay = float(config["sign_weight_decay"])
    cosine_annealing = config["sign_cosine_annealing"]
    loss_fun = config["sign_loss_fun"]

    # Load other training parameters
    decimation_factor = config["decimation_factor"]
    leave_out_fraction = config["leave_out_fraction"]
    split_fraction = config["split_fraction"]
    whole_dataset_on_GPU = config["whole_dataset_on_GPU"]
    half_precision_database = config["half_precision_database"]
    min_epoch_save = config["min_epoch_save"]
    checkpoint_freq = config["checkpoint_freq"]
    random_seed = config["random_seed"]
    num_workers = config["num_workers"]

    # Load SignNet hyperparameters
    branch_input_dim = config["branch_input_dim"]
    nomad_secondary_input_dim = config["nomad_secondary_input_dim"]
    branch_hidden_dim = config["branch_hidden_dim"]
    nomad_hidden_dim = config["sign_nomad_hidden_dim"]
    branch_hidden_layers = config["branch_hidden_layers"]
    nomad_hidden_layers = config["nomad_hidden_layers"]
    n_terms = config["n_terms"]
    activation_type = config["activation_type"]
    optimizer_algo = config["optimizer_algo"]
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
    data = np.load("../database/preprocessed/sign_data.npz")
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

    # Normalize
    analysis_data -= analysis_mean_input
    analysis_data /= analysis_std_input
    mean_input = analysis_mean_input
    std_input = analysis_std_input
    print("Dataset normalized")

    # Create tensors
    if half_precision_database:
        input_tensor = None
        target_tensor = torch.tensor(targets, dtype=torch.float16)
        analysis_data_tensor = torch.tensor(analysis_data, dtype=torch.float16)
        analysis_data_indexes = torch.tensornp.real((inputs[:, 0]), dtype=torch.int32)
        analysis_data_factor = torch.tensor(inputs[:, 1], dtype=torch.complex64)
    else:
        input_tensor = None
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        analysis_data_tensor = torch.tensor(analysis_data, dtype=torch.float32)
        analysis_data_indexes = torch.tensor(np.real(inputs[:, 0]), dtype=torch.int32)
        analysis_data_factor = torch.tensor(inputs[:, 1], dtype=torch.complex64)
    inputs = None
    targets = None
    analysis_data = None
    print("Training tensors created")

    # Save normalization parameters
    normalization_path = os.path.join(
        partial_results_path,
        f"signnet_normalization.npz",
    )
    normalization_params = {
        "mean_input": mean_input,
        "std_input": std_input,
    }
    torch.save(normalization_params, normalization_path)
    print("Normalization parameters saved successfully")

    # Instantiate SignNet
    signnet_components = [
        NeuberNetComponent(
            branch_input_dim,
            nomad_secondary_input_dim - 2,  # No input coordinates!
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

    signnet = SignNet(signnet_components).to(device)

    # Train the model
    train_accuracies, test_accuracies = train_model_bce(
        signnet,
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
        name="signnet",
    )
    train_accuracies = np.stack(train_accuracies)
    test_accuracies = np.stack(test_accuracies)

    # Load neubernet and assign SignNet to it
    neubernet = torch.load("trained/neubernet.pt", weights_only=False)
    neubernet.signnet = signnet
    neubernet.production_mode = True  # Set NeuberNet in production mode
    torch.save(neubernet, "trained/neubernet.pt")

    # Plot the training and test loss curves, use object-oriented approach
    fig, ax = plt.subplots()
    ax.plot(range(1, num_epochs + 1), train_accuracies, color="k", alpha=0.5)
    ax.plot(range(1, num_epochs + 1), test_accuracies, color="r", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Test Accuracy Curves")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend()

    plt.show()
