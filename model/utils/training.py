"""
training.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Training utilities for machine-learning scripts
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from .definitions import FEMDataset


def train_model(
    model,
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
    decimation_factor=1,
    leave_out_fraction=0,
    whole_dataset_on_gpu=True,
    half_precision_database=False,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    batch_size=1024,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    cosine_annealing=True,
    min_epoch_save=1,
    checkpoint_freq=1,
    name="model",
):
    # Define the indices for the train and test split
    num_train = len(target_tensor)
    indices = list(range(num_train))
    split = int(split_fraction * num_train)  # 80-20 split

    # Splitting the indices into train and test
    train_indices, test_indices = indices[:split], indices[split:]

    # Decimate data for experiments with reduced datasets
    train_indices = train_indices[: len(train_indices) // decimation_factor]
    test_indices = test_indices[: len(test_indices) // decimation_factor]

    # Shuffle the indices for the training/test sets
    # (done *after* splitting, so that the test set are XY points from experiments that are not in the training set)
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    print("Training/test indices shuffled")

    # Drop out a fraction of the points
    train_indices = train_indices[
        int(leave_out_fraction * len(train_indices)) : len(train_indices)
    ]
    test_indices = test_indices[
        int(leave_out_fraction * len(test_indices)) : len(test_indices)
    ]

    # Check whether we are training over spatial coordinates or not
    input_train = input_tensor[train_indices] if input_tensor is not None else None
    input_test = input_tensor[test_indices] if input_tensor is not None else None

    # Load Dataset
    if whole_dataset_on_gpu and torch.cuda.is_available():
        print("Loading the entire dataset on GPU")
        dataset_device = device
    else:
        print("Loading the entire dataset on RAM")
        dataset_device = torch.device("cpu")

    train_loader = FEMDataset(
        input_train,
        target_tensor[train_indices],
        analysis_data_tensor[:, :branch_input_dim],
        analysis_data_tensor[:, branch_input_dim:],
        analysis_data_indexes[train_indices],
        analysis_data_factor[train_indices],
        batch_size,
        dataset_device,
    )
    test_loader = FEMDataset(
        input_test,
        target_tensor[test_indices],
        analysis_data_tensor[:, :branch_input_dim],
        analysis_data_tensor[:, branch_input_dim:],
        analysis_data_indexes[test_indices],
        analysis_data_factor[test_indices],
        batch_size,
        dataset_device,
    )

    # Define the loss function
    loss_function = nn.MSELoss(reduction="none")

    # Define the optimizer
    optimizer = getattr(optim, optimizer_algo)(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )

    if half_precision_database:
        scaler = GradScaler()

    # Check for checkpoint
    checkpoint_path = os.path.join(partial_results_path, name + "_latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        load_checkpoint = input(
            "Checkpoint found. Do you want to load from this checkpoint? (y/n): "
        ).lower()
        if load_checkpoint == "y":
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        else:
            start_epoch = 1
            train_losses = []
            test_losses = []
    else:
        start_epoch = 1
        train_losses = []
        test_losses = []

    # Schedule a learning rate reduction
    if cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, last_epoch=start_epoch - 2
        )
        # Last epoch is 0-indexed, while start_epoch is 1-indexed, so we subtract 2
    else:
        scheduler = None

    # Initiate min_losses to the minimum of each "column" of test_losses if test_losses is not empty
    if test_losses:
        min_losses = np.min(test_losses, axis=0)
    else:
        min_losses = np.inf * np.ones(target_tensor.shape[1])

    # Training the model
    print("Starting training")
    for epoch in range(start_epoch, num_epochs + 1):  # Epochs start from 1
        model.train()
        running_loss = torch.zeros((1, target_tensor.shape[1])).to(device)

        for i, data_batch in enumerate(train_loader, 0):
            inputs_batch, targets_batch = data_batch
            if not (whole_dataset_on_gpu and torch.cuda.is_available()):
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
            optimizer.zero_grad()

            if half_precision_database:
                with autocast():
                    outputs = model(inputs_batch)
                    loss_vec = loss_function(outputs, targets_batch).sum(dim=0)
                    loss = loss_vec.sum()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(inputs_batch)
                loss_vec = loss_function(outputs, targets_batch).sum(dim=0)
                loss = loss_vec.sum()
                loss.backward()
                optimizer.step()

            running_loss += loss_vec.detach()

        if cosine_annealing:
            scheduler.step()

        train_losses.append(
            running_loss.cpu().numpy().squeeze() / len(train_indices)
        )  # Since losses are summed, divide by number of samples

        # Evaluate the model on the test set
        model.eval()
        test_loss = torch.zeros((1, target_tensor.shape[1])).to(device)
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader, 0):
                inputs_batch, targets_batch = data_batch
                if not (whole_dataset_on_gpu and torch.cuda.is_available()):
                    inputs_batch = inputs_batch.to(device)
                    targets_batch = targets_batch.to(device)

                if half_precision_database:
                    with autocast():
                        outputs = model(inputs_batch)
                        loss_vec = loss_function(outputs, targets_batch).sum(dim=0)
                else:
                    outputs = model(inputs_batch)
                    loss_vec = loss_function(outputs, targets_batch).sum(dim=0)

                test_loss += loss_vec.detach()

        test_losses.append(
            test_loss.cpu().numpy().squeeze() / len(test_indices)
        )  # Since MSE is summed, divide by number of samples

        formatted_train_loss = ", ".join([f"{x:.6f}" for x in train_losses[-1]])
        formatted_test_loss = ", ".join([f"{x:.6f}" for x in test_losses[-1]])
        formatted_rms_test_loss = ", ".join(
            [
                f"{x:.6e}"
                for x in normalization_params["std_target"] * np.sqrt(test_losses[-1])
            ]
        )
        print(
            f"Epoch {epoch}/{num_epochs}, "
            f"Train Loss: [{formatted_train_loss}], "
            f"Test Loss: [{formatted_test_loss}], "
            f"RMS Test Loss: {formatted_rms_test_loss}",
        )

        # Save best components
        for i, (component, loss) in enumerate(zip(model.components, test_losses[-1])):
            if epoch >= min_epoch_save and loss < min_losses[i]:
                min_losses[i] = loss
                component_path = os.path.join(
                    partial_results_path, f"{name}_component_{i}_temp.pt"
                )
                torch.save(component.state_dict(), component_path)
                print(
                    f"Component {i + 1} of {name} saved at epoch {epoch} with test loss {loss:.6f}"
                )

        # Save checkpoint
        if epoch % checkpoint_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint for epoch {epoch}")

    print("Finished training")

    # Save training and test losses in trained_results_path
    np.savez(
        os.path.join(trained_results_path, f"{name}_losses.npz"),
        train_losses=train_losses,
        test_losses=test_losses,
    )
    print("Training and test losses saved successfully")

    # Load the saved state for each component
    for i, component in enumerate(model.components):
        component_path = os.path.join(
            partial_results_path, f"{name}_component_{i}_temp.pt"
        )
        component.load_state_dict(torch.load(component_path, weights_only=True))
        print(f"Loaded parameters for component {i} from {component_path}")

    model.normalize = True
    model.input_mean = torch.tensor(
        normalization_params["mean_input"], dtype=torch.float32
    )
    model.input_std = torch.tensor(
        normalization_params["std_input"], dtype=torch.float32
    )
    model.output_mean = torch.tensor(
        normalization_params["mean_target"], dtype=torch.float32
    )
    model.output_std = torch.tensor(
        normalization_params["std_target"], dtype=torch.float32
    )

    torch.save(model, os.path.join(trained_results_path, f"{name}.pt"))
    print(f"{name} parameters saved successfully")

    return train_losses, test_losses


def train_model_bce(
    model,
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
    decimation_factor=1,
    leave_out_fraction=0,
    whole_dataset_on_gpu=True,
    half_precision_database=False,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    batch_size=1024,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    cosine_annealing=True,
    min_epoch_save=1,
    checkpoint_freq=1,
    name="model_bce",
):
    # Define the indices for the train and test split
    num_train = len(target_tensor)
    indices = list(range(num_train))
    split = int(split_fraction * num_train)  # 80-20 split

    # Splitting the indices into train and test
    train_indices, test_indices = indices[:split], indices[split:]

    # Decimate data for experiments with reduced datasets
    train_indices = train_indices[: len(train_indices) // decimation_factor]
    test_indices = test_indices[: len(test_indices) // decimation_factor]

    # Shuffle the indices for the training/test sets
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    print("Training/test indices shuffled")

    # Drop out a fraction of the points
    train_indices = train_indices[
        int(leave_out_fraction * len(train_indices)) : len(train_indices)
    ]
    test_indices = test_indices[
        int(leave_out_fraction * len(test_indices)) : len(test_indices)
    ]

    # Check whether we are training over spatial coordinates or not
    input_train = input_tensor[train_indices] if input_tensor is not None else None
    input_test = input_tensor[test_indices] if input_tensor is not None else None

    # Load Dataset
    if whole_dataset_on_gpu and torch.cuda.is_available():
        print("Loading the entire dataset on GPU")
        dataset_device = device
    else:
        print("Loading the entire dataset on RAM")
        dataset_device = torch.device("cpu")

    train_loader = FEMDataset(
        input_train,
        target_tensor[train_indices],
        analysis_data_tensor[:, :branch_input_dim],
        analysis_data_tensor[:, branch_input_dim:],
        analysis_data_indexes[train_indices],
        analysis_data_factor[train_indices],
        batch_size,
        dataset_device,
    )
    test_loader = FEMDataset(
        input_test,
        target_tensor[test_indices],
        analysis_data_tensor[:, :branch_input_dim],
        analysis_data_tensor[:, branch_input_dim:],
        analysis_data_indexes[test_indices],
        analysis_data_factor[test_indices],
        batch_size,
        dataset_device,
    )

    # Define the loss function
    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    # Define the optimizer
    optimizer = getattr(optim, optimizer_algo)(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )

    if half_precision_database:
        scaler = GradScaler()

    # Check for checkpoint
    checkpoint_path = os.path.join(partial_results_path, name + "_latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        load_checkpoint = input(
            "Checkpoint found. Do you want to load from this checkpoint? (y/n): "
        ).lower()
        if load_checkpoint == "y":
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            train_accuracies = checkpoint["train_accuracies"]
            test_accuracies = checkpoint["test_accuracies"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        else:
            start_epoch = 1
            train_accuracies = []
            test_accuracies = []
            train_losses = []
            test_losses = []
    else:
        start_epoch = 1
        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses = []

    # Schedule a learning rate reduction
    if cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, last_epoch=start_epoch - 2
        )
    else:
        scheduler = None

    # Initiate max_accuracies
    if test_accuracies:
        max_accuracies = np.max(test_accuracies, axis=0)
        min_loss = np.min(test_losses, axis=0)
    else:
        max_accuracies = np.zeros(target_tensor.shape[1])
        min_loss = np.inf * np.ones(target_tensor.shape[1])

    # Training the model
    print("Starting training")
    for epoch in range(start_epoch, num_epochs + 1):  # Epochs start from 1
        model.train()
        correct_predictions = torch.zeros((1, target_tensor.shape[1])).to(device)
        running_loss = torch.zeros((1, target_tensor.shape[1])).to(device)

        for i, data_batch in enumerate(train_loader, 0):
            inputs_batch, targets_batch = data_batch
            if not (whole_dataset_on_gpu and torch.cuda.is_available()):
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
            optimizer.zero_grad()

            if half_precision_database:
                with autocast():
                    outputs = model(inputs_batch)
                    loss_vec = loss_function(outputs, targets_batch).sum(dim=0)
                    loss = loss_vec.sum()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(inputs_batch)
                loss_vec = loss_function(outputs, targets_batch).sum(dim=0)
                loss = loss_vec.sum()
                loss.backward()
                optimizer.step()

            # Calculate accuracy for this batch
            predictions = (outputs > 0).float()
            correct_predictions += (
                (predictions == targets_batch).float().sum(dim=0).detach()
            )
            running_loss += loss_vec.detach()

        if cosine_annealing:
            scheduler.step()

        # Average accuracies
        train_accuracies.append(
            correct_predictions.cpu().numpy().squeeze() / len(train_indices)
        )
        train_losses.append(
            running_loss.cpu().numpy().squeeze() / len(train_indices)
        )  # Since losses are summed, divide by number of samples

        # Evaluate the model on the test set
        model.eval()
        correct_predictions = torch.zeros((1, target_tensor.shape[1])).to(device)
        test_loss = torch.zeros((1, target_tensor.shape[1])).to(device)

        with torch.no_grad():
            for i, data_batch in enumerate(test_loader, 0):
                inputs_batch, targets_batch = data_batch
                if not (whole_dataset_on_gpu and torch.cuda.is_available()):
                    inputs_batch = inputs_batch.to(device)
                    targets_batch = targets_batch.to(device)

                if half_precision_database:
                    with autocast():
                        outputs = model(inputs_batch)
                else:
                    outputs = model(inputs_batch)

                # Calculate test accuracy for this batch
                predictions = (outputs > 0).float()
                correct_predictions += (
                    (predictions == targets_batch).float().sum(dim=0).detach()
                )
                test_loss += loss_vec.detach()

        test_accuracies.append(
            correct_predictions.cpu().numpy().squeeze() / len(test_indices)
        )
        test_losses.append(test_loss.cpu().numpy().squeeze() / len(test_indices))

        # Print epoch summary
        formatted_train_accuracy = ", ".join([f"{x:.6f}" for x in train_accuracies[-1]])
        formatted_test_accuracy = ", ".join([f"{x:.6f}" for x in test_accuracies[-1]])

        print(
            f"Epoch {epoch}/{num_epochs}, "
            f"Train Accuracy: [{formatted_train_accuracy}], "
            f"Test Accuracy: [{formatted_test_accuracy}]",
        )

        # # Save best components based on accuracy
        # for i, (component, accuracy) in enumerate(
        #     zip(model.components, test_accuracies[-1])
        # ):
        #     if epoch >= min_epoch_save and accuracy > max_accuracies[i]:
        #         max_accuracies[i] = accuracy
        #         component_path = os.path.join(
        #             partial_results_path, f"{name}_component_{i}_temp.pt"
        #         )
        #         torch.save(component.state_dict(), component_path)
        #         print(
        #             f"Component {i + 1} of {name} saved at epoch {epoch} with accuracy {accuracy:.4f}"
        #         )

        # Save best components based on losses
        for i, (component, loss) in enumerate(zip(model.components, test_losses[-1])):
            if epoch >= min_epoch_save and loss < min_loss[i]:
                min_loss[i] = loss
                component_path = os.path.join(
                    partial_results_path, f"{name}_component_{i}_temp.pt"
                )
                torch.save(component.state_dict(), component_path)
                print(f"Component {i + 1} of {name} saved at epoch {epoch}")

        # Save checkpoint
        if epoch % checkpoint_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_accuracies": train_accuracies,
                    "test_accuracies": test_accuracies,
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint for epoch {epoch}")

    print("Finished training")

    # Save training and test accuracies in trained_results_path
    np.savez(
        os.path.join(trained_results_path, f"{name}_accuracies.npz"),
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        train_losses=train_losses,
        test_losses=test_losses,
    )
    print("Training and test results saved successfully")

    # Load the saved state for each component
    for i, component in enumerate(model.components):
        component_path = os.path.join(
            partial_results_path, f"{name}_component_{i}_temp.pt"
        )
        component.load_state_dict(torch.load(component_path, weights_only=True))
        print(f"Loaded parameters for component {i} from {component_path}")

    model.normalize = True
    model.input_mean = torch.tensor(
        normalization_params["mean_input"], dtype=torch.float32
    )
    model.input_std = torch.tensor(
        normalization_params["std_input"], dtype=torch.float32
    )

    torch.save(model, os.path.join(trained_results_path, f"{name}.pt"))
    print(f"{name} parameters saved successfully")

    return train_accuracies, test_accuracies
