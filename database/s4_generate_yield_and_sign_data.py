"""
s4_generate_yield_and_sign_data.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Generates the training data for the YieldNet and SignSVM models.
"""

import numpy as np
import yaml

# Load configuration file
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

N_data_per_analysis = config["N_yield_data"]

# Load data from database/training_data.npz
print("Loading training data...")
data = np.load("./preprocessed/training_data.npz")
inputs = data["inputs"]
analysis_data = data["analysis_data"]
print("Loading completed")

# Retrieve the analysis data indexes and factors
analysis_data_indexes = inputs[:, 0]
analysis_data_factors = inputs[:, 1]

# Initialize the training data
yield_training_data = np.zeros((3 * (N_data_per_analysis + 1) * len(analysis_data), 3))
sign_training_data = np.zeros((3 * (N_data_per_analysis + 1) * len(analysis_data), 4))

bc_index = 0
for i, analysis in enumerate(analysis_data):
    print(f"Generating yield data for analysis {i+1}")

    # Select the data points corresponding to the current analysis
    indexes_i = np.where(analysis_data_indexes == i)[0]

    # Get the number of different load factors for the current analysis
    factors = np.unique(analysis_data_factors[indexes_i])

    max_factor = np.max(factors)

    # The first column of yield_data is the analysis index, the second column is the ratio wrt to the base analysis_data,
    # the third column is the ratio between the current analysis and the one corresponding to the maximum load factor

    # Generate N_data_per_analysis random data points from 0 to 1
    random_factors = np.random.rand(N_data_per_analysis)

    yield_training_data[bc_index, 0] = i
    yield_training_data[bc_index, 1] = 0
    yield_training_data[bc_index, 2] = max_factor
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 0] = i
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 1] = (
        random_factors
    )
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 2] = (
        max_factor
    )

    bc_index += N_data_per_analysis + 1

    # Generate N_data_per_analysis random data points from 1 to max_factor
    random_factors = np.random.rand(N_data_per_analysis) * (max_factor - 1) + 1

    yield_training_data[bc_index, 0] = i
    yield_training_data[bc_index, 1] = 1
    yield_training_data[bc_index, 2] = max_factor

    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 0] = i
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 1] = (
        random_factors
    )
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 2] = (
        max_factor
    )

    bc_index += N_data_per_analysis + 1

    # Generate N_data_per_analysis random data points, logarithmically spaced between max_factor and 10 * max_factor
    random_exponents = np.random.rand(N_data_per_analysis)
    random_factors = np.power(10, random_exponents) * max_factor

    yield_training_data[bc_index, 0] = i
    yield_training_data[bc_index, 1] = max_factor
    yield_training_data[bc_index, 2] = max_factor

    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 0] = i
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 1] = (
        random_factors
    )
    yield_training_data[bc_index + 1 : bc_index + 1 + N_data_per_analysis, 2] = (
        max_factor
    )

    bc_index += N_data_per_analysis + 1


# Separate the data into inputs and targets
yield_inputs = yield_training_data[:, 0:2]  # Analysis index and load factor (for a given BC in the database...)
yield_targets = yield_training_data[:, 1:]  # Load factor and maximum load factor (...predict the factor wrt yield and the factor wrt max)

# Save the data to a file
print("Saving yield data...")
np.savez(
    "preprocessed/yield_data.npz",
    analysis_data=analysis_data,
    inputs=yield_inputs,
    targets=yield_targets,
)
print("Saving completed")

# Generate SignSVM training data
# sign_inputs is simply a stack of 4 copies of the yield_inputs, but with a *complex* dtype
sign_inputs = np.zeros((4 * len(yield_inputs), 2), dtype=np.complex64)

sign_inputs[: len(yield_inputs), 0] = yield_inputs[:, 0]
sign_inputs[: len(yield_inputs), 1] = yield_inputs[:, 1] + 1j * yield_inputs[:, 1]

sign_inputs[len(yield_inputs) : 2 * len(yield_inputs), 0] = yield_inputs[:, 0]
sign_inputs[len(yield_inputs) : 2 * len(yield_inputs), 1] = (
    -yield_inputs[:, 1] + 1j * yield_inputs[:, 1]
)

sign_inputs[2 * len(yield_inputs) : 3 * len(yield_inputs), 0] = yield_inputs[:, 0]
sign_inputs[2 * len(yield_inputs) : 3 * len(yield_inputs), 1] = (
    yield_inputs[:, 1] - 1j * yield_inputs[:, 1]
)

sign_inputs[3 * len(yield_inputs) :, 0] = yield_inputs[:, 0]
sign_inputs[3 * len(yield_inputs) :, 1] = -yield_inputs[:, 1] - 1j * yield_inputs[:, 1]

# sign_targets has size sign_inputs.shape[0] x 2
# the first column is the sign of the real part of the load factor (1 if positive, 0 if negative)
# the second column is the sign of the imaginary part of the load factor (1 if positive, 0 if negative)
sign_targets = np.zeros((sign_inputs.shape[0], 2))
sign_targets[:, 0] = np.sign(sign_inputs[:, 1].real)
sign_targets[:, 1] = np.sign(sign_inputs[:, 1].imag)
sign_targets = sign_targets.astype(np.float32)
sign_targets = sign_targets * 0.5 + 0.5

# Remove all rows where the target is [0.5, 0.5] (i.e. the load factor is zero) -> We want binary classification
zero_rows = np.where(np.all(sign_targets == 0.5, axis=1))[0]
sign_inputs = np.delete(sign_inputs, zero_rows, axis=0)
sign_targets = np.delete(sign_targets, zero_rows, axis=0)

# Save the data to a file
print("Saving sign data...")
np.savez(
    "preprocessed/sign_data.npz",
    analysis_data=analysis_data,
    inputs=sign_inputs,
    targets=sign_targets,
)
print("Saving completed")
