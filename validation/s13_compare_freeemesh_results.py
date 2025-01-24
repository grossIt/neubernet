"""
s13_compare_freeemesh_results.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Scripts that compares and plots the trend in RMS errors across variables for the freeemesh results
The results have already been processed by s9_evaluate_freeemesh_predictions.py, and stored as pickle files in the
freemesh_data folder. Then, one has only to load those files, making sure that string freemesh is in them.
"""


import os
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load the neubernet results on the original database, for comparison
neubernet_metrics = pd.read_pickle("metrics/neubernet_global_metrics.pkl")

# Get pickle files in the freemesh_data folder (do not define new functions)
files = os.listdir("freemesh_data")

# Check if the files are pickle files and whether freemesh is in the name
files = [
    file for file in files if ".pkl" in file and "freemesh" in file and "global" in file
]

# Element size is the last number of the file name
element_sizes = [int(file.split(".")[0].split("_")[-1]) for file in files]

# Load the data
data = [pd.read_pickle(os.path.join("./freemesh_data", file)) for file in files]

# Create a new dataframe, with the RMS errors for each variable (variable on headers, element size on rows)
rms_error = pd.DataFrame(index=element_sizes, columns=data[0].index)

# Do the same with rms errors on max values
rms_error_on_max_values = pd.DataFrame(index=element_sizes, columns=data[0].index)

# Do the same with max errors
max_error_on_max_values = pd.DataFrame(index=element_sizes, columns=data[0].index)

# Fill the dataframe with the RMS errors
for i, df in enumerate(data):
    for row in df.index:
        rms_error.loc[element_sizes[i], row] = df.loc[row, "RMS error"]
        rms_error_on_max_values.loc[element_sizes[i], row] = df.loc[
            row, "RMS error on max values"
        ]
        max_error_on_max_values.loc[element_sizes[i], row] = df.loc[
            row, "Max abs error on max values"
        ]

# Sort the dataframe by element size
rms_error.sort_index(inplace=True)
rms_error_on_max_values.sort_index(inplace=True)
max_error_on_max_values.sort_index(inplace=True)

# Create a similar dataframe with the RMS errors for the neubernet results
neubernet_rms_error = neubernet_metrics["RMS error"].copy().to_frame().T
neubernet_rms_error.index = ["Test Set"]
neubernet_rms_error_on_max_values = (
    neubernet_metrics["RMS error on max values"].copy().to_frame().T
)
neubernet_rms_error_on_max_values.index = ["Test Set"]
neubernet_max_error_on_max_values = (
    neubernet_metrics["Max abs error on max values"].copy().to_frame().T
)
neubernet_max_error_on_max_values.index = ["Test Set"]

# Concatenate the two dataframes
rms_error = pd.concat([neubernet_rms_error, rms_error])
rms_error_on_max_values = pd.concat(
    [neubernet_rms_error_on_max_values, rms_error_on_max_values]
)
max_error_on_max_values = pd.concat(
    [neubernet_max_error_on_max_values, max_error_on_max_values]
)

# Convert the element sizes to string
rms_error.index = rms_error.index.astype(str)
rms_error_on_max_values.index = rms_error_on_max_values.index.astype(str)
max_error_on_max_values.index = max_error_on_max_values.index.astype(str)

# Print the dataframe
print(rms_error)
print(rms_error_on_max_values)
print(max_error_on_max_values)

# Plot the RMS errors, variable by variable on separate figures
for column1, column2, column3 in zip(
    rms_error.columns, rms_error_on_max_values.columns, max_error_on_max_values.columns
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        rms_error.index,
        rms_error[column1],
        "o-",
        markersize=3,
        label="RMS error (global)",
    )
    ax.plot(
        rms_error_on_max_values.index,
        rms_error_on_max_values[column2],
        "o-",
        markersize=3,
        label="RMS error on max values",
    )
    ax.plot(
        max_error_on_max_values.index,
        max_error_on_max_values[column3],
        "o-",
        markersize=3,
        label="Max abs error on max values",
    )
    ax.set_title(column1)
    ax.set_xlabel("Element size / $R_{notch}$")
    ax.set_ylabel("Error")
    ax.grid(which="both")
    ax.legend(loc="upper left")

    plt.show()
