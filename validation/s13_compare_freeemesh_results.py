"""
s13_compare_freeemesh_results.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that compares the analysis metrics for increasing element size the freemesh analyses
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# # Use the latex engine for text rendering, using package newtxmath
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}"
# })

custom_palette = [
    [n / 255 for n in [186, 74, 74]],
    [n / 255 for n in [77, 136, 194]],
    [n / 255 for n in [237, 190, 94]],
]

# Load the neubernet results on the original database, for comparison
neubernet_metrics = pd.read_pickle("metrics/neubernet_global_metrics.pkl")

# Get pickle files in the freemesh_data folder
files = [
    file
    for file in os.listdir("freemesh_data")
    if ".pkl" in file and "freemesh" in file and "global" in file
]

# Extract element sizes from file names
element_sizes = [int(file.split(".")[0].split("_")[-1]) for file in files]

# Load the data
data = [pd.read_pickle(os.path.join("./freemesh_data", file)) for file in files]

# Initialize dataframes
rms_error = pd.DataFrame(index=element_sizes, columns=data[0].index)
rms_error_on_max_values = pd.DataFrame(index=element_sizes, columns=data[0].index)
max_error_on_max_values = pd.DataFrame(index=element_sizes, columns=data[0].index)

# Fill the dataframes
for i, df in enumerate(data):
    for row in df.index:
        rms_error.loc[element_sizes[i], row] = df.loc[row, "RMS error"]
        rms_error_on_max_values.loc[element_sizes[i], row] = df.loc[
            row, "RMS error on max values"
        ]
        max_error_on_max_values.loc[element_sizes[i], row] = df.loc[
            row, "Max abs error on max values"
        ]

# Sort by element size
rms_error.sort_index(inplace=True)
rms_error_on_max_values.sort_index(inplace=True)
max_error_on_max_values.sort_index(inplace=True)

# Create a similar dataframe with the RMS errors for the neubernet results
neubernet_rms_error = neubernet_metrics["RMS error"].copy().to_frame().T
neubernet_rms_error.index = ["Test\nSet"]
neubernet_rms_error_on_max_values = (
    neubernet_metrics["RMS error on max values"].copy().to_frame().T
)
neubernet_rms_error_on_max_values.index = ["Test\nSet"]
neubernet_max_error_on_max_values = (
    neubernet_metrics["Max abs error on max values"].copy().to_frame().T
)
neubernet_max_error_on_max_values.index = ["Test\nSet"]

# Append the neubernet results to the dataframes
rms_error = pd.concat([neubernet_rms_error, rms_error])
rms_error_on_max_values = pd.concat(
    [neubernet_rms_error_on_max_values, rms_error_on_max_values]
)
max_error_on_max_values = pd.concat(
    [neubernet_max_error_on_max_values, max_error_on_max_values]
)

# Convert indices to string
rms_error.index = rms_error.index.astype(str)
rms_error_on_max_values.index = rms_error_on_max_values.index.astype(str)
max_error_on_max_values.index = max_error_on_max_values.index.astype(str)

# Normalize data
# for df in [rms_error, rms_error_on_max_values, max_error_on_max_values]:
#     df /= df.iloc[0]  # Normalize over first values

# Plot all variables in a single overlaid figure
fig, ax = plt.subplots(1, 3, figsize=(4.5, 2))
plt.subplots_adjust(wspace=0.5)
colors = custom_palette
labels = [
    "RMS error (global)",
    "RMS error on max values",
    "Max abs error on max values",
]

for i, (df, label) in enumerate(
    zip([rms_error, rms_error_on_max_values, max_error_on_max_values], labels)
):
    for j, column in enumerate(df.columns):
        if j < 6:
            color = colors[0]
        elif j < 12:
            color = colors[1]
        else:
            color = colors[2]
        ax[i].plot(
            df.index,
            df[column],
            "o",
            markersize=1,
            label=f"{label} - {column}",
            color=color,
            alpha=0.5,
        )
        ax[i].plot(
            df.index,
            df[column],
            "-",
            label=f"{label} - {column}",
            color=color,
            alpha=0.5,
        )

    ax[i].set_yscale("log")

    ax[i].set_xticks([0, 1, 10])
    ax[i].set_xticks(range(2, 10), minor=True)
    ax[i].set_xticklabels(["", "1", "10"])
    ax[i].set_xlim(-0.5, 10.5)

ax[0].set_ylim(3e-6, 7e-2)
ax[1].set_ylim(3e-6, 7e-2)
ax[2].set_ylim(3e-5, 7e-1)

ax[1].set_xlabel(r"$l_\mathrm{e} / R_\mathrm{n}$", fontsize=12)
ax[0].set_title("RMSE")
ax[1].set_title("RMSE(Peak)")
ax[2].set_title("MaxAE(Peak)")
plt.tight_layout()
plt.show()

# # Save the figure
# fig.savefig("Freemesh.svg", bbox_inches="tight")
# fig.savefig("Freemesh.pdf", bbox_inches="tight")
