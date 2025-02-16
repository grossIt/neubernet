"""
s12_evaluate_freemesh_predictions.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that runs evaluate_global_metrics() on each npz freemesh database in the "freemesh data" folder
"""

import os
import pandas as pd
import sys
import torch

from utils.helper_funcs import evaluate_neubernet_metrics

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
sys.path.append("../model")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"  # Force CPU

# Get a list of npz files in the "freemesh data" folder
freemesh_data_folder = "./freemesh_data"
npz_files = [f for f in os.listdir(freemesh_data_folder) if f.endswith(".npz")]

# Remove files that contain the string neubernet
npz_files = [f for f in npz_files if "neubernet" not in f]

# Iterate over npz files
for npz_file in npz_files:
    print(f"Evaluating predictions on {npz_file}...")
    evaluate_neubernet_metrics(
        device=device,
        freemesh_data_path=os.path.join(freemesh_data_folder, npz_file),
    )
