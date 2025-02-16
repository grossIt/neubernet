"""
s8_evaluate_predicted_database.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that evaluates NeuberNet predictions on the database
"""

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

evaluate_neubernet_metrics(device=device)
