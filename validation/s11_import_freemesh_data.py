"""
s11_import_freemesh_data.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Script that runs import_freemesh_bcs() on each freemesh database in the "freemesh data" folder
"""

import os

from utils.helper_funcs import import_freemesh_bcs

# Get a list of folders in the "freemesh data" folder
freemesh_data_folder = "./freemesh_data"
folder_list = os.listdir(freemesh_data_folder)

# Iterate over folders
for folder in folder_list:
    # Check if the folder is a directory
    if os.path.isdir(os.path.join(freemesh_data_folder, folder)):
        print(f"Importing data in {folder}...")

        import_freemesh_bcs(os.path.join(freemesh_data_folder, folder))
