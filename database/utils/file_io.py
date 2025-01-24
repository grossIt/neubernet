"""
file_io.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

File-import utilities
"""

import numpy as np


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
