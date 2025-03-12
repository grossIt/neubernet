[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14880154.svg)](https://doi.org/10.5281/zenodo.14880154)

# NeuberNet

*A Neural Operator for Solving Elastic-Plastic PDEs at V-Notches from Low-Fidelity Elastic Simulations*

---

## Overview

This repository contains the code and data that support the findings of the paper:

**"NeuberNet: A Neural Operator Solving Elastic-Plastic PDEs at V-Notches from Low-Fidelity Elastic Simulations"**  
by T. Grossi, M. Beghini, and M. Benedetti.

(under peer review)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/grossIt/neubernet
   cd neubernet
   ```

2. Set up the environment (choose one):

   - Using `pip` with `requirements.txt`:
     ```bash
     python3 -m venv neubernet
     source neubernet/bin/activate   # On Windows: neubernet\Scripts\activate
     pip install -r requirements.txt
     ```

   - Using `conda` with `environment.yaml`:
     ```bash
     conda env create --name neubernet --file environment.yaml
     conda activate neubernet
     ```

All machine learning scripts were run using the CUDA Toolkit 12.4 compute platform. If a compatible CUDA device is not available, training and inference will run on the CPU, though with an orders-of-magnitude increase in execution time. While this may still be acceptable for inference, full training on a CPU is strongly discouraged. Using the provided scripts, training runs at approximately 100 epochs per day on an RTX 4090 GPU.
Please note that ANSYS Mechanical must be installed to execute the APDL macros related to the FE analyses.

The FE dataset, along with all intermediate and final results, is available in this [Zenodo](https://doi.org/10.5281/zenodo.14880154) folder.
Simply unzip the dataset into the repository folder.

The `config.yaml` file centralizes all settings, allowing for easy tuning of the entire process.

---

## Usage

The scripts in the repository are named sequentially (e.g., `s1_generate_input_files.py`, `s2_ansys_database_macro.mac`, `s3_import_database.py`) to indicate the order in which they should be executed to reproduce the findings of the paper. Each script should be executed from its respective directory.

---

## Repository Structure
```
neubernet/
├── database/                 
│   ├── utils/
│   │   ├── file_io.py
│   ├── s1_generate_input_files.py
│   ├── s3_import_database.py
│   ├── s4_generate_yield_and_sign_data.py
├── fem/
│   ├── s2_ansys_database_macro.mac
│   ├── s10_ansys_database_freemesh_macro.mac
│   ├── s14_simulate_3d_shaft.mac
├── model/                
│   ├── trained/
│   │   ├── neubernet.pt
│   │   ├── neubernet_losses.pt
│   │   ├── yieldnet.pt
│   │   ├── yieldnet_losses.pt
│   │   ├── signsvm.pt
│   │   ├── signsvm_accuracies.pt
│   ├── utils/
│   │   ├── definitions.py
│   │   ├── training.py
│   ├── s5_neubernet_training.py
│   ├── s6_yieldnet_training.py
│   ├── s7_signsvm_training.py
├── validation/                 
│   ├── utils/
│   │   ├── helper_funcs.py
│   ├── s8_evaluate_predicted_database.py
│   ├── s9_plot_analysis.py
│   ├── s11_import_freemesh_data.py
│   ├── s12_evaluate_freemesh_predictions.py
│   ├── s13_compare_freeemesh_results.py
│   ├── s15_postprocess_3d_shaft.py
├── config.yaml
├── environment.yaml
├── requirements.txt
└── README.md
```

## Contact
For questions, suggestions, or collaboration opportunities, please contact:
- **T. Grossi**: [tommaso.grossi@ing.unipi.it](mailto:tommaso.grossi@ing.unipi.it)
- Open an issue on this repository.

