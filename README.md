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

Please note that an installation of ANSYS Mechanical is required to execute the APDL macros related to the FE analyses.

---

## Usage

The scripts in the repository are named sequentially (e.g., `s1_generate_input_files.py`, `s2_ansys_database_macro.mac`, `s3_import_database.py`) to indicate the order in which they should be executed to reproduce the findings of the paper.

---

## Repository Structure
```
neubernet/
├── database/                 
│   ├── utils
│   │   ├── file_io.py
│   ├── s1_generate_input_files.py
│   ├── s3_import_database.py
│   ├── s4_generate_yield_and_sign_data.py
├── fem/
│   ├── s2_ansys_database_macro.mac
│   ├── s10_ansys_database_freemesh_macro.mac
│   ├── s14_simulate_3d_shaft.mac
├── model/                
│   ├── trained
│   │   ├── neubernet.pt
│   │   ├── neubernet_losses.pt
│   │   ├── yieldnet.pt
│   │   ├── yieldnet_losses.pt
│   │   ├── signsvm.pt
│   │   ├── signsvm_accuracies.pt
│   ├── utils
│   │   ├── definitions.py
│   │   ├── training.py
│   ├── s5_neubernet_training.py
│   ├── s8_yieldnet_training.py
│   ├── s9_signsvm_training.py
├── validation/                 
│   ├── utils
│   │   ├── helper_funcs.py
│   ├── s6_evaluate_predicted_database.py
│   ├── s7_plot_analysis.py
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

