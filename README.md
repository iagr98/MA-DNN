# Data-Driven Models for DPZ and λ Prediction

This repository contains the datasets, trained models, and Jupyter notebooks used to develop and evaluate two deep neural network (DNN) models for predicting **separator performance indicators** — namely **DPZ** and **λ**.  
The repository is organized to ensure reproducibility and transparency of the data-driven modeling workflow.


---

## Input Data

The folder **`Input/`** contains all datasets used for model construction and validation:

- **`df_dpz.csv`** – Primary dataset used to train and test the DNN model for predicting DPZ.  
- **`df_lam.csv`** – Primary dataset used to train and test the DNN model for predicting λ.  
- **`sz.csv`** – Supplemental dataset obtained from RWTH Aachen University (SZ dataset).  
- **`ye.csv`** – Supplemental dataset obtained from TU Berlin (YE dataset).  

These datasets include experimental and simulated data used to train, validate, and benchmark the models.

---

## Trained Models and Results

The folder **`saved_models/`** stores the trained neural networks, associated metadata, and Optuna optimization results:

- **`dnn_(dpz)/`**  
  Contains the trained model **`dnn_dpz.pt`**, all intermediate data used during training, and the **Optuna** study results from the hyperparameter optimization phase.

- **`dnn_(lam)/`**  
  Contains the trained model **`dnn_lam.pt`**, along with the corresponding training data and Optuna results.


---

## Jupyter Notebooks

Two Jupyter notebooks guide the entire model development workflow:

- **`dnn_dpz.ipynb`**  
  Implements the complete pipeline for the DPZ model, including data preprocessing, network definition, training, validation, testing, and visualization of results.

- **`dnn_lam.ipynb`**  
  Implements the corresponding pipeline for the λ model, following the same structure and methodology as the DPZ notebook.

Both notebooks are self-contained and can be executed independently once the datasets are available in the `Input/` directory.

