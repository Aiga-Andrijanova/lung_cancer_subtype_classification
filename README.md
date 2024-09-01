# Lung Cancer Subtype Classification

This repository contains the code implementation for the machine learning methods used in my master's thesis "Lung cancer subtype analysis" at Imperial College London.

## Project Overview

This project investigates the potential of integrating biophysical measurements from Atomic Force Microscopy with traditional clinical and histopathological data to enhance lung cancer subtype classification. The study utilizes the novel Lung Cancer Physical Properties (LCPP) dataset and the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Contents

- `exploratory_data_analysis/biophysical_data_EDA.ipynb`: Jupyter notebook containing exploratory data analysis of the biophysical features in the LCPP dataset. This includes visualization of distributions, correlation analysis, and statistical tests for AFM-derived measurements.

- `exploratory_data_analysis/clinical_data_EDA.ipynb`: Jupyter notebook focused on the exploratory analysis of clinical and histopathological features in the LCPP dataset. 
- `modeling/BN_on_LCPP.ipynb`: Jupyter notebook implementing Bayesian Network models on the LCPP dataset. This includes structure learning, parameter estimation, and evaluation of BN performance for lung cancer subtype classification.
- `modeling/BN_on_WDBC.ipynb`: Jupyter notebook applying Bayesian Network models to the Wisconsin Diagnostic Breast Cancer dataset. It serves as a benchmark and comparison for BN performance on a well-established dataset.
- `modeling/ML_gridsearch_on_LCPP.py`: Python script performing grid search for hyperparameter optimization of various machine learning models (e.g., SVM, Random Forest, Logistic Regression) on the LCPP dataset. It includes feature selection, cross-validation, and performance evaluation.
- `modeling/ML_gridsearch_on_WDBC.py`: Python script conducting grid search and model evaluation on the WDBC dataset. This script validates the implemented machine learning techniques and provides a performance baseline for comparison with the LCPP results.

## Key Features

- Implementation of various machine learning models (Logistic Regression, SVM, Random Forest, etc.)
- Bayesian Network implementation with structure learning
- Feature selection and dimensionality reduction techniques
- Cross-validation methods (10-fold CV and LOOCV)
- Comprehensive evaluation metrics
- Visualization of results

## Usage

1. Clone the repository
2. Install the required dependencies: `pip install -r /.devcontainer/requirements.txt` or use the provided devcontainer
