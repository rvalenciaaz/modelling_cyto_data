# Classification of Flow Cytometry Events into Bacterial Species

This repository provides a complete pipeline to classify flow cytometry events into bacterial species using classical Machine Learning models and Neural Networks (PyTorch).

---

## Overview

The pipeline includes:
- Data preprocessing
- Feature selection using Median Absolute Deviation (MAD)
- Classical ML models: Logistic Regression, SVM, Random Forest, and XGBoost
- Neural Network model implemented with PyTorch
- Hyperparameter optimization with Optuna
- Generation of embeddings using VAE and testing the latent space via a classifier
- Confidence in classifier predictions via calibration and Bayesian neural netowrks (BNN)
- 5-fold cross-validation
- Evaluation and artifact saving for reproducibility
- Plot reproduction scripts

---

## Features

- **Data Ingestion:** Combines multiple CSV files (`species1.csv`, `species2.csv`, etc.) into one dataset.
- **Feature Filtering:** Filters numerical features using MAD.
- **Model Training:** Configurable PyTorch neural network with customizable layers.
- **Hyperparameter Tuning:** Optuna hyperparameter optimization for both classical models and neural networks.
- **Cross-Validation:** Estimates uncertainty with 5-fold cross-validation.
- **Artifact Saving:** Saves loss curves, confusion matrices, model weights, and logs.
- **Plot Reproduction:** Standalone scripts to reproduce loss and confusion matrix plots.
  
---

## Requirements

- Python 3.7+ (Tested on Python 3.12)

### Python Libraries

| Library                                              | Installation                                |
|------------------------------------------------------|---------------------------------------------|
| [NumPy](https://numpy.org/)                          | `pip install numpy`                         |
| [Pandas](https://pandas.pydata.org/)                 | `pip install pandas`                        |
| [SciPy](https://www.scipy.org/)                      | `pip install scipy`                         |
| [scikit-learn](https://scikit-learn.org/)            | `pip install scikit-learn`                  |
| [Matplotlib](https://matplotlib.org/)                | `pip install matplotlib`                    |
| [Seaborn](https://seaborn.pydata.org/)               | `pip install seaborn`                       |
| [Polars](https://www.pola.rs/)                       | `pip install polars`                        |

### Conda Environment (optional)

```bash
mamba create -n pytorch python=3.12 scipy numpy matplotlib seaborn pandas polars scikit-learn
```
### Additional packages to install via pip

| Library                                              | Installation                                |
|------------------------------------------------------|---------------------------------------------|
| [XGBoost](https://xgboost.readthedocs.io/)           | `pip install xgboost`                       |
| [PyTorch](https://pytorch.org/)                      | `pip install torch torchvision torchaudio`  |
| [Pyro](https://pyro.ai/)                             | `pip install pyro-ppl`                      |
| [SciencePlots](https://github.com/garrettj403/SciencePlots) | `pip install SciencePlots`                  |
| [Optuna]()                                           | `pip install optuna`                        |

---

## Project Structure

There are several scripts that execute different pipelines:

The main* scripts perform the training of the models

- `.py`: Pipeline for training of a multi-class classifier using a MLP. The hyperparameters are optimised with Optuna, including 
- `.py`: Pipeline for training of a multi-class classifier using a Bayesian neural network (MLP). The networks outputs a posterior predictive distribution for each of the classes.
- `.py`: Pipeline for training of a multi-class classifier using classical machine learning models. 
- `.py`: Pipeline for training of a multi-class classifier using a NN based on the TabTransformer architecture. 

The scripts take the channel values csv from pure culture run in a format (`species1.csv`, `species2.csv`, ...), as it will be specified later.

To do inference (prediction) on a new dataset, use the inference* scripts:


- `.py`: Multi-class prediction using a MLP 
- `.py`: Multi-class prediction using a Bayesian neural network (MLP). The networks outputs a posterior predictive distribution for each of the classes.
- `.py`: Multi-class prediction using the classical models (Logistic classification, SVM, Random Forests, xgboost)
- `.py`: Multi-class prediction using the TabTransformer architecture.


---

## Usage

### 1. Prepare Data
There are two ways of preparing the data:

2) The events can be exported as channel values in csv format using FlowJo. Place CSV files with the following format (`species1.csv`, `species2.csv`, etc.) in the project directory.

### 2a. Train the Model (main_scripts)
```bash
python main.py
```

**This script:**
- Preprocesses data and selects features
- Tunes hyperparameters (GridSearchCV, Optuna)
- Performs cross-validation
- Saves artifacts (loss curves, confusion matrix, models, logs)
- Generates and saves plots

Artifacts generated:
- `cv_plot_data.npz`
- `confusion_matrix.npy`
- `best_model_state.h5`
- `best_estimator.h5`
- Plots (`cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`)
- Log file (`log_steps.json`)

### 2b. Train the Model (project folder)

```bash
python main_.py
python main_.py
python main_.py
python main_.py
```

**This script:**
- Preprocesses data and selects features
- Tunes hyperparameters (Optuna)
- Performs cross-validation
- Saves artifacts (loss curves, confusion matrix, models, logs)
- Generates and saves plots

Artifacts generated by python main_.py:
- `cv_plot_data.npz`
- `confusion_matrix.npy`
- `best_model_state.h5`
- `best_estimator.h5`
- Plots (`cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`)
- Log file (`log_steps.json`)

Artifacts generated by python main_.py:
- `cv_plot_data.npz`
- `confusion_matrix.npy`
- `best_model_state.h5`
- `best_estimator.h5`
- Plots (`cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`)
- Log file (`log_steps.json`)

Artifacts generated by python main_.py:
- `cv_plot_data.npz`
- `confusion_matrix.npy`
- `best_model_state.h5`
- `best_estimator.h5`
- Plots (`cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`)
- Log file (`log_steps.json`)

Artifacts generated by python main_.py:
- `cv_plot_data.npz`
- `confusion_matrix.npy`
- `best_model_state.h5`
- `best_estimator.h5`
- Plots (`cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`)
- Log file (`log_steps.json`)

### 3. Reproduce Plots
To ensure reproducibility, the plots can be recreated from saved data:

```bash
python reproduce_plots_.py
python reproduce_plots_.py
python reproduce_plots_.py
python reproduce_plots_.py
```

Generated plots by :
- `cv_all_folds_loss_reproduced.png`
- `cv_mean_confidence_loss_reproduced.png`
- `confusion_matrix_reproduced.png`

### 4. Load model for inference and perform prediction

Load saved model for predictions in a new dataset:

```python
python inference_.py
python inference_.py
python inference_.py
python inference_.py
```

---

## Results

### Classification with dataset of 9 species and 100k events per specie.



### Classification with dataset of 9 species and 1m* events per specie.

---
## License

Distributed under the  License. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

We would like to acknowledge

---

