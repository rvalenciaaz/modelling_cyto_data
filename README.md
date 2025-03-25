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

There are several scripts that execute different pipelines, these are located in the main_scripts folder:

- ``: Pipeline for
- ``: Pipeline for
- ``: Pipeline for
- ``: Pipeline for
- ``: Pipeline for
- ``: Pipeline for
The scripts take the channel values csv from pure culture run in a format (`species1.csv`, `species2.csv`, ...), as it will be specified later.

In addition, these scripts are consolidated into a project located in the folder ``

---

## Usage

### 1. Prepare Data
There are two ways of preparing the data:

2) The events can be exported as channel values in csv format using FlowJo. Place CSV files with the following format (`species1.csv`, `species2.csv`, etc.) in the project directory.

### 2. Train the Model
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

### 3. Reproduce Plots
To recreate plots from saved data:

```bash
python reproduce_plots.py
```

Generated plots:
- `cv_all_folds_loss_reproduced.png`
- `cv_mean_confidence_loss_reproduced.png`
- `confusion_matrix_reproduced.png`

### 4. Load Model for Inference

Load saved model for predictions:

```python
from tensorflow.keras.models import load_model

model = load_model("best_estimator.h5")
predictions = model.predict(data)
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

- [TensorFlow and Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [SciencePlots](https://github.com/garrettj403/SciencePlots)

---

Happy coding!
