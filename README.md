# modelling_cyto_data

```markdown
# Classification of flow cytometry events into bacterial species using classical ML models and neural networks

This repository contains a complete pipeline for classification of flow cytometry events into bacterial species using classical ML models, such as Logistic Regression, Support Vector Machine, Random Forest and xgboost (Boosted trees). Additionally, it can train a neural network classifier on the same data using a pytorch implementation.

The project includes data preprocessing, feature filtering (using the Median Absolute Deviation), hyperparameter tuning via grid search, 5-fold cross-validation, model evaluation, and saving necessary artifacts for plot reproduction. Additional scripts are given for hyperparameter optimisation of the neural network model using optuna.

## Features

- **Data Ingestion:** Reads multiple CSV files (e.g., `species1.csv`, `species2.csv`, etc.) and combines them into a single dataset.
- **Feature Filtering:** Uses the Median Absolute Deviation (MAD) to filter numerical features.
- **Model Training:** Builds a configurable Keras neural network classifier with a variable number of hidden layers.
- **Hyperparameter Tuning:** Uses `GridSearchCV` to select optimal model parameters.
- **Cross-Validation:** Performs 5-fold cross-validation to estimate model uncertainty.
- **Artifact Saving:** Saves raw loss curves, confusion matrix data, model weights, and logs for reproducibility.
- **Plot Reproduction:** Includes a standalone script to reproduce training loss and confusion matrix plots from saved files.

## Requirements

- Python 3.7+ (tested with Python 3.12)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [TensorFlow (with Keras)](https://www.tensorflow.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [SciencePlots](https://github.com/garrettj403/SciencePlots)

Install the required packages via conda
```bash
$MAMBA_EXEC create -n pytorch python=3.12 scipy numpy matplotlib seaborn pandas polars scikit-learn
```
or pip (after installing Python):

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn polars
```
Then install xgboost via pip

```bash
pip install xgboost
```
For pytorch and pyro, also install it via pip

```bash
pip3 install torch torchvision torchaudio
pip3 install pyro-ppl
```

## Project Structure

- **`main.py`**  
  Main script that:
  - Reads and preprocesses the CSV data.
  - Filters features based on MAD.
  - Splits data into training and test sets.
  - Builds and tunes a Keras neural network using GridSearchCV.
  - Performs 5-fold cross-validation while tracking training and validation losses.
  - Saves model weights, evaluation metrics, and raw data for plot reproduction.
  
- **`reproduce_plots.py`**  
  Standalone script to reproduce the plots using saved data files:
  - Loads cross-validation loss curves from `cv_plot_data.npz`.
  - Loads the confusion matrix from `confusion_matrix.npy`.
  - Generates and saves the plots without needing to retrain the model.

- **`README.md`**  
  This documentation file.

## Usage

1. **Prepare Your Data**  
   Ensure that your CSV files (e.g., `species1.csv`, `species2.csv`, etc.) are placed in the project directory. Each CSV file should contain numeric features along with a species label.

2. **Train the Model and Save Artifacts**  
   Run the main script to train the model and save artifacts:
   
   ```bash
   python main.py
   ```
   
   This will:
   - Preprocess the data and filter features.
   - Perform grid search and cross-validation.
   - Save training/validation loss data to `cv_plot_data.npz`.
   - Save the confusion matrix to `confusion_matrix.npy`.
   - Save the final model weights and full model to `best_model_state.h5` and `best_estimator.h5`.
   - Generate and save plots (e.g., `cv_all_folds_loss.png`, `cv_mean_confidence_loss.png`, `confusion_matrix_nn.png`).
   - Log all steps in `log_steps.json`.

3. **Reproduce the Plots**  
   Use the provided script to recreate the plots from the saved data:
   
   ```bash
   python reproduce_plots.py
   ```
   
   This script generates:
   - `cv_all_folds_loss_reproduced.png`
   - `cv_mean_confidence_loss_reproduced.png`
   - `confusion_matrix_reproduced.png`

4. **Load the Model for Inference**  
   To load the saved model for predictions without retraining:
   
   ```python
   from tensorflow.keras.models import load_model
   model = load_model("best_estimator.h5")
   # Now use model.predict() as needed
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project leverages:
- [TensorFlow and Keras](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [SciencePlots](https://github.com/garrettj403/SciencePlots) for high-quality scientific plotting

Happy coding!
```
