# HVAC Predictive Maintenance

This project explores machine learning for predictive maintenance of HVAC chiller equipment. It uses sensor readings to predict multiple fault labels, with exploratory data analysis (EDA), a Random Forest baseline, and a multilayer perceptron (MLP).

## Project plan

1. Load and inspect the chiller maintenance dataset.
2. Explore sensor features and fault labels through EDA.
3. Train and evaluate a multi-output Random Forest model.
4. Train and evaluate a multi-output MLP model.
5. Compare model performance on the provided holdout split.
6. Save trained models and any preprocessing objects needed for inference.

## Dataset

The notebooks use `chiller_predictive_maintenance_dataset.csv` in the project directory. The dataset includes a `Split` column; the notebooks use rows marked `train` for model development and `holdout_manual` for final holdout evaluation.

The notebooks expect the CSV at the repository root. If you keep the dataset outside the repository, update the CSV path in each notebook accordingly.

## Models

### Random Forest

The Random Forest notebook uses `RandomForestRegressor` with 17 sensor inputs and 26 output columns. It explores model settings and reports metrics including R², mean absolute error (MAE), and root mean squared error (RMSE). The final model is fitted on the training split and evaluated on the manual holdout split.

### Multilayer perceptron (MLP)

The MLP notebook uses a fully connected neural network with two hidden layers (64 and 32 ReLU units) and a 26-value linear output layer. Sensor inputs are standardized with `StandardScaler`. The scaler must be saved and reused with the trained MLP when making predictions.

Both notebooks treat the 26 outputs as numeric prediction targets. The reported metrics should be reviewed alongside the label definitions when interpreting predictions.

## Repository structure

```text
.
├── chiller_predictive_maintenance_dataset.csv
├── hvac_dataset.ipynb       # EDA and Random Forest experiments
├── Idea_Lab_MLP.ipynb       # MLP experiments
├── models/                  # Saved model and preprocessing artifacts (planned)
└── README.md
```

## Running the notebooks

The notebooks were developed for Google Colab and currently contain Google Drive mounting and data-loading cells. To run them locally, install the required Python packages and change the data-loading cells to read the CSV from the project directory.

The notebooks use Python libraries including pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and TensorFlow/Keras.

## Saved artifacts

The `models/` directory is intended for trained model files. Include the MLP's fitted `StandardScaler` with its model so inference applies the same feature transformation used during training. Add instructions for loading the saved artifacts when they are committed.