#!/usr/bin/env python
# coding: utf-8
import scipy.io
import os
import sys
import h5py
import numpy as np
from scipy.interpolate import interp1d
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sbi import utils as sbi_utils
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- 1. Data Loading and Function Definitions ---
name_format = "GBM_Hist_run20_new"
print("Note: use results33.mat as train data, results37.m as test data; B only has positive Z values; Outliar; 100, 0.12, 14")

train_file_path = os.path.join('MATLAB_DATA', 'results33.mat')  # Clean data for training
test_file_path = os.path.join('MATLAB_DATA', 'results37.mat')   # Noisy data for testing

# (All function definitions from your original code are kept here)
def generate_filename(base_name, extension):
    return f"{name_format}_{base_name}.{extension}"


h5_file_numbers = {3, 6, 7, 9}


# Function to load data from .h5 file
def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as hdf:
        results_group = hdf['results']
        B_MAG_refs = results_group['B_MAG'][()]
        EDC_MAG_refs = results_group['EDC_MAG'][()]
        X_refs = results_group['X'][()]
        I_refs = results_group['I'][()]
        B_MAG_raw = [hdf[ref][()] for ref in B_MAG_refs.flatten()]
        EDC_MAG_raw = [hdf[ref][()] for ref in EDC_MAG_refs.flatten()]
        X_raw = [hdf[ref][()] for ref in X_refs.flatten()]
        I_raw = [hdf[ref][()] for ref in I_refs.flatten()]
    return B_MAG_raw, EDC_MAG_raw, X_raw, I_raw


# Function to load data from .mat file
def load_mat_data(file_path):
    mat_contents = scipy.io.loadmat(file_path)
    results = mat_contents['results']
    B_MAG_raw = [result['B_MAG'][0] for result in results[0]]
    EDC_MAG_raw = [result['EDC_MAG'][0] for result in results[0]]
    X_raw = [result['X'][0] for result in results[0]]  # Scaling X values
    I_raw = [result['I'][0] for result in results[0]]
    return B_MAG_raw, EDC_MAG_raw, X_raw, I_raw


# Function to resample and smooth data
def resample_and_smooth(X, I, new_length):
    X_new = np.linspace(6562.3, 6563.3, num=new_length)
    f = interp1d(X, I, kind='cubic', fill_value="extrapolate")
    I_new = f(X_new)
    return X_new, I_new


# Function to process data
def process_data(B_MAG_raw, EDC_MAG_raw, X_raw, I_raw, fixed_length=1200):
    X_data, I_data, B_MAG_data, EDC_MAG_data = [], [], [], []
    for i in range(len(X_raw)):
        X = np.squeeze(X_raw[i])
        I = np.squeeze(I_raw[i])
        B = np.squeeze(B_MAG_raw[i])
        EDC = np.squeeze(EDC_MAG_raw[i])
        X_scaled = X * 10**10
        X_resampled, I_resampled = resample_and_smooth(X_scaled, I, fixed_length)
        X_data.append(X_resampled)
        I_data.append(I_resampled)
        B_MAG_data.append(B)
        EDC_MAG_data.append(EDC)
    return np.array(B_MAG_data), np.array(EDC_MAG_data), np.array(X_data), np.array(I_data)

# Main function to determine the type of file and load data accordingly
def load_and_process_file(file_path):
    file_name = os.path.basename(file_path)
    
    # Extract the file number (e.g., results1, results5, etc.)
    file_number = int(file_name.lstrip('results').rstrip('.mat'))
    
    if file_number in h5_file_numbers:
        # It's an HDF5 file
        print(f"Processing {file_name} as an HDF5 file...", flush=True)
        B_MAG_raw, EDC_MAG_raw, X_raw, I_raw = load_h5_data(file_path)
    else:
        # It's a standard .mat file
        print(f"Processing {file_name} as a .mat file...", flush=True)
        B_MAG_raw, EDC_MAG_raw, X_raw, I_raw = load_mat_data(file_path)
    
    # Process the loaded data
    B_MAG_data, EDC_MAG_data, X_data, I_data = process_data(B_MAG_raw, EDC_MAG_raw, X_raw, I_raw)
    
    return B_MAG_data, EDC_MAG_data, X_data, I_data

# Load the full clean dataset (for training) and the noisy dataset (for testing)
print("Loading clean data for training...")
B_MAG_data_clean, EDC_MAG_data_clean, _, I_data_clean = load_and_process_file(train_file_path)
print("Loading noisy data for testing...")
B_MAG_data_noisy, EDC_MAG_data_noisy, _, I_data_noisy = load_and_process_file(test_file_path)

# --- 2. Create Raw Input/Target Arrays ---
target_data_clean_raw = np.column_stack((B_MAG_data_clean[:, 2], EDC_MAG_data_clean))
target_data_test_noisy_raw = np.column_stack((B_MAG_data_noisy[:, 2], EDC_MAG_data_noisy))

input_data_clean_raw = I_data_clean
input_data_test_noisy_raw = I_data_noisy

# --- 3. Split the CLEAN dataset to create the final training set ---
# The test set from this split is discarded, as we are using the separate noisy file for testing.
input_train_raw, _, target_train_raw, _ = train_test_split(
    input_data_clean_raw, target_data_clean_raw, test_size=0.2, random_state=42
)
print(f"Using {len(input_train_raw)} clean samples for training.")
print(f"Using {len(input_data_test_noisy_raw)} noisy samples for testing.")

# --- 4. Input Data Preprocessing (Z-Score Normalization) ---

# Calculate std dev for filtering *only* from the clean training set
stds_train = np.std(input_train_raw, axis=0)
start_index, end_index = 0, input_train_raw.shape[1] - 1
for i in range(input_train_raw.shape[1]):
    if stds_train[i] >= 0.01:
        start_index = i
        break
for i in range(input_train_raw.shape[1] - 1, -1, -1):
    if stds_train[i] >= 0.01:
        end_index = i
        break
print(f"Data will be filtered from column {start_index} to {end_index}.")

# Apply the same filtering window to both the clean training data and noisy test data
I_train_filtered = input_train_raw[:, start_index:end_index + 1]
I_test_filtered = input_data_test_noisy_raw[:, start_index:end_index + 1]

# Calculate mean and std for normalization *only* from the filtered clean training set
means_train_filtered = np.mean(I_train_filtered, axis=0)
stds_train_filtered = np.std(I_train_filtered, axis=0)

# Normalize both datasets using parameters from the clean training set
normalized_I_train = (I_train_filtered - means_train_filtered) / stds_train_filtered
normalized_I_test = (I_test_filtered - means_train_filtered) / stds_train_filtered

# Final interpolation to ensure fixed length of 1200
def interpolate_to_fixed_length(data, new_length=1200):
    resampled_data = []
    for row in data:
        x_original = np.linspace(0, 1, num=len(row))
        x_new = np.linspace(0, 1, num=new_length)
        f = interp1d(x_original, row, kind='cubic', fill_value="extrapolate")
        resampled_data.append(f(x_new))
    return np.array(resampled_data)

input_data_train = interpolate_to_fixed_length(normalized_I_train)
input_data_test = interpolate_to_fixed_length(normalized_I_test)
print("Input data preprocessing complete.")

# --- 5. Target Data Preprocessing (MinMax Scaling) ---

# Separate the components for clean training and noisy test sets
B3_train, E1_train, E2_train, E3_train = target_train_raw[:,0:1], target_train_raw[:,1:2], target_train_raw[:,2:3], target_train_raw[:,3:4]
B3_test, E1_test, E2_test, E3_test = target_data_test_noisy_raw[:,0:1], target_data_test_noisy_raw[:,1:2], target_data_test_noisy_raw[:,2:3], target_data_test_noisy_raw[:,3:4]

# Initialize scalers
scaler_B3 = MinMaxScaler(feature_range=(0, 1))
scaler_E1 = MinMaxScaler(feature_range=(0, 1))
scaler_E2 = MinMaxScaler(feature_range=(0, 1))
scaler_E3 = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the CLEAN TRAINING data
B3_scaled_train = scaler_B3.fit_transform(B3_train)
E1_scaled_train = scaler_E1.fit_transform(E1_train)
E2_scaled_train = scaler_E2.fit_transform(E2_train)
E3_scaled_train = scaler_E3.fit_transform(E3_train)

# ONLY transform the NOISY TEST data
B3_scaled_test = scaler_B3.transform(B3_test)
E1_scaled_test = scaler_E1.transform(E1_test)
E2_scaled_test = scaler_E2.transform(E2_test)
E3_scaled_test = scaler_E3.transform(E3_test)

# Recombine the scaled components
target_data_train = np.hstack((B3_scaled_train, E1_scaled_train, E2_scaled_train, E3_scaled_train))
target_data_test = np.hstack((B3_scaled_test, E1_scaled_test, E2_scaled_test, E3_scaled_test))
print("Target data preprocessing complete.")



from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor




gbm = HistGradientBoostingRegressor(
    max_iter=120,            # Maximum number of boosting iterations
    learning_rate=0.08,      # Learning rate
    max_depth=10,             # Maximum depth of the tree
    validation_fraction=0.2, # Fraction of the data to use for validation
    min_samples_leaf=10,       # Increase minimum samples per leaf
    max_bins=128,            # Limit bins for feature quantization
    max_leaf_nodes=31,       # Limit the number of leaf nodes in each tree
    l2_regularization=2.0,     # Higher regularization to prevent overfitting
    early_stopping=True,     # Early stopping flag
    tol=1e-4,                # Tolerance for stopping
    verbose=1,                # To monitor the training process
    random_state=42,
    loss='least_squares'
)



import joblib


# Define the path where the model was saved
model_load_path = os.path.join("/home/botingl/machine learning copy/GBM_Hist_noise.joblib")


# Load the model using joblib
model = joblib.load(model_load_path)
print(f"Model loaded from {model_load_path}")


# Whole model evaluation (NMSE, rMAE, R²)
def evaluate_whole_model(y_true, y_pred):
    nmse = mean_squared_error(y_true, y_pred) / np.var(y_true)
    rmae = mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))
    r2 = r2_score(y_true, y_pred)
    return nmse, rmae, r2


# Component-wise evaluation (MSE, MAE, R²)
def evaluate_components(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


# Evaluation metrics for the training data
print("Start to evaluate the training set.", flush=True)
import time

# Measure the time taken for the entire prediction process
start_time = time.time()

# Make predictions on the training data
predictions_train = model.predict(input_data_train)

# Measure the end time
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"Time taken for prediction on the entire training set: {time_taken:.6f} seconds")

num_input = len(input_data_train)
time_each = time_taken/num_input
print(f"Time taken for prediction on each training set: {time_each:.6f} seconds")


# Scale back the predicted values to original range
B_pred_train = scaler_B3.inverse_transform(predictions_train[:, 0].reshape(-1, 1))
E_pred_train = np.column_stack((
    scaler_E1.inverse_transform(predictions_train[:, 1].reshape(-1, 1)),
    scaler_E2.inverse_transform(predictions_train[:, 2].reshape(-1, 1)),
    scaler_E3.inverse_transform(predictions_train[:, 3].reshape(-1, 1))
))


# Scale back the true values to original range
B_true_train = scaler_B3.inverse_transform(target_data_train[:, 0].reshape(-1, 1))
E_true_train = np.column_stack((
    scaler_E1.inverse_transform(target_data_train[:, 1].reshape(-1, 1)),
    scaler_E2.inverse_transform(target_data_train[:, 2].reshape(-1, 1)),
    scaler_E3.inverse_transform(target_data_train[:, 3].reshape(-1, 1))
))

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have already calculated the following:
# B_true_test, B_pred_test, E_true_test, E_pred_test

# Step 1: Calculate the errors for B3, E1, E2, E3 components
B3_error_train = B_true_train - B_pred_train
E1_error_train = E_true_train[:, 0] - E_pred_train[:, 0]
E2_error_train = E_true_train[:, 1] - E_pred_train[:, 1]
E3_error_train = E_true_train[:, 2] - E_pred_train[:, 2]

# Step 2: Plot the error histograms
plt.figure(figsize=(18, 4))  # Adjust figure size to match the layout in your image

# Plot B3 error
plt.subplot(1, 4, 1)
plt.hist(B3_error_train, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$|\mathbf{B}|_{\mathrm{NN}} - |\mathbf{B}|_{\mathrm{Training}}$', fontsize=14)
plt.ylabel('Counts', fontsize=14)
# plt.xlim([-0.2, 0.2])  # Adjust limits to match your plot
plt.grid(True)

# Plot E1 error
plt.subplot(1, 4, 2)
plt.hist(E1_error_train, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_x^{\mathrm{NN}} - E_x^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Plot E2 error
plt.subplot(1, 4, 3)
plt.hist(E2_error_train, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_y^{\mathrm{NN}} - E_y^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Plot E3 error
plt.subplot(1, 4, 4)
plt.hist(E3_error_train, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_z^{\mathrm{NN}} - E_z^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# Combine the scaled-back true and predicted values for evaluation

true_train_combined = np.column_stack((B_true_train, E_true_train))
pred_train_combined = np.column_stack((B_pred_train, E_pred_train))


# Whole model evaluation

nmse_train, rmae_train, r2_whole_model_train = evaluate_whole_model(true_train_combined, pred_train_combined)
print(f"Whole Model NMSE: {nmse_train}, rMAE: {rmae_train}, R²: {r2_whole_model_train}")


# Component-wise evaluation

mse_train_B3, mae_train_B3, r2_train_B3 = evaluate_components(B_true_train, B_pred_train)
mse_train_E1, mae_train_E1, r2_train_E1 = evaluate_components(E_true_train[:, 0], E_pred_train[:, 0])
mse_train_E2, mae_train_E2, r2_train_E2 = evaluate_components(E_true_train[:, 1], E_pred_train[:, 1])
mse_train_E3, mae_train_E3, r2_train_E3 = evaluate_components(E_true_train[:, 2], E_pred_train[:, 2])


# Output the evaluation for each component

print(f"B3: MSE = {mse_train_B3}, MAE = {mae_train_B3}, R² = {r2_train_B3}")
print(f"E1: MSE = {mse_train_E1}, MAE = {mae_train_E1}, R² = {r2_train_E1}")
print(f"E2: MSE = {mse_train_E2}, MAE = {mae_train_E2}, R² = {r2_train_E2}")
print(f"E3: MSE = {mse_train_E3}, MAE = {mae_train_E3}, R² = {r2_train_E3}")


# Save evaluation metrics to a text file

metrics_train_file = os.path.join("/home/botingl/machine learning copy/results", generate_filename("evaluation_metrics_train", "txt"))
with open(metrics_train_file, "w") as f:
    f.write(f"Whole Model: NMSE: {nmse_train}, rMAE: {rmae_train}, R²: {r2_whole_model_train}\n")
    f.write(f"B3: MSE = {mse_train_B3}, MAE = {mae_train_B3}, R² = {r2_train_B3}\n")
    f.write(f"E1: MSE = {mse_train_E1}, MAE = {mae_train_E1}, R² = {r2_train_E1}\n")
    f.write(f"E2: MSE = {mse_train_E2}, MAE = {mae_train_E2}, R² = {r2_train_E2}\n")
    f.write(f"E3: MSE = {mse_train_E3}, MAE = {mae_train_E3}, R² = {r2_train_E3}\n")


# Plot the true vs. predicted B and E values for the training set

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(B_true_train, B_pred_train, alpha=0.5)
plt.plot([-0.05, 1.1], [-0.05, 1.1], 'g--')
plt.xlim(-0.05, 0.35)
plt.ylim(-0.05, 0.35)
plt.xlabel('True B3')
plt.ylabel('Predicted B3')
plt.title('True vs. Predicted B3 (Train)')

plt.subplot(2, 2, 2)
plt.scatter(E_true_train[:, 0], E_pred_train[:, 0], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E1')
plt.ylabel('Predicted E1')
plt.title('True vs. Predicted E1 (Train)')

plt.subplot(2, 2, 3)
plt.scatter(E_true_train[:, 1], E_pred_train[:, 1], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E2')
plt.ylabel('Predicted E2')
plt.title('True vs. Predicted E2 (Train)')

plt.subplot(2, 2, 4)
plt.scatter(E_true_train[:, 2], E_pred_train[:, 2], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E3')
plt.ylabel('Predicted E3')
plt.title('True vs. Predicted E3 (Train)')

plt.suptitle(f'{name_format}_train', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to avoid overlap with suptitle
figure_test_file = os.path.join("/home/botingl/machine learning copy/results", generate_filename("true_vs_predictions_train", "png"))
plt.savefig(figure_test_file, dpi=300, facecolor='white')
plt.show()

print("Training set evaluation done.", flush=True)


# Evaluate the model on the testing set
print("Start to evaluate the testing set.", flush=True)

import time

# Measure the time taken for the entire prediction process
start_time = time.time()

# Make predictions on the training data
predictions_test = model.predict(input_data_test)

# Measure the end time
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"Time taken for prediction on the entire testing set: {time_taken:.4f} seconds")

um_input = len(input_data_test)
time_each = time_taken/num_input
print(f"Time taken for prediction on each testing set: {time_each:.6f} seconds")


# Scale back the predicted values to original range for the test data
B_pred_test = scaler_B3.inverse_transform(predictions_test[:, 0].reshape(-1, 1))

E_pred_test = np.column_stack((
    scaler_E1.inverse_transform(predictions_test[:, 1].reshape(-1, 1)),
    scaler_E2.inverse_transform(predictions_test[:, 2].reshape(-1, 1)),
    scaler_E3.inverse_transform(predictions_test[:, 3].reshape(-1, 1))
))


# Scale back the true values to original range for the test data

B_true_test = scaler_B3.inverse_transform(target_data_test[:, 0].reshape(-1, 1))

E_true_test = np.column_stack((
    scaler_E1.inverse_transform(target_data_test[:, 1].reshape(-1, 1)),
    scaler_E2.inverse_transform(target_data_test[:, 2].reshape(-1, 1)),
    scaler_E3.inverse_transform(target_data_test[:, 3].reshape(-1, 1))
))


import numpy as np
import matplotlib.pyplot as plt

# Assuming you have already calculated the following:
# B_true_test, B_pred_test, E_true_test, E_pred_test

# Step 1: Calculate the errors for B3, E1, E2, E3 components
B3_error = B_true_test - B_pred_test
E1_error = E_true_test[:, 0] - E_pred_test[:, 0]
E2_error = E_true_test[:, 1] - E_pred_test[:, 1]
E3_error = E_true_test[:, 2] - E_pred_test[:, 2]

# Step 2: Plot the error histograms
plt.figure(figsize=(18, 4))  # Adjust figure size to match the layout in your image

# Plot B3 error
plt.subplot(1, 4, 1)
plt.hist(B3_error, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$|\mathbf{B}|_{\mathrm{NN}} - |\mathbf{B}|_{\mathrm{Training}}$', fontsize=14)
plt.ylabel('Counts', fontsize=14)
# plt.xlim([-0.2, 0.2])  # Adjust limits to match your plot
plt.grid(True)

# Plot E1 error
plt.subplot(1, 4, 2)
plt.hist(E1_error, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_x^{\mathrm{NN}} - E_x^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Plot E2 error
plt.subplot(1, 4, 3)
plt.hist(E2_error, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_y^{\mathrm{NN}} - E_y^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Plot E3 error
plt.subplot(1, 4, 4)
plt.hist(E3_error, bins=30, color='orange', alpha=0.7)
plt.xlabel(r'$E_z^{\mathrm{NN}} - E_z^{\mathrm{Training}}$', fontsize=14)
# plt.xlim([-0.2, 0.2])
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# Combine the scaled-back true and predicted values for evaluation (test data)

true_test_combined = np.column_stack((B_true_test, E_true_test))
pred_test_combined = np.column_stack((B_pred_test, E_pred_test))


# Whole model evaluation

nmse_test, rmae_test, r2_whole_model_test = evaluate_whole_model(true_test_combined, pred_test_combined)
print(f"Whole Model NMSE: {nmse_test}, rMAE: {rmae_test}, R²: {r2_whole_model_test}")


# Component-wise evaluation

mse_test_B3, mae_test_B3, r2_test_B3 = evaluate_components(B_true_test, B_pred_test)
mse_test_E1, mae_test_E1, r2_test_E1 = evaluate_components(E_true_test[:, 0], E_pred_test[:, 0])
mse_test_E2, mae_test_E2, r2_test_E2 = evaluate_components(E_true_test[:, 1], E_pred_test[:, 1])
mse_test_E3, mae_test_E3, r2_test_E3 = evaluate_components(E_true_test[:, 2], E_pred_test[:, 2])


# Output the evaluation for each component

print(f"B3: MSE = {mse_test_B3}, MAE = {mae_test_B3}, R² = {r2_test_B3}")
print(f"E1: MSE = {mse_test_E1}, MAE = {mae_test_E1}, R² = {r2_test_E1}")
print(f"E2: MSE = {mse_test_E2}, MAE = {mae_test_E2}, R² = {r2_test_E2}")
print(f"E3: MSE = {mse_test_E3}, MAE = {mae_test_E3}, R² = {r2_test_E3}")


# Save evaluation metrics to a text file

metrics_train_file = os.path.join("/home/botingl/machine learning copy/results", generate_filename("evaluation_metrics_test", "txt"))
with open(metrics_train_file, "w") as f:
    f.write(f"Whole Model: NMSE: {nmse_test}, rMAE: {rmae_test}, R²: {r2_whole_model_test}\n")
    f.write(f"B3: MSE = {mse_test_B3}, MAE = {mae_test_B3}, R² = {r2_test_B3}\n")
    f.write(f"E1: MSE = {mse_test_E1}, MAE = {mae_test_E1}, R² = {r2_test_E1}\n")
    f.write(f"E2: MSE = {mse_test_E2}, MAE = {mae_test_E2}, R² = {r2_test_E2}\n")
    f.write(f"E3: MSE = {mse_test_E3}, MAE = {mae_test_E3}, R² = {r2_test_E3}\n")


# Plot the true vs. predicted B and E values for the test set
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(B_true_test, B_pred_test, alpha=0.5)
plt.plot([-0.05, 1.1], [-0.05, 1.1], 'g--')
plt.xlim(-0.05, 0.35)
plt.ylim(-0.05, 0.35)
plt.xlabel('True B3')
plt.ylabel('Predicted B3')
plt.title('True vs. Predicted B3 (Test)')

plt.subplot(2, 2, 2)
plt.scatter(E_true_test[:, 0], E_pred_test[:, 0], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E1')
plt.ylabel('Predicted E1')
plt.title('True vs. Predicted E1 (Test)')

plt.subplot(2, 2, 3)
plt.scatter(E_true_test[:, 1], E_pred_test[:, 1], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E2')
plt.ylabel('Predicted E2')
plt.title('True vs. Predicted E2 (Test)')

plt.subplot(2, 2, 4)
plt.scatter(E_true_test[:, 2], E_pred_test[:, 2], alpha=0.5)
plt.plot([-100, 2100], [-100, 2100], 'g--')
plt.xlim(-100, 2100)
plt.ylim(-100, 2100)
plt.xlabel('True E3')
plt.ylabel('Predicted E3')
plt.title('True vs. Predicted E3 (Test)')

plt.suptitle(f'{name_format}_test', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to avoid overlap with suptitle
figure_test_file = os.path.join("/home/botingl/machine learning copy/results", generate_filename("true_vs_predictions_test", "png"))
plt.savefig(figure_test_file, dpi=300, facecolor='white')
plt.show()

print("Testing set evaluation done.", flush=True)





