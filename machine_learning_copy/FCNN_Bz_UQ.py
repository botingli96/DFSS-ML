#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py
import os
import scipy.io
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random

# Set seeds for reproducibility
random.seed(42)  # For Python's built-in random module
np.random.seed(42)  # For NumPy
tf.random.set_seed(42)  # For TensorFlow

# Optionally, set TensorFlow to deterministic mode for even more reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# Function to generate file names based on the input format


# Function to generate file names based on the input format
name_format = "FCNN_Bz_UQtest"
print("Note: use results16.mat which has 10000 datasets; B only has positive Z values")
file_path = os.path.join('MATLAB_DATA', 'results16.mat')  # Replace with your actual file path
def generate_filename(base_name, extension):
    return f"{name_format}_{base_name}.{extension}"


h5_file_numbers = {3, 6, 7, 9}


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


# File paths for training and testing data


# Example usage
B_MAG_data, EDC_MAG_data, X_data, I_data = load_and_process_file(file_path)

# Data is now loaded and processed.
print("Data reading completed.")

# Filter out low-variance columns
means = np.mean(I_data, axis=0)
stds = np.std(I_data, axis=0)
start_index, end_index = 0, I_data.shape[1] - 1

for i in range(I_data.shape[1]):
    if stds[i] >= 0.01:
        start_index = i
        break

for i in range(I_data.shape[1] - 1, -1, -1):
    if stds[i] >= 0.01:
        end_index = i
        break

I_data_filtered = I_data[:, start_index:end_index + 1]
means_filtered = means[start_index:end_index + 1]
stds_filtered = stds[start_index:end_index + 1]
normalized_I = (I_data_filtered - means_filtered) / stds_filtered

def resample_and_smooth_data(normalized_data, new_length=1200):
    resampled_data = []
    
    for row in normalized_data:
        original_length = len(row)
        # Define the original x values (relative position in the original data)
        x_original = np.linspace(0, 1, num=original_length)
        
        # Define the new x values (for the desired length)
        x_new = np.linspace(0, 1, num=new_length)
        
        # Use cubic interpolation to resample and smooth
        f = interp1d(x_original, row, kind='cubic', fill_value="extrapolate")
        resampled_row = f(x_new)
        
        resampled_data.append(resampled_row)
    
    return np.array(resampled_data)

# Resample and smooth the normalized data to ensure each element has a length of 1200
normalized_I_resampled = resample_and_smooth_data(normalized_I, new_length=1200)

# Flatten the I_data to use as input for the neural network
I_data_flat = normalized_I_resampled.reshape(normalized_I_resampled.shape[0], -1)

# Stack only non-zero B and E components to create the output (target) data
target_data = np.column_stack((B_MAG_data[:, 2], EDC_MAG_data))

# Split the data into training and testing sets
input_train, input_test, target_train, target_test = train_test_split(I_data_flat, target_data, test_size=0.2, random_state=42)

# Only the B3 component is used for training now
B_train = target_train[:, :1]
E_train = target_train[:, 1:]
B_test = target_test[:, :1]
E_test = target_test[:, 1:]

# Scaling the B3 and E components
scaler_B3 = MinMaxScaler()
B_train_scaled = scaler_B3.fit_transform(B_train)
B_test_scaled = scaler_B3.transform(B_test)

scaler_E1 = MinMaxScaler()
scaler_E2 = MinMaxScaler()
scaler_E3 = MinMaxScaler()

E_train_scaled = np.column_stack((
    scaler_E1.fit_transform(E_train[:, 0].reshape(-1, 1)),
    scaler_E2.fit_transform(E_train[:, 1].reshape(-1, 1)),
    scaler_E3.fit_transform(E_train[:, 2].reshape(-1, 1))
))
E_test_scaled = np.column_stack((
    scaler_E1.transform(E_test[:, 0].reshape(-1, 1)),
    scaler_E2.transform(E_test[:, 1].reshape(-1, 1)),
    scaler_E3.transform(E_test[:, 2].reshape(-1, 1))
))

target_train_scaled = np.column_stack((B_train_scaled, E_train_scaled))
target_test_scaled = np.column_stack((B_test_scaled, E_test_scaled))


# Data is now loaded and processed.


print("Data organized.", flush=True)


from tensorflow.keras.callbacks import Callback
class LossMonitor(Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch}: loss = {logs['loss']}")

# Define the FCNN model structure as a function for reuse
def create_fcnn_model():
    model = Sequential([
        Dense(512, kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(256, kernel_regularizer=l2(0.0003)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(128, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.1),
        
        Dense(64, kernel_regularizer=l2(0.0001), activation='swish'),
        Dropout(0.1),
        
        Dense(32, activation='swish'),
        Dense(16, activation='swish'),
        Dense(4)  # Output layer
    ])
    
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mse', metrics=['mse'])
    return model


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)


# Define training parameters
n_bootstrap = 50  # Number of bootstrap models
bootstrap_models = []
bootstrap_predictions = []

# Train multiple models on bootstrap samples
for i in range(n_bootstrap):
    # Generate bootstrap sample
    indices = np.random.choice(len(input_train), len(input_train), replace=True)
    X_bootstrap = input_train[indices]
    y_bootstrap = target_train_scaled[indices]
    
    # Create and train a new FCNN model
    model = create_fcnn_model()
    model.fit(
        X_bootstrap, 
        y_bootstrap, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=0)
        ],
        verbose=0  # Suppress training output
    )
    bootstrap_models.append(model)

# Generate predictions and calculate uncertainty
predictions_test = np.array([model.predict(input_test) for model in bootstrap_models])  # Shape: (n_bootstrap, num_samples, num_outputs)
mean_test = predictions_test.mean(axis=0)
std_dev_test = predictions_test.std(axis=0)  # Standard deviation as aleatoric uncertainty

# Confidence intervals
lower_bound_test = mean_test - 1.96 * std_dev_test
upper_bound_test = mean_test + 1.96 * std_dev_test

print("Mean Predictions:", mean_test)
print("Standard Deviations:", std_dev_test)

# Define component names
components = ["B3", "E1", "E2", "E3"]

# Create subplots for each component
plt.figure(figsize=(12, 12))

for i in range(4):  # Iterate through the 4 components
    plt.subplot(2, 2, i + 1)
    
    # Scatter plot for true vs predicted
    plt.scatter(target_test_scaled[:, i], mean_test[:, i], alpha=0.6, label="Predicted Mean")
    
    # Error bars for confidence intervals
    plt.errorbar(target_test_scaled[:, i], mean_test[:, i], 
                 yerr=[mean_test[:, i] - lower_bound_test[:, i], upper_bound_test[:, i] - mean_test[:, i]], 
                 fmt='o', ecolor='orange', alpha=0.5, label="95% Confidence Interval")
    
    # Reference line (y = x)
    plt.plot([min(target_test_scaled[:, i]), max(target_test_scaled[:, i])], 
             [min(target_test_scaled[:, i]), max(target_test_scaled[:, i])], 
             'g--', label="Ideal Prediction (y=x)")
    
    # Axis labels and titles
    plt.xlabel(f"True {components[i]}")
    plt.ylabel(f"Predicted {components[i]}")
    plt.title(f"True vs Predicted {components[i]} with Uncertainty")
    plt.legend()

# Add overall title and adjust layout
plt.suptitle("True vs Predicted with Confidence Intervals (Test Data)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Avoid overlap with suptitle
figure_test_file = os.path.join("/home/botingl/machine learning", generate_filename("Predictions with 95% Confidence Intervals (Test Data)", "png"))
plt.savefig(figure_test_file, dpi=300, facecolor='white')
plt.show()

