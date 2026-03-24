#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ask the user to input the name format
name_format = "large_run11"
print("Run:", name_format)

# Function to generate file names based on the input format
def generate_filename(base_name, extension):
    return f"{name_format}_{base_name}.{extension}"


# In[3]:


import scipy.io
import os
import numpy as np
from scipy.interpolate import interp1d

print("Loading data...", flush=True)

# Define the paths to the .mat files
b_file_path = os.path.join('MATLAB_DATA', 'B.mat')
e_file_path = os.path.join('MATLAB_DATA', 'EDC.mat')
x_file_path = os.path.join('MATLAB_DATA', 'X.mat')
i_file_path = os.path.join('MATLAB_DATA', 'I.mat')

# Load the .mat files
b_contents = scipy.io.loadmat(b_file_path)
e_contents = scipy.io.loadmat(e_file_path)
x_contents = scipy.io.loadmat(x_file_path)
i_contents = scipy.io.loadmat(i_file_path)

# Access the variables from the loaded .mat files
B_MAG = b_contents['results1']
EDC_MAG = e_contents['results2']
X_data_raw = x_contents['results3']
I_data_raw = i_contents['results4']

# Scale the X values by 10^10 (assuming you still need this step)
for i in range(len(X_data_raw)):
    X_data_raw[i][0][0][0] = X_data_raw[i][0][0][0] * 10**10

# Define a fixed length for resampling
fixed_length = 1200  # Adjust based on the average length of your data

# Initialize lists to hold the resampled data
X_data = []
I_data = []
B_MAG_data = []
EDC_MAG_data = []
# Function to resample and smooth data
def resample_and_smooth(X, I, new_length):
    # Define new X as evenly spaced values between 6562.5 and 6563.1
    X_new = np.linspace(6562.3, 6563.3, num=new_length)
    
    # Interpolate the I values over the new X values
    f = interp1d(X, I, kind='cubic', fill_value="extrapolate")  # 'cubic' provides smoothing
    I_new = f(X_new)
    
    return X_new, I_new

# Iterate through the data and resample
for i in range(len(X_data_raw)):
    X = X_data_raw[i][0][0][0]
    I = I_data_raw[i][0][0][0]
    B = B_MAG[i][0][0][0]
    EDC = EDC_MAG[i][0][0][0]
    # Resample X and smooth I correspondingly
    X_resampled, I_resampled = resample_and_smooth(X, I, fixed_length)
    
    # Store the resampled data in lists
    X_data.append(X_resampled)
    I_data.append(I_resampled)
    B_MAG_data.append(B)
    EDC_MAG_data.append(EDC)

# Convert lists to numpy arrays
X_data = np.array(X_data)
I_data = np.array(I_data)
B_MAG_data = np.array(B_MAG_data)
EDC_MAG_data = np.array(EDC_MAG_data)

print("Data loaded and processed.", flush=True)


# In[4]:


import numpy as np

# Assuming I_data is your input numpy array
# Step 1: Calculate the mean of each column (axis=0 means column-wise operation)
means = np.mean(I_data, axis=0)

# Step 2: Calculate the standard deviation of each column
stds = np.std(I_data, axis=0)

# Step 2: Initialize indices for the columns to keep
start_index = 0
end_index = I_data.shape[1] - 1

# Step 3: Find the first column with std larger than 0.01 from the start
for i in range(I_data.shape[1]):
    if stds[i] >= 0.01:
        start_index = i
        break

# Step 4: Find the first column with std larger than 0.01 from the end
for i in range(I_data.shape[1] - 1, -1, -1):
    if stds[i] >= 0.01:
        end_index = i
        break

# Step 5: Slice the I_data, B_MAG_data, and EDC_MAG_data based on the start and end indices
I_data_filtered = I_data[:, start_index:end_index + 1]
X_resampled_filtered = X_resampled[start_index:end_index + 1]
means_filtered = means[start_index:end_index + 1]
stds_filtered = stds[start_index:end_index + 1]
print(f"Data filtered from column {start_index} to {end_index}.")


# Step 3: Normalize the array (I_data - mean) / std
normalized_I = (I_data_filtered - means_filtered) / stds_filtered

# Now normalized_data contains the result of (I - mean) / std for each column.


# In[5]:


import torch
import sbi
from sbi import utils as sbi_utils
from sbi.inference import SNPE


# In[5]:


# Concatenate B and E to form the input data
input_data = np.hstack((B_MAG_data, EDC_MAG_data))

# Stack X and I to form the target data
# target_data = np.hstack((X_data, normalized_I))
target_data = normalized_I

# Convert lists to numpy arrays (if not already done)
input_data = np.array(input_data)
target_data = np.array(target_data)

from sklearn.preprocessing import MinMaxScaler

# Example: Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
input_data= scaler.fit_transform(input_data)
# target_data= scaler.fit_transform(target_data)

input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
target_data_tensor = torch.tensor(target_data, dtype=torch.float32)


# In[6]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
input_data_train, input_data_test, target_data_train, target_data_test = train_test_split(
    input_data, target_data, test_size=0.2, random_state=42
)

# Convert these splits to PyTorch tensors
input_data_train_tensor = torch.tensor(input_data_train, dtype=torch.float32)  # Shape: (train_samples, 2400)
input_data_test_tensor = torch.tensor(input_data_test, dtype=torch.float32)  # Shape: (test_samples, 2400)
target_data_train_tensor = torch.tensor(target_data_train, dtype=torch.float32)  # Shape: (train_samples, 6)
target_data_test_tensor = torch.tensor(target_data_test, dtype=torch.float32)  # Shape: (test_samples, 6)

print("Data organized.", flush=True)


# In[7]:


# Define prior bounds for your data. Adjust according to your specific problem.
prior_min = torch.tensor([input_data_tensor.min().item()] * input_data_tensor.shape[1])
prior_max = torch.tensor([input_data_tensor.max().item()] * input_data_tensor.shape[1])

# prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)

# Widening the prior range slightly
margin = 0.1 * (prior_max - prior_min)
prior = sbi_utils.BoxUniform(low=prior_min - margin, high=prior_max + margin)


# In[8]:


# Train the model
print("Start training.", flush=True)

from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# Initialize the SNPE with a custom neural network architecture
neural_net = posterior_nn(model='nsf',  # Consider 'maf' (Masked Autoregressive Flow) or 'nsf' (Neural Spline Flow)
                          hidden_features=64,  # Increase the number of hidden features
                          num_transforms=10)  # Increase the number of transformations

# Initialize SNPE with the prior and the custom neural network
inference = SNPE(prior=prior, density_estimator=neural_net)

# Set the training steps during the training process (not during `train()`)
inference.append_simulations(input_data_train_tensor, target_data_train_tensor)
density_estimator = inference.train(max_num_epochs=200, learning_rate=0.001, training_batch_size=64)  # Increase epochs for potentially better results

# Build posterior
posterior = inference.build_posterior(density_estimator)
# Build the posterior using NUTS for sampling
# posterior = inference.build_posterior(density_estimator, sample_with='mcmc')

print("Training done.", flush=True)


# In[ ]:


# from sklearn.model_selection import KFold
# import torch.utils.data as data_utils

# # Example: Setting up a range of hyperparameters to test
# hidden_features_options = [20, 64, 128]
# learning_rate_options = [0.001, 0.0005, 0.0001]

# # Placeholder to store the best parameters and their performance
# best_avg_loss = float('inf')
# best_params = None

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# dataset = data_utils.TensorDataset(input_data_tensor, target_data_tensor)

# # Iterate over each combination of hyperparameters
# for hidden_features in hidden_features_options:
#     for lr in learning_rate_options:
#         cv_results = []

#         print(f"Testing configuration: hidden_features={hidden_features}, learning_rate={lr}")

#         for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
#             train_subsampler = data_utils.SubsetRandomSampler(train_idx)
#             val_subsampler = data_utils.SubsetRandomSampler(val_idx)

#             train_loader = data_utils.DataLoader(dataset, batch_size=64, sampler=train_subsampler)
#             val_loader = data_utils.DataLoader(dataset, batch_size=64, sampler=val_subsampler)

#             # Initialize the model for each fold
#             neural_net = posterior_nn(model='nsf', hidden_features=hidden_features, num_transforms=5)
#             inference = SNPE(prior=prior, density_estimator=neural_net)

#             inference.append_simulations(input_data_train_tensor, target_data_train_tensor)
#             density_estimator = inference.train(max_num_epochs=200, learning_rate=lr)

#             # Evaluate on validation set
#             val_loss = 0.0
#             for val_inputs, val_targets in val_loader:
#                 val_outputs = neural_net(val_inputs)
#                 val_loss += criterion(val_outputs, val_targets).item()  # Use appropriate loss function

#             val_loss /= len(val_loader)
#             cv_results.append(val_loss)

#             print(f"Validation Loss for fold {fold+1}: {val_loss:.4f}")

#         # Average the validation loss over all folds
#         avg_loss = sum(cv_results) / len(cv_results)
#         print(f"Average Validation Loss for configuration: {avg_loss:.4f}")

#         # If this configuration is the best so far, save it
#         if avg_loss < best_avg_loss:
#             best_avg_loss = avg_loss
#             best_params = {'hidden_features': hidden_features, 'learning_rate': lr}

# # After evaluating all configurations, choose the best one
# print(f"Best configuration: {best_params} with average validation loss: {best_avg_loss:.4f}")

# # Finally, train the model on the full dataset using the best hyperparameters
# neural_net = posterior_nn(model='nsf', hidden_features=best_params['hidden_features'], num_transforms=5)
# inference = SNPE(prior=prior, density_estimator=neural_net)

# inference.append_simulations(input_data_train_tensor, target_data_train_tensor)
# density_estimator = inference.train(max_num_epochs=200, learning_rate=best_params['learning_rate'])

# # This is your final posterior
# posterior = inference.build_posterior(density_estimator)


# In[ ]:


# Save the model
import torch

# Define the file paths for saving
posterior_path = os.path.join("/home/botingl/machine learning", generate_filename("posterior", "pt"))
density_estimator_path = os.path.join("/home/botingl/machine learning", generate_filename("density_estimator", "pt"))

# Save the posterior (which includes the trained model)
torch.save(posterior, posterior_path)

# Save the density estimator state dictionary
torch.save(density_estimator.state_dict(), density_estimator_path)

print("Model saved successfully.")


# In[ ]:


# Load saved model
import torch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn

# Define the file paths for loading
posterior_path = os.path.join("/home/botingl/machine learning", generate_filename("posterior", "pt"))

# Load the posterior
posterior = torch.load(posterior_path)

print("Model loaded successfully.")


# In[ ]:


print("Start to evaluate the training set.", flush=True)
# Generate predictions for all test sets
import sys
import numpy as np
from tqdm import tqdm

# Limit the test set to only 10 groups for faster evaluation
train_subset_indices = np.random.choice(len(input_data_train_tensor), size=1000, replace=False)
train_target_data_subset = target_data_train_tensor[train_subset_indices]
train_input_data_subset = input_data_train_tensor[train_subset_indices]

# Generate predictions for the test set
predictions = []
for i in tqdm(range(len(train_target_data_subset)), desc="Processing samples", leave=True, file=sys.stdout):
    test_input = train_target_data_subset[i]  # This is the X and I for this test sample
    predicted_posterior = posterior.sample((1000,), x=test_input, show_progress_bars=False)  # Disable internal progress bars
    
    # Extract mean prediction for B and E
    predicted_mean = predicted_posterior.mean(dim=0)
    predictions.append(predicted_mean)
    
    sys.stdout.flush()  # Manually flush output

# Convert predictions to numpy array
predictions = torch.stack(predictions).detach().numpy()

print("Evaluation done for the training set.", flush=True)


# In[ ]:


output_dir = "/home/botingl/machine learning"
# Convert the true B and E values to numpy array
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
true_values = train_input_data_subset.numpy()

def normalized_root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (y_true.max() - y_true.min())

# Calculate evaluation metrics
mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
r2 = r2_score(true_values, predictions)
nrmse = normalized_root_mean_squared_error(true_values, predictions)

print(f'NRMSE: {nrmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')

# Save evaluation metrics to a text file
metrics_file = os.path.join(output_dir, generate_filename("evaluation_metrics_train", "txt"))
with open(metrics_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Normalized RMSE (NRMSE): {nrmse}\n")

# Optionally, you can visualize the results
import matplotlib.pyplot as plt

# Plot the true vs. predicted B and E values
plt.figure(figsize=(15, 10))

# Plotting for B components
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)
    plt.xlabel(f'True B[{i+1}]')
    plt.ylabel(f'Predicted B[{i+1}]')
    plt.title(f'True vs. Predicted B[{i+1}]')

# Plotting for E components
for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.scatter(true_values[:, i+3], predictions[:, i+3], alpha=0.5)
    plt.xlabel(f'True E[{i+1}]')
    plt.ylabel(f'Predicted E[{i+1}]')
    plt.title(f'True vs. Predicted E[{i+1}]')

plt.tight_layout()

figure_file = os.path.join(output_dir, generate_filename("true_vs_predictions_train", "png"))
plt.savefig(figure_file, dpi=300, facecolor='white')
plt.show()
plt.close()


# In[ ]:


print("Start to evaluate the test set.", flush=True)
# Generate predictions for all test sets
import sys
import numpy as np
from tqdm import tqdm

# Generate predictions for the test set
predictions = []
for i in tqdm(range(len(target_data_test)), desc="Processing samples", leave=True, file=sys.stdout):
    test_input = target_data_test[i]  # This is the X and I for this test sample
    predicted_posterior = posterior.sample((1000,), x=test_input, show_progress_bars=False)  # Disable internal progress bars
    
    # Extract mean prediction for B and E
    predicted_mean = predicted_posterior.mean(dim=0)
    predictions.append(predicted_mean)
    
    sys.stdout.flush()  # Manually flush output

# Convert predictions to numpy array
predictions = torch.stack(predictions).detach().numpy()

print("Evaluation done for the test set.", flush=True)


# In[ ]:


# Evaluate the model using all test sets
output_dir = "/home/botingl/machine learning"
# Convert the true B and E values to numpy array
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
true_values = input_data_test

def normalized_root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (y_true.max() - y_true.min())

# Calculate evaluation metrics
mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
r2 = r2_score(true_values, predictions)
nrmse = normalized_root_mean_squared_error(input_data_test, predictions)

print(f'NRMSE: {nrmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2 Score: {r2}')

# Save evaluation metrics to a text file
metrics_file = os.path.join(output_dir, generate_filename("evaluation_metrics_test", "txt"))
with open(metrics_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Normalized RMSE (NRMSE): {nrmse}\n")

# Optionally, you can visualize the results
import matplotlib.pyplot as plt

# Plot the true vs. predicted B and E values
plt.figure(figsize=(15, 10))

# Plotting for B components
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.scatter(true_values[:, i], predictions[:, i], alpha=0.5)
    plt.xlabel(f'True B[{i+1}]')
    plt.ylabel(f'Predicted B[{i+1}]')
    plt.title(f'True vs. Predicted B[{i+1}]')

# Plotting for E components
for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.scatter(true_values[:, i+3], predictions[:, i+3], alpha=0.5)
    plt.xlabel(f'True E[{i+1}]')
    plt.ylabel(f'Predicted E[{i+1}]')
    plt.title(f'True vs. Predicted E[{i+1}]')

plt.tight_layout()

figure_file = os.path.join(output_dir, generate_filename("true_vs_predictions_test", "png"))
plt.savefig(figure_file, dpi=300, facecolor='white')
plt.show()
plt.close()


# In[ ]:


# # Limit the test set to only 10 groups for faster evaluation
# test_subset_indices = np.random.choice(len(target_data_test_tensor), size=10, replace=False)
# test_target_data_subset = target_data_test_tensor[test_subset_indices]
# test_input_data_subset = input_data_test_tensor[test_subset_indices]

# # Example of making predictions on the limited test set
# predictions = []
# for i in range(len(test_target_data_subset)):
#     test_input = test_target_data_subset[i]
#     predicted_posterior = posterior.sample((1000,), x=test_input)
    
#     # Extract mean prediction
#     predicted_mean = predicted_posterior.mean(dim=0)
#     predictions.append(predicted_mean)

# # Convert predictions to a numpy array for further evaluation
# predictions = torch.stack(predictions).detach().numpy()


# In[ ]:


# # Plot and compare the 10 groups
# import matplotlib.pyplot as plt
# import numpy as np

# # Assume 'predictions' is the numpy array with predicted values
# # and 'test_data_subset' is the corresponding actual test data.

# # Number of test sets
# num_test_sets = predictions.shape[0]

# # Labels for the B and E values in each test set
# B_labels = ['B1', 'B2', 'B3']  # Adjust based on your specific data
# E_labels = ['E1', 'E2', 'E3']  # Adjust based on your specific data

# # Plotting each test set
# for i in range(num_test_sets):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Two subplots side by side
    
#     # Indices for the bars
#     B_indices = np.arange(3)  # Indices for B1, B2, B3
#     E_indices = np.arange(3)  # Indices for E1, E2, E3
    
#     # Width of the bars
#     width = 0.2
    
#     # Bar plot for B values
#     ax1.bar(B_indices, test_input_data_subset[i, :3], width, label='Actual B')
#     ax1.bar(B_indices + width, predictions[i, :3], width, label='Predicted B')
#     ax1.set_xlabel('B Value Index')
#     ax1.set_ylabel('B Value Magnitude')
#     ax1.set_title(f'Comparison of Actual vs. Predicted B Values for Test Set {i+1}')
#     ax1.set_xticks(B_indices + width / 2)
#     ax1.set_xticklabels(B_labels)
#     ax1.legend()
    
#     # Bar plot for E values
#     ax2.bar(E_indices, test_input_data_subset[i, 3:], width, label='Actual E')
#     ax2.bar(E_indices + width, predictions[i, 3:], width, label='Predicted E')
#     ax2.set_xlabel('E Value Index')
#     ax2.set_ylabel('E Value Magnitude')
#     ax2.set_title(f'Comparison of Actual vs. Predicted E Values for Test Set {i+1}')
#     ax2.set_xticks(E_indices + width / 2)
#     ax2.set_xticklabels(E_labels)
#     ax2.legend()
    
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
    
#     # Show plot
#     plt.show()


# In[ ]:


# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# def normalized_root_mean_squared_error(y_true, y_pred):
#     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
#     return rmse / (y_true.max() - y_true.min())

# # Calculate NRMSE for each test set
# nrmse = normalized_root_mean_squared_error(test_input_data_subset.numpy(), predictions)
# print(f'NRMSE: {nrmse:.2f}')

# # Calculate R-squared (R²)
# r2 = r2_score(test_input_data_subset, predictions)
# print(f"R-squared (R²): {r2}")

