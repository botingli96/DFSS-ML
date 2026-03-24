#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read and preprocess the data.

import scipy.io
import os
import numpy as np
from scipy.interpolate import interp1d
print("Loading data...", flush=True)
# Define the path to the .mat file
file_path = os.path.join('MATLAB_DATA', 'results1.mat')

# Load the .mat file
mat_contents = scipy.io.loadmat(file_path)

# Access the 'results' variable
results = mat_contents['results']

for result in results[0]:
    # Scale the I values by 10^10
    result['X'][0] = result['X'][0] * 10**10

# Define a fixed length for resampling
fixed_length = 1200  # Adjust based on the average length of your data

# Initialize lists to hold the data
X_data = []
I_data = []
B_MAG_data = []
EDC_MAG_data = []

# Function to resample and smooth data
def resample_and_smooth(X, I, new_length):
    # Define new X as evenly spaced values between 6562.5 and 6563.1
    X_new = np.linspace(6562.5, 6563.1, num=new_length)
    
    # Interpolate the I values over the new X values
    f = interp1d(X, I, kind='cubic', fill_value="extrapolate")  # 'cubic' provides smoothing
    I_new = f(X_new)
    
    return X_new, I_new

# Iterate through the results
for result in results[0]:
    B_MAG = result['B_MAG'][0]
    EDC_MAG = result['EDC_MAG'][0]
    X = result['X'][0] 
    I = result['I'][0]
    
    # print(f'B_MAG: {B_MAG}, EDC_MAG: {EDC_MAG}')

    # Resample X and smooth I correspondingly
    X_resampled, I_resampled = resample_and_smooth(X, I, fixed_length)
    
    # Store the data in lists
    X_data.append(X_resampled)
    I_data.append(I_resampled)
    B_MAG_data.append(B_MAG)
    EDC_MAG_data.append(EDC_MAG)

# Convert lists to numpy arrays
X_data = np.array(X_data)
I_data = np.array(I_data)
B_MAG_data = np.array(B_MAG_data)
EDC_MAG_data = np.array(EDC_MAG_data)

print("Data loaded.", flush=True)


# In[2]:


import torch
import sbi
from sbi import utils as sbi_utils
from sbi.inference import SNPE


# In[3]:


# Concatenate B and E to form the input data
input_data = np.hstack((B_MAG_data, EDC_MAG_data))

# Stack X and I to form the target data
# target_data = np.hstack((X_data, I_data))
target_data = I_data

# Convert lists to numpy arrays (if not already done)
input_data = np.array(input_data)
target_data = np.array(target_data)

from sklearn.preprocessing import MinMaxScaler

# Example: Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
input_data= scaler.fit_transform(input_data)
target_data= scaler.fit_transform(target_data)

# Convert numpy arrays to PyTorch tensors
input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
target_data_tensor = torch.tensor(target_data, dtype=torch.float32)


# In[4]:


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


# In[5]:


# Define prior bounds for your data. Adjust according to your specific problem.
prior_min = torch.tensor([input_data_tensor.min().item()] * input_data_tensor.shape[1])
prior_max = torch.tensor([input_data_tensor.max().item()] * input_data_tensor.shape[1])

# prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)

# Widening the prior range slightly
margin = 0.1 * (prior_max - prior_min)
prior = sbi_utils.BoxUniform(low=prior_min - margin, high=prior_max + margin)


# In[ ]:


# Train the model
print("Start training.", flush=True)

from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn

# Initialize the SNPE with a custom neural network architecture
neural_net = posterior_nn(model='maf',  # Consider 'maf' (Masked Autoregressive Flow) or 'nsf' (Neural Spline Flow)
                          hidden_features=64,  # Increase the number of hidden features
                          num_transforms=10)  # Increase the number of transformations

# Initialize SNPE with the prior and the custom neural network
inference = SNPE(prior=prior, density_estimator=neural_net)

# Set the training steps during the training process (not during `train()`)
inference.append_simulations(input_data_train_tensor, target_data_train_tensor)
density_estimator = inference.train(max_num_epochs=150)  # Increase epochs for potentially better results

# Build posterior
posterior = inference.build_posterior(density_estimator)
# Build the posterior using NUTS for sampling
# posterior = inference.build_posterior(density_estimator, sample_with='mcmc')

print("Training done.", flush=True)


# In[ ]:


# Save the model

import torch

# Define the file paths for saving
posterior_path = "/p/ltxdata/DFSS/machine learning/posterior_1.pt"
density_estimator_path = "/p/ltxdata/DFSS/machine learning/density_estimator_1.pt"

# Save the posterior (which includes the trained model)
torch.save(posterior, posterior_path)

# Save the density estimator state dictionary
torch.save(density_estimator.state_dict(), density_estimator_path)

print("Model saved successfully.")


# In[6]:


# Load saved model

import torch
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn

# Define the file paths for loading
posterior_path = "/p/ltxdata/DFSS/machine learning/posterior_1.pt"
density_estimator_path = "/p/ltxdata/DFSS/machine learning/density_estimator_1.pt"

# Load the posterior
posterior = torch.load(posterior_path)

print("Model loaded successfully.")


# In[7]:


print("Start to evaluate.", flush=True)
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

print("Evaluation done.", flush=True)


# In[13]:


# Evaluate the model using all test sets
output_dir = "/p/ltxdata/DFSS/machine learning"
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
metrics_file = os.path.join(output_dir, "evaluation_metrics_1.txt")
with open(metrics_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"R^2 Score: {r2}\n")
    f.write(f"Normalized RMSE (NRMSE): {nrmse}\n")

# Optionally, you can visualize the results
import matplotlib.pyplot as plt

# Plot the true vs. predicted B and E values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(true_values[:, 0], predictions[:, 0], alpha=0.5)
plt.xlabel('True B')
plt.ylabel('Predicted B')
plt.title('True vs. Predicted B')

plt.subplot(1, 2, 2)
plt.scatter(true_values[:, 1], predictions[:, 1], alpha=0.5)
plt.xlabel('True E')
plt.ylabel('Predicted E')
plt.title('True vs. Predicted E')

plt.tight_layout()

figure_file = os.path.join(output_dir, "true_vs_predictions_1.png")
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

