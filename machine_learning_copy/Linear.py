import os
import scipy.io
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

name_format = "Linear_results5_2"
print("Run:", name_format)


# Function to generate file names based on the input format


def generate_filename(base_name, extension):
    return f"{name_format}_{base_name}.{extension}"

# Load and preprocess the data
print("Loading data...", flush=True)

# Define the path to the .mat file
file_path = os.path.join('MATLAB_DATA', 'results5.mat')

# Load the .mat file
mat_contents = scipy.io.loadmat(file_path)

# Access the 'results' variable
results = mat_contents['results']

# Define a fixed length for resampling
fixed_length = 1200  # Adjust based on the average length of your data

# Initialize lists to hold the data
X_data = []
I_data = []
B_MAG_data = []
EDC_MAG_data = []

# Function to resample and smooth data
def resample_and_smooth(X, I, new_length):
    # Define new X as evenly spaced values between 6562.3 and 6563.3
    X_new = np.linspace(6562.3, 6563.3, num=new_length)
    
    # Interpolate the I values over the new X values
    f = interp1d(X, I, kind='cubic', fill_value="extrapolate")
    I_new = f(X_new)
    
    return X_new, I_new

# Iterate through the results
for result in results[0]:
    B_MAG = result['B_MAG'][0]
    EDC_MAG = result['EDC_MAG'][0]
    X = result['X'][0]
    I = result['I'][0]
    
    # Scale the X values by 10^10
    X = X * 10**10

    # Resample X and smooth I
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

# Preprocess the data
means = np.mean(I_data, axis=0)
stds = np.std(I_data, axis=0)

# Find the first and last column with std >= 0.01
start_index = 0
end_index = I_data.shape[1] - 1

for i in range(I_data.shape[1]):
    if stds[i] >= 0.01:
        start_index = i
        break

for i in range(I_data.shape[1] - 1, -1, -1):
    if stds[i] >= 0.01:
        end_index = i
        break

# Filter the data based on the calculated start and end indices
X_resampled = X_data[0]
I_data_filtered = I_data[:, start_index:end_index + 1]
X_resampled_filtered = X_resampled[start_index:end_index + 1]
means_filtered = means[start_index:end_index + 1]
stds_filtered = stds[start_index:end_index + 1]

# Normalize the filtered data
normalized_I = (I_data_filtered - means_filtered) / stds_filtered

# Combine B and E components to create the output data (target)
target_data = np.column_stack((B_MAG_data, EDC_MAG_data))

# Split the data into training and test sets
I_train, I_test, target_train, target_test = train_test_split(normalized_I, target_data, test_size=0.2, random_state=42)

# Separate B and E components
B_train = target_train[:, :3]
E_train = target_train[:, 3:]
B_test = target_test[:, :3]
E_test = target_test[:, 3:]

# # Initialize a single scaler for all B components
# scaler_B = MinMaxScaler()

# # Reshape B_train to a single column (1D vector), scale it, and then reshape it back
# B_train_reshaped = B_train.reshape(-1, 1)
# B_train_scaled = scaler_B.fit_transform(B_train_reshaped).reshape(B_train.shape)

# # Do the same for B_test
# B_test_reshaped = B_test.reshape(-1, 1)
# B_test_scaled = scaler_B.transform(B_test_reshaped).reshape(B_test.shape)

# # Initialize a single scaler for all B components
# scaler_E = MinMaxScaler()

# # Reshape B_train to a single column (1D vector), scale it, and then reshape it back
# E_train_reshaped = E_train.reshape(-1, 1)
# E_train_scaled = scaler_E.fit_transform(E_train_reshaped).reshape(E_train.shape)

# # Do the same for B_test
# E_test_reshaped = E_test.reshape(-1, 1)
# E_test_scaled = scaler_E.transform(E_test_reshaped).reshape(E_test.shape)

scaler_B1, scaler_B2, scaler_B3 = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
scaler_E1, scaler_E2, scaler_E3 = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()


B_train_scaled = np.column_stack((
    scaler_B1.fit_transform(B_train[:, 0].reshape(-1, 1)),
    scaler_B2.fit_transform(B_train[:, 1].reshape(-1, 1)),
    scaler_B3.fit_transform(B_train[:, 2].reshape(-1, 1))
))


B_test_scaled = np.column_stack((
    scaler_B1.transform(B_test[:, 0].reshape(-1, 1)),
    scaler_B2.transform(B_test[:, 1].reshape(-1, 1)),
    scaler_B3.transform(B_test[:, 2].reshape(-1, 1))
))


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


# Combine the scaled B and E components for the train and test sets
target_train_scaled = np.column_stack((B_train_scaled, E_train_scaled))
target_test_scaled = np.column_stack((B_test_scaled, E_test_scaled))

# Train the linear regression model
print("Training linear regression model...", flush=True)
linear_model = LinearRegression()
linear_model.fit(I_train, target_train_scaled)


train_subset_indices = np.random.choice(len(I_train), size=1000, replace=False)
train_input_data_subset = I_train[train_subset_indices]
train_target_data_subset = target_train_scaled[train_subset_indices]
# Make predictions on the train set
predictions_train = linear_model.predict(train_input_data_subset)

mae_train = mean_absolute_error(train_target_data_subset, predictions_train)
rmse_train = np.sqrt(mean_squared_error(train_target_data_subset, predictions_train))
r2_train = r2_score(train_target_data_subset, predictions_train)



def normalized_root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (y_true.max() - y_true.min())



nrmse_train = normalized_root_mean_squared_error(train_target_data_subset, predictions_train)


print(f'NRMSE (Train): {nrmse_train:.2f}')
print(f'Mean Absolute Error (MAE, Train): {mae_train}')
print(f'Root Mean Squared Error (RMSE, Train): {rmse_train}')
print(f'R^2 Score (Train): {r2_train}')

metrics_test_file = os.path.join("/home/botingl/machine learning", generate_filename("evaluation_metrics_train", "txt"))
with open(metrics_test_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae_train}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse_train}\n")
    f.write(f"R^2 Score: {r2_train}\n")
    f.write(f"Normalized RMSE (NRMSE): {nrmse_train}\n")

# Plot the true vs. predicted B and E values for the training set

plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.scatter(train_target_data_subset[:, i], predictions_train[:, i], alpha=0.5)
    plt.xlabel(f'True B[{i+1}]')
    plt.ylabel(f'Predicted B[{i+1}]')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'True vs. Predicted B[{i+1}] (Train)')



for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.scatter(train_target_data_subset[:, i+3], predictions_train[:, i+3], alpha=0.5)
    plt.xlabel(f'True E[{i+1}]')
    plt.ylabel(f'Predicted E[{i+1}]')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'True vs. Predicted E[{i+1}] (Train)')


plt.tight_layout()
figure_train_file = os.path.join("/home/botingl/machine learning", generate_filename("true_vs_predictions_train", "png"))
plt.savefig(figure_train_file, dpi=300, facecolor='white')
plt.show()
plt.close()


# Test set evaluation

# In[79]:

# Make predictions on the test set
predictions_test = linear_model.predict(I_test)
mae_test = mean_absolute_error(target_test_scaled, predictions_test)
rmse_test = np.sqrt(mean_squared_error(target_test_scaled, predictions_test))
r2_test = r2_score(target_test_scaled, predictions_test)
nrmse_test = normalized_root_mean_squared_error(target_test_scaled, predictions_test)


# In[80]:


print(f'NRMSE (Test): {nrmse_test:.2f}')
print(f'Mean Absolute Error (MAE, Test): {mae_test}')
print(f'Root Mean Squared Error (RMSE, Test): {rmse_test}')
print(f'R^2 Score (Test): {r2_test}')


# In[81]:


metrics_test_file = os.path.join("/home/botingl/machine learning", generate_filename("evaluation_metrics_test", "txt"))
with open(metrics_test_file, "w") as f:
    f.write(f"Mean Absolute Error (MAE): {mae_test}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse_test}\n")
    f.write(f"R^2 Score: {r2_test}\n")
    f.write(f"Normalized RMSE (NRMSE): {nrmse_test}\n")


# In[82]:


plt.figure(figsize=(15, 10))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.scatter(target_test_scaled[:, i], predictions_test[:, i], alpha=0.5)
    plt.xlabel(f'True B[{i+1}]')
    plt.ylabel(f'Predicted B[{i+1}]')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'True vs. Predicted B[{i+1}] (Test)')


# In[83]:


for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.scatter(target_test_scaled[:, i+3], predictions_test[:, i+3], alpha=0.5)
    plt.xlabel(f'True E[{i+1}]')
    plt.ylabel(f'Predicted E[{i+1}]')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title(f'True vs. Predicted E[{i+1}] (Test)')


# In[84]:


plt.tight_layout()
figure_test_file = os.path.join("/home/botingl/machine learning", generate_filename("true_vs_predictions_test", "png"))
plt.savefig(figure_test_file, dpi=300, facecolor='white')
plt.show()
plt.close()


