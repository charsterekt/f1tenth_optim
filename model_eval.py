import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data
data = np.load("X_train.npz", allow_pickle=True)
X_train = pd.DataFrame({file: data[file] for file in data.files})
data2 = np.load("X_val.npz", allow_pickle=True)
X_val = pd.DataFrame({file: data2[file] for file in data2.files})
data3 = np.load("y_train.npz", allow_pickle=True)
y_train = pd.DataFrame({file: data3[file] for file in data3.files})
data4 = np.load("y_val.npz", allow_pickle=True)
y_val = pd.DataFrame({file: data4[file] for file in data4.files})

# Normalize images
def normalize_images(img):
    return img / 255.0

X_train['image_arrays'] = X_train['image_arrays'].apply(normalize_images)
X_val['image_arrays'] = X_val['image_arrays'].apply(normalize_images)

# Convert DataFrames to numpy arrays
X_train_images = np.stack(X_train['image_arrays'].values)
X_train_lidar = np.stack(X_train['lidar_ranges'].values)
X_val_images = np.stack(X_val['image_arrays'].values)
X_val_lidar = np.stack(X_val['lidar_ranges'].values)

# Combine features
X_train_combined = [X_train_images, X_train_lidar]
X_val_combined = [X_val_images, X_val_lidar]

# Verify data shapes
print("X_train_images shape:", X_train_images.shape)
print("X_train_lidar shape:", X_train_lidar.shape)
print("y_train shape:", y_train.shape)
print("X_val_images shape:", X_val_images.shape)
print("X_val_lidar shape:", X_val_lidar.shape)
print("y_val shape:", y_val.shape)

# Load the model
model_path = '/Users/chari/autodrive/models/orpheus_base_saved_model'
simple_model = load_model(model_path)

# Confirm that the model structure is as expected
simple_model.summary()

# Evaluate the model on the validation set
val_loss, val_mae = simple_model.evaluate(X_val_combined, y_val, verbose=2)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

# Get raw predictions
raw_predictions = simple_model.predict(X_val_combined)
np.clip(raw_predictions, -1, 1)

# Plot the raw predictions
plt.figure(figsize=(10, 5))
plt.plot(y_val[:100], label='Actual Steering Angles')
plt.plot(raw_predictions[:100], label='Predicted Steering Angles (Raw)')
plt.xlabel('Sample Index')
plt.ylabel('Steering Angle')
plt.title('Actual vs Predicted Steering Angles on Validation Set (Raw)')
plt.legend()
plt.show()

print("Raw Predictions Min:", raw_predictions.min())
print("Raw Predictions Max:", raw_predictions.max())