import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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


# Ensure y_train and y_val are numpy arrays
y_train_steering = y_train['steering'].values
y_train_throttle = y_train['throttle'].values
y_val_steering = y_val['steering'].values
y_val_throttle = y_val['throttle'].values

# Stack steering and throttle together
y_train = np.column_stack((y_train_steering, y_train_throttle))
y_val = np.column_stack((y_val_steering, y_val_throttle))

# Convert the DataFrames to numpy arrays
X_train_images = np.stack(X_train['image_arrays'].values)
X_train_lidars = np.stack(X_train['lidar_ranges'].values)
X_val_images = np.stack(X_val['image_arrays'].values)
X_val_lidars = np.stack(X_val['lidar_ranges'].values)

# Verify the shapes of the data
print(f'Original X_train_images shape: {X_train_images.shape}')
print(f'Original X_train_lidars shape: {X_train_lidars.shape}')
print(f'Original X_val_images shape: {X_val_images.shape}')
print(f'Original X_val_lidars shape: {X_val_lidars.shape}')
print(f'Original y_train shape: {y_train.shape}')
print(f'Original y_val shape: {y_val.shape}')



class SteeringDataset(Dataset):
    def __init__(self, images, lidars, targets, seq_length):
        self.images = images
        self.lidars = lidars
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.images) - self.seq_length + 1

    def __getitem__(self, idx):
        image_seq = self.images[idx:idx + self.seq_length].astype(np.float32) / 255.0
        lidar_seq = self.lidars[idx:idx + self.seq_length].astype(np.float32) / 10.0
        target = self.targets[idx + self.seq_length - 1].astype(np.float32)  # Predict target for the last time step

        image_seq_tensor = torch.tensor(image_seq).permute(0, 3, 1, 2)  # Adjust dimensions to [seq_length, channels, height, width]
        lidar_seq_tensor = torch.tensor(lidar_seq)
        target_tensor = torch.tensor(target)

        return image_seq_tensor, lidar_seq_tensor, target_tensor

# Define sequence length
seq_length = 10

# Create datasets
train_dataset = SteeringDataset(X_train_images, X_train_lidars, y_train, seq_length)
val_dataset = SteeringDataset(X_val_images, X_val_lidars, y_val, seq_length)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Verify the shapes of the data loaders
for image_seq, lidar_seq, target in train_loader:
    print(f'Image sequence shape: {image_seq.shape}')
    print(f'Lidar sequence shape: {lidar_seq.shape}')
    print(f'Target shape: {target.shape}')
    break


class TimeDistributedCNN(nn.Module):
    def __init__(self):
        super(TimeDistributedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        
        # Calculate the output size after convolution layers
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self._get_conv_output()

        # Adjust the dimensions to match the expected input size of LSTM
        self.fc = nn.Linear(self._to_linear, 100)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 360, 640)
            x = self.convs(x)
            self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = x.view(batch_size, seq_length, -1)
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, seq_length):
        super(CNNLSTMModel, self).__init__()
        self.seq_length = seq_length
        self.time_distributed_cnn = TimeDistributedCNN()
        self.lstm_image = nn.LSTM(input_size=100, hidden_size=64, batch_first=True)
        self.lstm_lidar = nn.LSTM(input_size=1081, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64 * 2, 2)  # Output 2 values: steering and throttle

    def forward(self, x_image, x_lidar):
        x_image = self.time_distributed_cnn(x_image)
        x_image, _ = self.lstm_image(x_image)
        x_lidar, _ = self.lstm_lidar(x_lidar)
        x = torch.cat((x_image[:, -1, :], x_lidar[:, -1, :]), dim=-1)
        x = self.fc(x)
        return x

model = CNNLSTMModel(seq_length=10)
model.load_state_dict(torch.load('/Users/chari/autodrive/models/orpheus_torch.pth', map_location=torch.device('cpu')))

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            
            self.model.train()
            for images_seq, lidars_seq, targets in self.train_loader:
                images_seq, lidars_seq, targets = images_seq.to(self.device), lidars_seq.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images_seq, lidars_seq)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            self.history['train_loss'].append(train_loss)
            
            self.model.eval()
            with torch.no_grad():
                for images_seq, lidars_seq, targets in self.val_loader:
                    images_seq, lidars_seq, targets = images_seq.to(self.device), lidars_seq.to(self.device), targets.to(self.device)
                    outputs = self.model(images_seq, lidars_seq)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
    
    def plot_predictions(self, y_val):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images_seq, lidars_seq, targets in self.val_loader:
                images_seq, lidars_seq = images_seq.to(self.device), lidars_seq.to(self.device)
                outputs = self.model(images_seq, lidars_seq)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        plt.figure(figsize=(10, 5))
        plt.plot(all_targets[:, 0], label='Actual Steering')
        plt.plot(all_preds[:, 0], label='Predicted Steering')
        plt.xlabel('Sample')
        plt.ylabel('Steering Angle')
        plt.legend()
        plt.title('Actual vs Predicted Steering Angle')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(all_targets[:, 1], label='Actual Throttle')
        plt.plot(all_preds[:, 1], label='Predicted Throttle')
        plt.xlabel('Sample')
        plt.ylabel('Throttle')
        plt.legend()
        plt.title('Actual vs Predicted Throttle')
        plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = ModelTrainer(model, train_loader, val_loader, num_epochs=10)
# history = trainer.train()
# trainer.plot_training_history()
# trainer.plot_predictions()
trainer.plot_predictions(y_val)

# Get a batch from the validation loader
for images_seq, lidars_seq, actual_targets in val_loader:
    break  # Take the first batch only

# Ensure we have the correct device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Take the first sample from the batch
images_seq = images_seq[0].unsqueeze(0).to(device)  # Add batch dimension
lidars_seq = lidars_seq[0].unsqueeze(0).to(device)  # Add batch dimension
actual_target = actual_targets[0].cpu().numpy()

print(f"Image input length: {images_seq.size()}")
print(f"Lidar input length: {lidars_seq.size()}")
print(f"Actual output: {actual_target}")

# Make a prediction using the model
with torch.no_grad():
    predicted_target = model(images_seq, lidars_seq).cpu().numpy()

print(f"Predicted Steering Angle: {predicted_target[0, 0]:.4f}")
print(f"Actual Steering Angle: {actual_target[0]:.4f}")
print(f"Predicted Throttle: {predicted_target[0, 1]:.4f}")
print(f"Actual Throttle: {actual_target[1]:.4f}")