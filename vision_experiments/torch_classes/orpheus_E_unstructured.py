import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the new pruned model architecture directly
class PrunedTimeDistributedCNN(nn.Module):
    def __init__(self):
        super(PrunedTimeDistributedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        self.flatten = nn.Flatten()
        
        self.convs = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self._get_conv_output()
        self.fc = nn.Linear(self._to_linear, 100)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 360, 640)
            x = self.convs(x)
            self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.convs(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = x.view(batch_size, seq_length, -1)
        return x

class PrunedCNNLSTMModel(nn.Module):
    def __init__(self, seq_length):
        super(PrunedCNNLSTMModel, self).__init__()
        self.seq_length = seq_length
        self.time_distributed_cnn = PrunedTimeDistributedCNN()
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