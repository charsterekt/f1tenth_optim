import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributedCNN(nn.Module):
    def __init__(self):
        super(TimeDistributedCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )
        self._get_conv_output()
        self.fc = nn.Linear(self._to_linear, 100)
        self.dequant = torch.quantization.DeQuantStub()

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.randn(1, 3, 360, 640)
            x = self.convs(x)
            self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.quant(x)
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
        x = self.dequant(x)
        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, seq_length):
        super(CNNLSTMModel, self).__init__()
        self.seq_length = seq_length
        self.time_distributed_cnn = TimeDistributedCNN()
        self.lstm_image = nn.LSTM(input_size=100, hidden_size=64, batch_first=True)
        self.lstm_lidar = nn.LSTM(input_size=1081, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64 * 2, 2)

    def forward(self, x_image, x_lidar):
        x_image = self.time_distributed_cnn(x_image)
        x_image, _ = self.lstm_image(x_image)
        x_lidar, _ = self.lstm_lidar(x_lidar)
        x = torch.cat((x_image[:, -1, :], x_lidar[:, -1, :]), dim=-1)
        x = self.fc(x)
        return x