import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from IMU_data_process.imu_dataset import IMU_dataset

class IMU_Network(nn.Module):
    def __init__(self):
        super(IMU_Network, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 10), stride=10),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(True),
                                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=3),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=(3, 10), stride=10)
        # self.bn = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,3), stride=3)
        self.fc1 = nn.Linear(in_features=2560, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=27)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





if __name__ == '__main__':
    train()



