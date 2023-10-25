import timm
import torch
import torch.nn as nn

class Simple(nn.Module):

    def __init__(self, num_classes=3, model_name="tv_densenet121", pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes)  # 224/2 = 112 (max pooling)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x