import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Linear(128*3*3, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 10))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = x.view(x.size(0), -1); return self.fc(x)
