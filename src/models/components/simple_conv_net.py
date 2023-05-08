import torch.nn as nn


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        conv3_channels: int = 256,
        fc1_size: int = 512,
        fc2_size: int = 256,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv1_channels),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv2_channels),
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv3_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(conv3_channels * 4 * 4, fc1_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc2_size, 100),
        )

    def forward(self, x):
        x = self.features(x)

        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        x = self.classifier(x)
        return x
