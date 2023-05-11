import torch.nn as nn
from torchvision.models import resnet18


class CIFARResnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=False, num_classes=100)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)
