import torch.nn as nn
from torchvision.models import resnet18


class CIFARResnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 100)

    def forward(self, x):
        return self.model(x)
