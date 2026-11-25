import torch
import torch.nn as nn
from torchvision import models


class EmotionEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(2048, 768)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(768, 256)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, return_features=False):
        x = self.backbone(x)
        x = self.flatten(x)

        feat = self.relu1(self.fc1(x))

        x = self.fc2(feat)
        va = self.head(x)

        if return_features:
            return va, feat
        return va