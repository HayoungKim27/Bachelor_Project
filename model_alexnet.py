import torch
import torch.nn as nn
from torchvision import models
import code

# performs transfer learning by loading pre-trained CNN model
class alexnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.alexnet(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 5)

    def forward(self, x):
        features = self.features(x)   # x.shape = [1, 3, 256, 256], features.shape = [1, 256, 7, 7]
        pooled_features = self.avgpool(features)   # pooled_features.shape = [1, 256]
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer(flattened_features)    # output.shape = [1, 1]
        
        return output