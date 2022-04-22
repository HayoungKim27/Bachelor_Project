import torch
import torch.nn as nn
from torchvision import models
import code

class vggnet16(nn.Module):
    def __init__(self):
        super(vggnet16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 5)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x



class vggnet19(nn.Module):
    def __init__(self):
        super(vggnet19, self).__init__()
        self.features = models.vgg19(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 5)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
