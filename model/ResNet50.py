import torch
import torch.nn as nn
from torchvision import models

def initialize_resnet50(num_classes):
    # load pre-trained model
    model = models.resnet50(pretrained=True)

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
