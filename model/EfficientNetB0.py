import torch
import torch.nn as nn
from torchvision import models

def initialize_efficientnetb0(num_classes):
    # load pre-trained model
    model = models.efficientnet_b0(pretrained=True)

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # replace the last layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model
