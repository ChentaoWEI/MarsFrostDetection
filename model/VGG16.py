import torch
import torch.nn as nn
from torchvision import models

def initialize_vgg16(num_classes):
    # load pre-trained model
    model = models.vgg16(pretrained=True)

    # freeze all parameters
    for param in model.features.parameters():
        param.requires_grad = False

    # replace the last layer
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model
