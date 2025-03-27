# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Conv1: (batch_size, 32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Pool1: (batch_size, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Conv2: (batch_size, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Pool2: (batch_size, 64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# Conv3: (batch_size, 128, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # Pool3: (batch_size, 128, 4, 4)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),                                          # Flatten: (batch_size, 128*4*4)
            nn.Linear(128 * 4 * 4, 256),                           # FC1: (batch_size, 256)
            nn.ReLU(),
            nn.Dropout(0.5),                                       # Dropout for regularization
            nn.Linear(256, 10)                                     # FC2: (batch_size, 10)
        )

    def forward(self, x):

        ##########################################
        # Define the forward path of your model. #
        ##########################################

        x = self.conv_layers(x)  # Pass through convolutional layers
        x = self.fc_layers(x)    # Pass through fully connected layers
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrained weights on ResNet18 are used. #
        ############################################

        # Load the ResNet18 model with pretrained weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Use pretrained weights

        # Modify the first convolutional layer to reduce kernel size and stride
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Replace the first maxpool layer with an identity layer
        self.resnet.maxpool = nn.Identity()

        # Modify the fully connected layer to output 10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
