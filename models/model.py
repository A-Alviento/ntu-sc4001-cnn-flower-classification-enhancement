import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import chain


# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, fcn_depth=3, num_classes=102, fcn_width=[1024, 512]):

        # specified depth must be 1 more than the number of specified widths (since fcn_width does not include the output layer)
        if len(fcn_width) != fcn_depth-1:
            raise ValueError("fcn_width must have length fcn_depth-1")

        super(CustomCNN, self).__init__()

        # Initialise the fully connected layers, so that we can unpack it in nn.Sequential
        fcn_list = []
        input_size = 5 * 5 * 512
        for width in fcn_width:
            fcn_list.extend([
                nn.Linear(input_size, width),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_size = width

        self.conv_stack = nn.Sequential(
            ## Convolutional Layers
            # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, stride=2),
        
            # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, stride=2),
        
            # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        
            # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            # paper details to skip the pooling layer after the fourth convolutional layer
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        
            # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),     
        )

        self.fcn_stack = nn.Sequential(
            # flatten the output of the last convolutional layer
            nn.Flatten(),

            # Fully-connected layers unpacked
            *fcn_list,
            
            # output layer
            nn.Linear(input_size, num_classes),
        )

    
    def forward(self, x):
        return self.fcn_stack(self.conv_stack(x))

    # Intended only for visualization - not for training
    def forward_with_intermediates(self, x):
        # return output of convolution layers (minus relu) and the output
        indexes = [0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 15, -1]
        ret_list = []
        interm = x

        with torch.no_grad():
            for layer in chain(self.conv_stack, self.fcn_stack):
                interm = layer(interm)
                ret_list.append(interm)
        
        return tuple(ret_list[i] for i in indexes)