import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain

# Skeleton of CNN with some useful functions
class SkeletonCNN(nn.Module):
    def __init__(self, fcn_depth, fcn_width):
        # specified depth must be 1 more than the number of specified widths (since fcn_width does not include the output layer)
        if len(fcn_width) != fcn_depth-1:
            raise ValueError("fcn_width must have length fcn_depth-1")

        super().__init__()
        self.pinned_layers = []
        self.pinned_indexes = None

        # To be added by models implementing this
        self.conv_stack = None
        self.fcn_stack = None

    def pin(self, layer):
        self.pinned_layers.append(layer)
        return layer

    # To be called at the end of child class __init__
    def calculate_pin_indexes(self):
        # Corresponding sequence of pinned layers in forward operation
        self.pinned_indexes = [j for j, l in enumerate(chain(self.conv_stack, self.fcn_stack))
                                    if l in self.pinned_layers]

    def forward(self, x):
        return self.fcn_stack(self.conv_stack(x))

    # Intended only for visualization - not for training
    def forward_with_intermediates(self, x):
        # return output of convolution layers (minus relu) and the output
        interm_list = []
        interm = x

        with torch.no_grad():
            for layer in chain(self.conv_stack, self.fcn_stack):
                interm = layer(interm)
                interm_list.append(interm)
        
        return tuple(interm_list[i] for i in self.pinned_indexes)

# Baseline CNN Model
class BaselineCNN(SkeletonCNN):
    def __init__(self, num_classes=102, fcn_depth=3, fcn_width=[1024, 512], dropout_rate=0.5):
        super().__init__(fcn_depth, fcn_width)

        self.conv_stack = nn.Sequential(
            ## Convolutional Layers
            # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
            self.pin(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
            self.pin(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
            
            # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
            self.pin(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            # paper details to skip the pooling layer after the fourth convolutional layer
            self.pin(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        
            # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            self.pin(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),    
        )

        # Initialise the fully connected layers in a list, so that we can unpack it in nn.Sequential
        fcn_list = []
        input_size = 5 * 5 * 512
        for width in fcn_width:
            fcn_list.extend([
                nn.Linear(input_size, width),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_size = width

        self.fcn_stack = nn.Sequential(
            # flatten the output of the last convolutional layer
            nn.Flatten(),

            # Fully-connected layers unpacked
            *fcn_list,
            
            # output layer
            self.pin(nn.Linear(input_size, num_classes)),
        )

        self.calculate_pin_indexes()

# BatchNorm CNN Model
class BatchNormCNN(SkeletonCNN):
    def __init__(self, num_classes=102, fcn_depth=3, fcn_width=[1024, 512], dropout_rate=0.5, batchnorm_moment=0.05):
        super().__init__(fcn_depth, fcn_width)

        self.conv_stack = nn.Sequential(
            ## Convolutional Layers
            # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
            nn.BatchNorm2d(3, momentum=None),
            self.pin(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
            nn.BatchNorm2d(64, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
            
            # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
            nn.BatchNorm2d(128, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            # paper details to skip the pooling layer after the fourth convolutional layer
            nn.BatchNorm2d(256, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        
            # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            nn.BatchNorm2d(512, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),    
        )

        # Initialise the fully connected layers in a list, so that we can unpack it in nn.Sequential
        fcn_list = []
        input_size = 5 * 5 * 512
        for width in fcn_width:
            fcn_list.extend([
                nn.BatchNorm1d(input_size, momentum=batchnorm_moment),
                nn.Linear(input_size, width),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_size = width

        self.fcn_stack = nn.Sequential(
            # flatten the output of the last convolutional layer
            nn.Flatten(),

            # Fully-connected layers unpacked
            *fcn_list,
            
            # output layer
            nn.BatchNorm1d(input_size, momentum=batchnorm_moment),
            self.pin(nn.Linear(input_size, num_classes)),
        )

        self.calculate_pin_indexes()

# BatchNorm with extraneous batchnorms reduced
class BatchNormReducedCNN(SkeletonCNN):
    def __init__(self, num_classes=102, fcn_depth=3, fcn_width=[1024, 512], dropout_rate=0.5, batchnorm_moment=0.05):
        super().__init__(fcn_depth, fcn_width)

        self.conv_stack = nn.Sequential(
            ## Convolutional Layers
            # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
            nn.BatchNorm2d(3, momentum=None),
            self.pin(nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
            nn.BatchNorm2d(64, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
            
            # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
            nn.BatchNorm2d(128, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            # paper details to skip the pooling layer after the fourth convolutional layer
            nn.BatchNorm2d(256, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        
            # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            nn.BatchNorm2d(512, momentum=batchnorm_moment),
            self.pin(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),    
        )

        # Initialise the fully connected layers in a list, so that we can unpack it in nn.Sequential
        fcn_list = []
        input_size = 5 * 5 * 512
        for width in fcn_width:
            fcn_list.extend([
                nn.Linear(input_size, width),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_size = width

        self.fcn_stack = nn.Sequential(
            # flatten the output of the last convolutional layer
            nn.Flatten(),

            # Fully-connected layers unpacked
            *fcn_list,
            
            # output layer
            nn.BatchNorm1d(input_size, momentum=batchnorm_moment),
            self.pin(nn.Linear(input_size, num_classes)),
        )

        self.calculate_pin_indexes()