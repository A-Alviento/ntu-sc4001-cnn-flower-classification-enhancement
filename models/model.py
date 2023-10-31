import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from itertools import chain

class DeformConv2d(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(DeformConv2d, self).__init__()

        self.conv_offset = nn.Conv2d(in_nc, 2 * (kernel_size**2), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

        self.dcn_conv = torchvision.ops.DeformConv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        return self.dcn_conv(x, offset=offset)

# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, fcn_depth=3, num_classes=102, fcn_width=[1024, 512], batch_norm_moment=0.05):

        # specified depth must be 1 more than the number of specified widths (since fcn_width does not include the output layer)
        if len(fcn_width) != fcn_depth-1:
            raise ValueError("fcn_width must have length fcn_depth-1")

        super(CustomCNN, self).__init__()

        self.pinned_layers = []
        
        # Initialise the fully connected layers, so that we can unpack it in nn.Sequential
        fcn_list = []
        input_size = 5 * 5 * 512
        for width in fcn_width:
            fcn_list.extend([
                nn.BatchNorm1d(input_size, momentum=batch_norm_moment),
                nn.Linear(input_size, width),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_size = width

        self.conv_stack = nn.Sequential(
            ## Convolutional Layers
            # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
            nn.BatchNorm2d(3, momentum=None),  # Use total running average
            DeformConv2d(3, 64, kernel_size=5, stride=1, padding=2),
            # nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            # nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3),
            # self.pin(nn.Conv2d(3, 64, kernel_size=1)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
            nn.BatchNorm2d(64, momentum=batch_norm_moment),
            # DeformConv2d(64, 128, kernel_size=5, stride=1, padding=2),
            # nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64),
            self.pin(nn.Conv2d(64, 128, kernel_size=1)),
            nn.ReLU(),
            self.pin(nn.LocalResponseNorm(5, 0.0001, 0.75, 2)),
            self.pin(nn.MaxPool2d(3, stride=2)),
            
            # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
            nn.BatchNorm2d(128, momentum=batch_norm_moment),
            # DeformConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),
            self.pin(nn.Conv2d(128, 256, kernel_size=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),
        
            # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            # paper details to skip the pooling layer after the fourth convolutional layer
            nn.BatchNorm2d(256, momentum=batch_norm_moment),
            # DeformConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            self.pin(nn.Conv2d(256, 512, kernel_size=1)),
            nn.ReLU(),
        
            # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
            nn.BatchNorm2d(512, momentum=batch_norm_moment),
            # DeformConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512),
            self.pin(nn.Conv2d(512, 512, kernel_size=1)),
            nn.ReLU(),
            self.pin(nn.MaxPool2d(3, stride=2)),    
        )

        self.fcn_stack = nn.Sequential(
            # flatten the output of the last convolutional layer
            nn.Flatten(),

            # Fully-connected layers unpacked
            *fcn_list,
            
            # output layer
            nn.BatchNorm1d(input_size, momentum=batch_norm_moment),
            self.pin(nn.Linear(input_size, num_classes)),
        )

        # Corresponding sequence of pinned layers in forward operation
        self.pinned_indexes = [j for j, l in enumerate(chain(self.conv_stack, self.fcn_stack))
                                    if l in self.pinned_layers]

    def pin(self, layer):
        self.pinned_layers.append(layer)
        return layer

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