import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, dropout_rate=0.5, fcn_depth=3, num_classes=102, fcn_width=[1024, 512]):

        # specified depth must be 1 more than the number of specified widths (since fcn_width does not include the output layer)
        if len(fcn_width) != fcn_depth-1:
            raise ValueError("fcn_width must have length fcn_depth-1")

        super(CustomCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2) # takes an input with 3 channels, applies 64 kernels of size 5x5 and outputs 64 feature map
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2) # takes an input with 64 channels, applies 128 kernels of size 5x5 and outputs 128 feature map
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # takes an input with 128 channels, applies 256 kernels of size 3x3 and outputs 256 feature map
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) # takes an input with 256 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1) # takes an input with 512 channels, applies 512 kernels of size 3x3 and outputs 512 feature map
        
        self.flatten = nn.Flatten() # flatten the output of the last convolutional layer
        
        # Dynamic Fully Connected Layers
        self.fcs = nn.ModuleList() # create a list of fully connected layers
        input_size = 5*5*512 # input size of the first fully connected layer

        # Add the fully connected layers to the list    
        for width in fcn_width:
            self.fcs.append(nn.Linear(input_size, width))
            input_size = width
        # output layer
        self.fc_final = nn.Linear(input_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) 

        # local response normalization
        self.lrn = nn.LocalResponseNorm(5, 0.0001, 0.75, 2)
        
    def forward(self, x):
        
        # First Convolutional Layer
        conv1 = F.relu(self.conv1(x))
        conv1_lrn = self.lrn(conv1)
        p1 = F.max_pool2d(conv1_lrn, 3, stride=2)
        
        # Second Convolutional Layer
        conv2 = F.relu(self.conv2(p1))
        conv2_lrn = self.lrn(conv2)
        p2 = F.max_pool2d(conv2_lrn, 3, stride=2)
        
        # Third Convolutional Layer
        conv3 = F.relu(self.conv3(p2))
        p3 = F.max_pool2d(conv3, 3, stride=2)
        
        # Fourth Convolutional Layer
        conv4 = F.relu(self.conv4(p3))
        # paper details to skip the pooling layer after the fourth convolutional layer

        # Fifth Convolutional Layer
        conv5 = F.relu(self.conv5(conv4))
        p5 = F.max_pool2d(conv5, 3, stride=2)

        # Flattening before Fully Connected Layers
        flatten = self.flatten(p5)
        
        # Dynamic Fully Connected Layers
        for fc in self.fcs:
            x = F.relu(fc(flatten))
            x = self.dropout(x)
            flatten = x  # Update flatten for the next layer

         # Output layer
        x = self.fc_final(flatten)

        # Softmax applied to the output layer
        out = x
        
        return conv1, p1, conv2, p2, conv3, p3, conv4, conv5, p5, out












