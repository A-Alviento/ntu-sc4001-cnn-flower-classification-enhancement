from common_utils import *
from model import *
import cv2
import torch
import torch.nn as nn
import torch as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class cnn_grad_cam(nn.Module):
    def __init__(self, model_class, model_path, device, last_conv_layer_idx):
        super(cnn_grad_cam, self).__init__()
        
        # get the pretrained VGG19 network
        self.cnn_grad_cam = model_class
        self.cnn_grad_cam.load_state_dict(torch.load(model_path, map_location=device))
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.cnn_grad_cam.conv_stack[:last_conv_layer_idx]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(3, stride=2)
        
        # get the classifier of the vgg19
        self.classifier = self.cnn_grad_cam.fcn_stack
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    

# return the heatmap
def get_heatmap(model, img): # ensure model is in eval mode
    # get the most likely prediction of the model
    preds = model(img)

    # get the score and index corresponding to the most likely class
    score, indices = preds.max(dim=1) 
    model.zero_grad() # zero the gradients
    score.backward() # backward pass to calculate the gradient of the highest score w.r.t. input image

    # get gradient for class activation 
    gradient = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])

    # get activations of the last convolutional layer
    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    # average the channels of the activations to get heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()

    # apply relu to keep features with positive influence
    heatmap = F.relu(heatmap)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap.numpy()


# overlay heatmap on image
def save_grad_map(img, heatmap, path):
    # resize heatmap to size of image with aliasing
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # convert to 0-255 uint8 scale
    heatmap = np.uint8(255 * heatmap)
    # use jet colormap to colorize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # apply heatmap to image
    superimposed_img = heatmap * 0.4 + np.uint8(255*img)

    # save image
    cv2.imwrite(path, superimposed_img)



class CustomFlowers102(Dataset):
    def __init__(self, root, split, download=True, transform=None):
        self.root = root
        self.dataset = datasets.Flowers102(root=root, split=split, download=download, transform=None)
        self.transform = transform

    def __getitem__(self, index):
        # Get the image file path from the underlying dataset
        img_file, _ = self.dataset._image_files[index], self.dataset._labels[index]

        img, label = self.dataset[index]

        # Apply the transformation
        if self.transform:
            img = self.transform(img)

        # Return the transformed image, label, and file name
        return img, label, str(img_file)

    def __len__(self):
        return len(self.dataset)



