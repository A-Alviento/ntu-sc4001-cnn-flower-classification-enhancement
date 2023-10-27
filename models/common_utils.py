import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm 
import time  
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

class CustomTransform:
    def __init__(self, mean_rgb):
        self.mean_rgb = mean_rgb

    def __call__(self, image):
        # convert pil image to numpy array
        image = np.array(image)

        # get saliency map
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliency_map) = saliency.computeSaliency(image)
        saliency_map = (saliency_map * 255).astype("uint8")

        # get luminance map
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        luminance_map = l_channel / 255.0

        combined_map = saliency_map * luminance_map

        y_0, y_1, x_0, x_1 = find_bounding_box(combined_map)

        # Crop and resize the image to 100x100
        cropped_resized_img = Image.fromarray(image[y_0:y_1, x_0:x_1]).resize((100, 100), Image.ANTIALIAS)

        # Convert to tensor
        tensor_img = transforms.ToTensor()(cropped_resized_img)
        
        # Subtract mean RGB values
        tensor_img -= self.mean_rgb.clone().detach().view(3, 1, 1)
        # clip to [0, 1]
        tensor_img = torch.clamp(tensor_img, 0, 1)

        return tensor_img


# Generate a kernel where the center is focused
def center_focused_kernel(v, exp_factor = 0.5):
    x, y = torch.meshgrid(torch.linspace(-1, 1, v, dtype=torch.float64),
                          torch.linspace(-1, 1, v, dtype=torch.float64))
    distance = torch.exp(np.sqrt(x**2 + y**2))
    kernel = torch.exp(-1 * exp_factor * distance)
    kernel /= kernel.max()
    
    return kernel


# Find a square bounding box of combined_map
def find_bounding_box(combined_map, conv_stride = 50):
    w, h = combined_map.shape
    v = min(w, h)
    combined_map_torch = torch.tensor(combined_map).view(1, 1, w, h)

    max_value = -1 * np.inf
    y_0, y_1, x_0, x_1 = None, None, None, None
    
    # start from smallest of width and height and decrease by -25 until 200 pixels
    for v in range(min(w,h), 100-1, -100):
        kernel = center_focused_kernel(v)
        kernel = kernel.view(1, 1, v, v) / ( v * v )
        res = F.conv2d(combined_map_torch, kernel, stride = conv_stride)
        
        _, _, t1, t2 = res.shape
        res = res.reshape(t1, t2)

        curr_max = torch.max(res)

        # Penalise small bounding boxes
        curr_value = curr_max * (v / w)
        
        if curr_value > max_value:
            max_value = curr_value
            t = (res == curr_max).nonzero()

            y_0 = t[0, 0].item()
            y_1 = y_0 + v
            
            x_0 = t[0, 1].item()
            x_1 = x_0 + v

        
    return y_0, y_1, x_0, x_1


# get mean rgb values of training dataset
def get_mean_rgb(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # this is a dataloader that loads one image at a time
    mean = torch.zeros(3) # initialise mean tensor

    for i, (x, y) in enumerate(train_loader):
        mean += x.mean(dim=(0,2,3)) # sum up the mean of each channel
    mean /= len(train_dataset) # divide by number of images

    return mean


# set random seed
def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 

# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience # how many epochs to wait before stopping when loss is no longer decreasing
        self.min_delta = min_delta # minimum difference between new loss and old loss to be considered as a decrease in loss
        self.counter = 0 # number of epochs since loss was last decreased
        self.min_validation_loss = np.inf # minimum validation loss achieved so far

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss: # new loss is lower than old loss
            self.min_validation_loss = validation_loss # update minimum loss
            self.counter = 0 # reset counter
        elif validation_loss > (self.min_validation_loss + self.min_delta): # new loss is higher than old loss + minimum difference
            self.counter += 1 # increase counter
            if self.counter >= self.patience:
                return True # stop training
        return False # continue training


# Train step
def train_step(model, trainloader, optimizer, device, lossfn):
    model.train()  # set model to training mode
    total_loss = 0.0

    # Iterate over the training data
    for i, data in trainloader:
        inputs, labels = data  # get the inputs and labels
        inputs, labels = inputs.to(device), labels.to(device)  # move them to the device

        optimizer.zero_grad()  # zero the gradients

        outputs = model(inputs)
        loss = lossfn(outputs, labels)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # accumulate the loss
        trainloader.set_postfix({'Training loss': '{:.4f}'.format(total_loss/(i+1))})  # Update the progress bar with the training loss

    train_loss = total_loss / len(trainloader)
    return train_loss


# Test step
def val_step(model, valloader, lossfn, device):
    model.eval() # set model to evaluation mode
    total_loss = 0.0
    correct = 0

    with torch.no_grad(): # disable gradient calculation
        for data in valloader:
            inputs, labels = data # get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device) # move them to the device

            outputs = model(inputs)
            loss = lossfn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability

            correct += (predicted == labels).sum().item() # accumulate correct predictions

    val_loss = total_loss / len(valloader)
    accuracy = 100 * correct / len(valloader.dataset)

    return val_loss, accuracy

# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Training loop
def train(model, tl, vl, opt, loss, device, epochs, early_stopper, path):
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()  # Record the start time of the epoch

        # Wrap the trainloader with tqdm for the progress bar
        pbar = tqdm(enumerate(tl), total=len(tl), desc=f"Epoch {epoch+1}/{epochs}")

        train_loss = train_step(model, pbar, opt, device, loss)  # Pass the tqdm-wrapped loader
        val_loss, val_acc = val_step(model, vl, loss, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # Print time taken for epoch
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'Epoch {epoch+1}/{epochs} took {elapsed_time:.2f}s | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.2f}% | EarlyStopper count: {early_stopper.counter}')

        # save as last_model after every epoch
        save_model(model, os.path.join(path, 'last_model.pt'))

        # save as best_model if validation loss is lower than previous best validation loss
        if val_loss < early_stopper.min_validation_loss:
            save_model(model, os.path.join(path, 'best_model.pt'))

        if early_stopper.early_stop(val_loss):
            print('Early stopping')
            break

    return train_loss_list, val_loss_list, val_acc_list


