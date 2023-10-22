import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchvision import datasets, transforms
import os
from tqdm import tqdm 
import time  
from PIL import Image
import cv2
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(), # convert image to tensor
    # standardise the dimensions of the image to 100x100
    transforms.Resize((100, 100)),
    # normalise the image by setting the mean and std to 0.5
    transforms.Normalize(mean=0.5, std=0.5)
])

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

        # Forward pass
        _, _, _, _, _, _, _, _, _, outputs = model(inputs)
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
    total = 0

    with torch.no_grad(): # disable gradient calculation
        for data in valloader:
            inputs, labels = data # get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device) # move them to the device

            # Forward pass
            _, _, _, _, _, _, _, _, _, outputs = model(inputs)
            loss = lossfn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability

            total += labels.size(0) # accumulate total number of labels
            correct += (predicted == labels).sum().item() # accumulate correct predictions

    val_loss = total_loss / len(valloader)
    accuracy = 100 * correct / total

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


