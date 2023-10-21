import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

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


# Training step
def train_step(model, trainloader, optimizer, device, loss):
    model.train() # set model to training mode
    total_loss = 0.0 

    # Iterate over the training data
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # get the inputs and labels
        inputs, labels = inputs.to(device), labels.to(device) # move them to the device

        optimizer.zero_grad() # zero the gradients

        # Forward pass
        _, _, _, _, _, _, _, _, _, outputs = model(inputs)
        loss = loss(outputs, labels) 

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() # accumulate the loss

    train_loss = total_loss / len(trainloader)
    return train_loss


# Test step
def val_step(model, valloader, loss, device):
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
            loss = loss(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability

            total += labels.size(0) # accumulate total number of labels
            correct += (predicted == labels).sum().item() # accumulate correct predictions

    val_loss = total_loss / len(valloader)
    accuracy = 100 * correct / total

    return val_loss, accuracy