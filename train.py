import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import CNNModel
from datasets import train_loader, test_loader

# Learning_parameters 
lr = 1e-3
epochs = 3

# Save the trained model to disk in order to inference later.
def save_model(epochs, model, optimizer, criterion):
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/model.pth')
    

# Function to save the loss and accuracy plots to disk.
def save_plots(train_acc, test_acc, train_loss, test_loss):
    plt.figure(figsize=(10, 7))                     # accuracy plots
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='Train accuracy'
    )
    plt.plot(
        test_acc, color='blue', linestyle='-', 
        label='Test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    plt.figure(figsize=(10, 7))                     # loss plots
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='Train loss'
    )
    plt.plot(
        test_loss, color='red', linestyle='-', 
        label='Test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')



device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

model = CNNModel().to(device)
print(model)


optimizer = optim.Adam(model.parameters(), lr=lr)   # Adam optimizer

criterion = nn.CrossEntropyLoss()                   # CrossEntropy loss function

# Training function
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(image)                                  # forward pass
        loss = criterion(outputs, labels)                       # calculate the loss
        train_running_loss += loss.item()                       # add the loss to the graph
        _, preds = torch.max(outputs.data, 1)                   # calculate the accuracy
        train_running_correct += (preds == labels).sum().item() # add the accuracy to the graph
        
        loss.backward()                                         # backward pass
        optimizer.step()                                        # ask optimizer to do a step
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Test function
def test(model, testloader, criterion):
    model.eval()
    print('Testing')
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)                                  # forward pass
            loss = criterion(outputs, labels)                       # calculate the loss
            test_running_loss += loss.item()                        # add the loss to the graph
            _, preds = torch.max(outputs.data, 1)                   # calculate the accuracy
            test_running_correct += (preds == labels).sum().item()  # add the accuracy to the graph
        
    # loss and accuracy for the complete epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# Define the lists that will make the graphs of losses and accuracies
train_loss, test_loss = [], []
train_acc, test_acc = [], []

# Training start
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                              optimizer, criterion)
    test_epoch_loss, test_epoch_acc = test(model, test_loader,  
                                                 criterion)
    train_loss.append(train_epoch_loss)
    test_loss.append(test_epoch_loss)
    train_acc.append(train_epoch_acc)
    test_acc.append(test_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training accuracy: {train_epoch_acc:.3f}")
    print(f"Test loss: {test_epoch_loss:.3f}, test accuracy: {test_epoch_acc:.3f}")
    print('-'*50)
    time.sleep(5)
    
# Save the trained model weights
save_model(epochs, model, optimizer, criterion)
# Save the loss and accuracy plots
save_plots(train_acc, test_acc, train_loss, test_loss)
print('---TRAINING COMPLETE---')

