import parameter
import torch
import torch.nn as nn
from models import VGG11
from dataset import TrainDataset, ValDataset, data_split
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
from matplotlib import pyplot as plt


DATA_PATH = parameter.data_path
TEST_PATH = parameter.test_path
CUDA_DEVICE = parameter.cuda_device
MODEL_PATH_SVM = parameter.model_path_svm
num_classes = len(os.listdir(DATA_PATH))


def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x_train, x_val, y_train, y_val = data_split()
    train_set = TrainDataset(x_train, y_train, data_transform)
    val_set = ValDataset(x_val, y_val, data_transform)
    train_data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=2)
    val_data_loader = DataLoader(dataset=val_set, batch_size=32, shuffle=True, num_workers=2)

    model = VGG11(num_classes=num_classes).cuda(CUDA_DEVICE)

    best_acc = 0.0
    num_epochs = 100
    training_curve, val_curve = [], []
    loss_func = nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0
        val_loss = 0.0
        val_corrects = 0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.cuda(CUDA_DEVICE)
            labels = labels.cuda(CUDA_DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_corrects += torch.sum(preds == labels.data).float()
            
        for i, (inputs, labels) in enumerate(val_data_loader):
            inputs = inputs.cuda(CUDA_DEVICE)
            labels = labels.cuda(CUDA_DEVICE)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_func(outputs, labels)

            val_loss += loss.item()
            val_corrects += torch.sum(preds == labels.data).float()
        
        training_loss = training_loss / len(train_set)
        training_curve.append(training_loss)
        training_acc = training_corrects / len(train_set)
        
        val_loss = val_loss / len(val_set)
        val_curve.append(val_loss)
        val_acc = val_corrects / len(val_set)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f} /\
            validation loss: {val_loss:.4f}\tvalidation accuracy: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, MODEL_PATH_SVM)
        
        
    plt.plot(training_curve, 'r', label="training loss")
    plt.plot(val_curve, 'b', label="validation loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss_SVM')
    plt.savefig('loss_curve_svm.png')
    plt.show()

if __name__ == '__main__':
    train()
