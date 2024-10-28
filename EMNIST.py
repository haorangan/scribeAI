import torchmetrics
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import cv2


#### transform data
train_data = torchvision.datasets.EMNIST(
    root='./data',
    train=True,
    download=True,
    split='byclass',
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
test_data = torchvision.datasets.EMNIST(
    root='./data',
    train=False,
    download=True,
    split='byclass',
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
####

class EMNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 62),
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x


####Instantiate and train model
model = EMNISTNetwork()
model.load_state_dict(torch.load(f="./models/EMNISTmodel.pt", weights_only=True))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 5
"""
for epoch in range(epochs):
    train_loss = 0.0
    for batch, (images, labels) in enumerate(train_dataloader):
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch % 50 == 0:
            print(f'batch {batch} / {len(train_dataloader)}')
        if batch == 1000:
            break

    correct, total = 0, 0
    model.eval()
    with torch.inference_mode():
        for images, labels in test_dataloader:
            y_pred = model(images)
            y_pred = torch.softmax(y_pred, dim=1)
            correct += (torch.argmax(y_pred, dim=1).eq(labels).sum()).item()
            total += labels.size(0)
        print(f'Accuracy: {100 * correct/total}%')
    print(f'Epoch: {epoch}, Loss: {train_loss}')
####
"""

#### save model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)
MODEL_NAME = "EMNISTmodel.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)\
####