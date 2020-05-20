import numpy as np
import os
import cv2
import torch 
import torch.nn as nn
import torch 
import torch.optim as optim
import torch.nn.functional as F
import math
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc=nn.Sequential(
            nn.Linear((20//4)*(20//4)*64, 1024),
            nn.Dropout(0.5), 
            nn.ReLU())
        self.rfc=nn.Sequential(
            nn.Linear(1024, 34),
        )

    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=out.view(out.size(0), -1)
        out=self.fc(out)
        out=self.rfc(out)
        out=F.log_softmax(out)
        return out


#This is the model actually used.
class CNN_adv2(nn.Module):
    def __init__(self):
        super(CNN_adv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(32, 10, kernel_size=1)

        self.fc1 = nn.Linear((20//4) * (20//4) * 10, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 34)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.avg_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x

        
KERNEL_SIZE=5
class LeNet5(nn.Module):
    def __init__(self,kernel_channel1=5,kernel_channel2=15):
        super(LeNet5,self).__init__()
        self.kernel_channel1=kernel_channel1
        self.kernel_channel2=kernel_channel2
        self.conv1=nn.Conv2d(1,kernel_channel1,KERNEL_SIZE,padding=2)
        self.conv2=nn.Conv2d(kernel_channel1,kernel_channel2,KERNEL_SIZE,padding=2)
        self.fc1=nn.Linear(kernel_channel2*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,34)
        self.logsoftmax=nn.LogSoftmax(dim=-1)
    def forward(self,x):
        x=self.conv1(x)
        x=F.max_pool2d(F.relu(x),(2,2))
        x=self.conv2(x)
        x=F.max_pool2d(F.relu(x),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        x=self.logsoftmax(x)
        return x
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features
