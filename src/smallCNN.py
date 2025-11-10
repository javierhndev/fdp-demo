import torch
import torch.nn as nn

class smallCNN(nn.Module):
    def __init__(self):
        super(smallCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=5,kernel_size=5,stride=2,padding=1)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=nn.Conv2d(in_channels=5,out_channels=15,kernel_size=5,stride=2,padding=0)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=1)
        self.fc1=nn.Linear(375,100)
        self.relu3=nn.ReLU()
        self.drop1=nn.Dropout(0.2)
        self.fc2=nn.Linear(100,30)
        self.relu4=nn.ReLU()
        self.drop2=nn.Dropout(0.2)
        self.fc3=nn.Linear(30,1)

    def forward(self,x):
        #input batch*1*65*65  output batch*5*31*31
        x=self.conv1(x)
        x=self.relu1(x)
        #input batch*5*31*31 output batch*5*15*15
        x=self.pool1(x)
        #input batch*5*15*15 output batch*15*6*6
        x=self.conv2(x)
        x=self.relu2(x)
        #input batch*15*6*6 output batch*15*5*5
        x=self.pool2(x)
        
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)
        x=self.drop1(x)
        x=self.fc2(x)
        x=self.relu4(x)
        x=self.drop2(x)
        x=self.fc3(x)
        return x