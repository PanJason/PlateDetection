import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class PlateDetector(nn.Module):
    """
    This is the new plate bbox detection network based on YOLO.
    Returns:
    * **xProb**: The probablity of the existence of the bounding box
    * **xAffine**: The Affine parameter of the base bounding box.
    """
    def __init__(self):
        super(PlateDetector,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.conv5=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv6=nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv7=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv8=nn.Conv2d(128,128,kernel_size=3,padding=1)

        self.prob=nn.Conv2d(128,2,kernel_size=3,padding=1)
        self.affine=nn.Conv2d(128,6,kernel_size=3,padding=1)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv3(x))
        #ResBlock
        x_res=F.relu(self.conv4(x))
        x_res=self.conv4(x_res)
        x=x+x_res
        x=F.relu(x)

        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv5(x))

        #ResBlock
        x_res1=F.relu(self.conv6(x))
        x_res1=self.conv6(x)
        x=x+x_res1
        x=F.relu(x)
        x_res2=F.relu(self.conv6(x))
        x_res2=self.conv6(x)
        x=x+x_res2
        x=F.relu(x)
        x=F.max_pool2d(x,(2,2))

        x_res3=F.relu(self.conv6(x))
        x_res3=self.conv6(x)
        x=x+x_res3
        x=F.relu(x)
        x_res4=F.relu(self.conv6(x))
        x_res4=self.conv6(x)
        x=x+x_res4
        x=F.relu(x)
        x=F.max_pool2d(x,(2,2))

        x=self.conv7(x)

        x_res5=F.relu(self.conv8(x))
        x_res5=self.conv8(x)
        x=x+x_res5
        x=F.relu(x)
        x_res6=F.relu(self.conv8(x))
        x_res6=self.conv8(x)
        x=x+x_res6
        x=F.relu(x)
        
        xProb=self.prob(x)
        xProb=F.log_softmax(xProb,dim=1)

        xAffine=self.affine(x)
        return xProb,xAffine
