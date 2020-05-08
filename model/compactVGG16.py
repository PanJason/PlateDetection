import torch as t
import torch.nn as nn
from torch.nn import functional as F

class compactVGG16(nn.Module):
    """
    The compactVGG16 is a compact version of VGG16. Since the given pictures are of \
        smaller size so too many layers are not necessary. The compactVGG16 only\
            utilized 6 conv+Relu layers with 4 maxpooling layers. The dropout layers\
                may be introduced later.
    Returns:
       * **Tensor**: Its shape is :math:`batch_size \\times 256 \\times H/16 \\times W/16`
    """
    def __init__(self):
        super(compactVGG16,self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv4=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv5=nn.Conv2d(256,256,kernel_size=3,padding=1)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv3(x))
        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv4(x))
        x=F.max_pool2d(x,(2,2))
        x=F.relu(self.conv5(x))
        x=F.relu(self.conv5(x))
        return x