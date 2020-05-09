import torch as t
import torch.nn as nn
from torch.nn import functional as F

class FasterRCNN(nn.Module):
    """This is the base class for Faster RCNN.
    It is composed of following three parts.
    1. **Feature Extraction**: This stage is to extract and down sampling features from the given image,\
    using conv layers and pooling layers.
    2. **Region Proposal Network**: Given the features extracted, this stage decides Regions of Interest\
     (RoIs) of the objects.
    3. **Localization and Classification Heads**: This stage uses the extracted features and RoIs to \
    classify the categories of the objects and promote localization performances.
    """
    def __init__(self,extractor,rpn,head):
        super(FasterRCNN,self).__init__()
        self.extractor=extractor
        self.rpn=rpn;
        self.head=head
    def forward(self,x,scale=1.):
        xFeatured=self.extractor(x)
        rpn_locs,rpn_scores,rois,roi_indices,anchor=self.rpn(xFeatured,x.size()[2:],scale)
        roi_cls_locs,roi_scores=self.head(xFeatured,rois,roi_indices)
        return roi_cls_locs,roi_scores,rois,roi_indices