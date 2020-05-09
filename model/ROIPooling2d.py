import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class ROIPooling2d(nn.Module):
    """
    This is the ROI pooling layer which down sample each RoI in the extracted feature
    to the same roi_size.
    """
    def __init__(self,roi_size,spatial_scale):
        super(ROIPooling2d,self).__init__()
        self.roi_size=roi_size
        self.spatial_scale=spatial_scale
        self.pooling=nn.AdaptiveAvgPool2d(roi_size)
    def forward(self,x,rois,roi_indices):
        rois=rois.data.float().clone()
        rois.mul_(self.spatial_scale)
        rois=rois.long()
        roi_indices=roi_indices.data.long().clone()
        output=[]

        for i in range(rois.size(0)):
            roi=rois[i]
            batch_indice=roi_indices[i]
            try:
                roi_feature = x[batch_indice, :, roi[0]:(roi[2]+1), roi[1]:(roi[3]+1)]
            except Exception as e:
                print(e, roi)
            pool_feature = self.pooling(roi_feature)
            output.append(pool_feature.unsqueeze(0))
        return torch.cat(output, 0)       