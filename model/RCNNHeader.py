import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PlateDetection.model.utils import widget
from PlateDetection.model.ROIPooling2d import ROIPooling2d
import random

class ProposalTargetCreator:
    """
    Assign ground truth to sampled proposals. This function is very similar to Anchor
    TargetCreator which assign ground truth labels and ground truth bounding box offsets
    to anchors.
    Returns:
       * **sampled_roi**: A tensor representing the rois picked out.
       * **gt_loc**: A tensor representing the corresponding bounding box offsets to the
       sampled rois.
       * **gt_label**: A tensor representing the corresponding label including background
       label to the sampled rois.
    """
    def __init__(self,n_sample=8,pos_thres=0.5,neg_thres=0.5,pos_ratio=0.25):
        self.n_sample=n_sample
        self.pos_thres=pos_thres
        self.neg_thres=neg_thres
        self.pos_ratio=pos_ratio
    def __call__(self,rois,roi_indices,bbox):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ious=widget.cal_iou(rois,bbox)
        gt_label=torch.ones(len(rois)).to(device)
        pos_index=torch.where(ious>=self.pos_thres)[0]
        neg_index=torch.where(ious<self.neg_thres)[0]

        pos_num=np.round(self.pos_ratio*self.n_sample)
        pos_num=int(pos_num)
        if len(pos_index)>pos_num:
            pos_index=pos_index[random.sample(list(range(0,len(pos_index))),pos_num)] 

        neg_num=self.n_sample-len(pos_index)
        if len(neg_index)>neg_num:
            neg_index=neg_index[random.sample(list(range(0,len(neg_index))),neg_num)]

        keep_index=torch.cat((pos_index, neg_index),dim=0)
        assert len(keep_index)!=0
        gt_label[neg_index]=0
        gt_label=gt_label[keep_index]
        sample_roi=rois[keep_index]
        sample_roi_indices=roi_indices[keep_index]
        gt_loc=widget.b2l(sample_roi, bbox)
        gt_label=gt_label.long()
        assert len(sample_roi)!=0

        #batch_size greater than 1 not supported. Problems remaining in calculate loc and label
        return sample_roi,sample_roi_indices,gt_loc,gt_label

class RCNNHeader(nn.Module):
    """
    This class is used as the head of Faster RCNN. This layer output the localization and
    classification based on the extracted features.

    """
    def __init__(self,roi_size=5,spatial_scale=1./16):
        super(RCNNHeader,self).__init__()
        
        self.roi_pooling=ROIPooling2d(roi_size,spatial_scale)
        self.fc1=nn.Linear(256*roi_size*roi_size,1024)
        self.fc2=nn.Linear(1024,1024)
        self.cls_loc=nn.Linear(1024,2*4)
        self.score=nn.Linear(1024,2)

        self.roi_size=roi_size
        self.spatial_scale=spatial_scale

    def forward(self,x,rois,roi_indices):
        try:
            pool=self.roi_pooling(x,rois,roi_indices)
        except Exception as e:
            print(x.size(),rois.size(),roi_indices.size())
        pool=pool.view(pool.size(0),-1)
        pool=F.relu(self.fc1(pool))
        pool=F.relu(self.fc2(pool))

        roi_cls_locs=self.cls_loc(pool)
        roi_scores=self.score(pool)

        return roi_cls_locs,roi_scores





                        
