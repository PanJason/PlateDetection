import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PlateDetection.model.utils import widget
import random

class ProposalCreator:
    """
    This class is used to generate 10 ROIs
    Returns:
       * **roi**: A tensor represents the regions of Interest
    """
    def __init__(self, nms_thresh=0.7, n_pre_nms=1000,n_post_nms=50,min_size=16):
        self.nms_thresh=nms_thresh
        self.n_pre_nms=n_pre_nms
        self.n_post_nms=n_post_nms
        self.min_size=min_size

    def __call__(self,loc,score,anchor,img_size,scale=1.0):
        roi=widget.l2b(anchor,loc)

        #clip bbox sizes 
        roi[:,0]=torch.clamp(roi[:,0],0,img_size[0])
        roi[:,1]=torch.clamp(roi[:,1],0,img_size[1])
        roi[:,2]=torch.clamp(roi[:,2],0,img_size[0])
        roi[:,3]=torch.clamp(roi[:,3],0,img_size[1])        

        #Remove bbox beyond threshold
        size_thres=self.min_size*scale
        keep=torch.where((roi[:,2]-roi[:,0]>=size_thres) & (roi[:,3]-roi[:,1]>=size_thres))[0]
        roi=roi[keep,:]
        score=score[keep]

        #Sort and keep top n_pre_nms bboxs
        if len(keep)>self.n_pre_nms:
            scores_pre_nms,indices=score.topk(k=self.n_pre_nms,dim=0,largest=True)
            roi=roi[indices,:]
        else:
            scores_pre_nms=score
        
        

        #print(scores_pre_nms)
        #print(roi)
        #Apply nms
        keep=widget.nms(roi.data,scores_pre_nms.data,self.nms_thresh,self.n_post_nms)
        roi=roi[keep,:]

        return roi

class AnchorTargetGenerator:
    """
    Assign ground truth bounding boxes and its labels to Anchor. These return values are
    used to calculate RPN losses.
    Returns:
       * **TrueLoc**: A Tensor representing the real transform parameters that transform
       the anchor to the bounding box which has the highest IoU with it.
       * **TrueLabel**: A Tensor representing the whether the anchor bounds foreground or
       back ground object. 
    """

    def __init__(self,n_sample=64,pos_thres=0.6,neg_thres=0.3,pos_ratio=0.5):
        self.n_sample=n_sample
        self.pos_thres=pos_thres
        self.neg_thres=neg_thres
        self.pos_ratio=pos_ratio
    def __call__(self,anchor,bbox,img_size):
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_H,img_W=img_size
        n_anchor=anchor.size()[0]
        inside_indices=torch.where((anchor[:, 0]>=0)&(anchor[:, 1]>=0)&(anchor[:, 2]<=img_H)&
        (anchor[:, 3]<=img_W))[0]
        anchor=anchor[inside_indices]
        label=self.get_label(inside_indices,anchor,bbox)
        loc=widget.b2l(anchor,bbox)

        #Here may be some problems about dimension 
        trueLabel=-1*torch.zeros(n_anchor).to(device)
        trueLabel[inside_indices]=label
        trueLabel=trueLabel.long()

        trueLoc=torch.zeros((n_anchor,4)).to(device)
        trueLoc[inside_indices,:]=loc

        return trueLoc,trueLabel

    def get_label(self, inside_indices, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        label=-1*torch.ones(len(inside_indices)).to(device)
        ious=widget.cal_iou(anchor, bbox)

        ni=torch.where(ious<self.neg_thres)[0]
        pi=torch.where(ious>=self.pos_thres)[0]

        label[ni]=0
        label[pi]=1

        # subsample positive labels if we have too many
        n_pos=int(self.pos_ratio*self.n_sample)
        pos_index=torch.where(label==1)[0]
        if len(pos_index)>n_pos:
            disable_index=pos_index[random.sample(list(range(0,len(pos_index))),len(pos_index)-n_pos)]
            label[disable_index]=-1

        # subsample negative labels if we have too many
        n_neg=self.n_sample-torch.sum(label==1)
        neg_index=torch.where(label==0)[0]
        if len(neg_index)>n_neg:
            disable_index=neg_index[random.sample(list(range(0,len(neg_index))),len(neg_index)-n_neg)]
            label[disable_index]=-1

        return label