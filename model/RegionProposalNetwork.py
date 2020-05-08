import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PlateDetection.model.creator import ProposalCreator

def generate_anchor(base_size=7, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """
    This function is to generate base anchors according to the given ratios and scales.
    Returns:
       * **anchor_base**: A tensor of base anchor window each line of which is\
        [ymin,xmin,ymax,xmax]. Remember x represents the axis of width and y the \
        axis of height.
    """
    py=base_size / 2.
    px=base_size / 2.

    anchor_base=np.zeros((len(ratios) * len(anchor_scales), 4),dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return torch.from_numpy(anchor_base)

def enumerate_anchor(anchor_base,feat_ratio,height,width):
    """
    This function returns the sliding anchor window according to the given base anchor\
        window.
    Returns:
       * **sliding anchor windows**: A tensor of all anchor windows.
    """
    shift_y = torch.arange(0, height * feat_ratio, feat_ratio)
    shift_x = torch.arange(0, width * feat_ratio, feat_ratio)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    shift=torch.from_numpy(shift)

    n_fixed_anchor = anchor_base.size()[0]
    n_positions = shift.size()[0]
    anchor = anchor_base.reshape((1, n_fixed_anchor, 4)) + shift.reshape((1, n_positions, 4)).permute((1, 0, 2))
    anchor = anchor.reshape((n_fixed_anchor * n_positions, 4)).type(torch.float32)
    return anchor

class RegionProposalNetwork(nn.Module):
    def __init__(self,
    in_channel=256,mid_channel=256,ratios=[0.5,1,2],anchor_scales=[8,16,32],
    feat_ratio=16,
    ):
        self.anchor=generate_anchor(ratios=ratios,anchor_scales=anchor_scales)
        self.feat_ratio=feat_ratio
        self.n_anchor=self.anchor.size()[0]
        super(RegionProposalNetwork,self).__init__()
        self.conv1=nn.Conv2d(in_channel,mid_channel,kernel_size=3,padding=1)
        self.score=nn.Conv2d(mid_channel,self.n_anchor*2,kernel_size=1,padding=0)
        self.loc=nn.Conv2d(mid_channel,self.n_anchor*4,kernel_size=1,padding=0)
        self.proposal_layer=ProposalCreator()

    
    def forward(self,x,img_size,scale=1.0):
        h=F.relu(self.conv1(x))
        rpn_loc=self.loc(h)
        rpn_score=self.score(h)

        xN,_,xH,xW=x.size()
        anchor=enumerate_anchor(self.anchor,self.feat_ratio,xH,xW)
        rpn_loc=rpn_loc.permute(0,2,3,1).contiguous().view(xN,-1,4)

        rpn_score=rpn_score.permute(0,2,3,1).contiguous()
        rpn_softmax_score=F.softmax(rpn_score.view(xN,xH,xW,self.n_anchor,2), dim=4)
        rpn_fg_score=rpn_softmax_score[:,:,:,:,1].contiguous().view(xN,-1)
        rpn_score=rpn_score.view(xN,-1,2)

        temp_roi=list()
        temp_roi_indices=list()
        for i in range(xN):
            roi=self.proposal_layer(rpn_loc[i].data,rpn_fg_score[i].data,anchor,img_size,scale=scale)
            batch_idx=i*torch.ones(len(roi)).long()
            temp_roi.append(roi)
            temp_roi_indices.append(batch_idx)

        rois=torch.cat(tuple(temp_roi),dim=0)
        roi_indices=torch.cat(tuple(temp_roi_indices),dim=0)

        return rpn_loc,rpn_score,rois,roi_indices,anchor

