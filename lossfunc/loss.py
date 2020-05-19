import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def loc_loss(pred_loc, gt_loc, gt_label, sigma):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_weight=torch.zeros(gt_loc.size()).to(device)
    in_weight[(gt_label>0).view(-1, 1).expand_as(in_weight)]=1
    loss = smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loss /= ((gt_label >= 0).sum().float())
    return loss

def affine_loss(affine,gt):
    """
    This function calculate the loss between predict bbox and the ground truth bbox.
    """
    batch_size,_,xH,xW=affine.size()
    ymin=-0.5*affine[:,0,:,:].unsqueeze(1)-0.5*affine[:,1,:,:].unsqueeze(1)+affine[:,4,:,:].unsqueeze(1)
    xmin=-0.5*affine[:,3,:,:].unsqueeze(1)-0.5*affine[:,2,:,:].unsqueeze(1)+affine[:,5,:,:].unsqueeze(1)
    ymax=0.5*affine[:,0,:,:].unsqueeze(1)+0.5*affine[:,1,:,:].unsqueeze(1)+affine[:,4,:,:].unsqueeze(1)
    xmax=0.5*affine[:,3,:,:].unsqueeze(1)+0.5*affine[:,2,:,:].unsqueeze(1)+affine[:,5,:,:].unsqueeze(1)

    affine_bbox=torch.cat((ymin,xmin,ymax,xmax),dim=1)

    return F.l1_loss(affine_bbox,gt) #here may be some bug without considering the masks.