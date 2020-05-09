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