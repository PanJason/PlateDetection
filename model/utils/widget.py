import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def cal_iou(anchor,bbox,inside_indices):
    """
    Calculate the ious between a set of anchors and a bounding box
    Returns:
       * **ious**: A Tensor representing the iou between each anchor and the bounding box
    """
    # top left
    tl=torch.max(anchor[:, None, :2], bbox[:, :2])
    # bottom right
    br=torch.min(anchor[:, None, 2:], bbox[:, 2:])
    area_i=torch.prod(br-tl, axis=2)*(tl<br).all(axis=2)
    area_a=torch.prod(anchor[:, 2:]-anchor[:, :2], axis=1)
    area_b=torch.prod(bbox[:, 2:]-bbox[:, :2], axis=1)
    return area_i/(area_a[:, None]+area_b-area_i)


def nms(roi,score,overlap=0.7,n_post_nms=50):
    """
    Given the roi and its corresponding scores, the non maximum suppress select the bbox
    if the IoUs between the bbox and the previous selected bboxes are less than overlap.
    Note that the length of rois and score must equal.
    Returns:
       * **Indices**: The indices of the selected bboxes in the original rois.  
    """
    keep=score.new(score.size(0)).zero_().long() 

    if roi.numel() == 0: 
        return keep
        
    y1=roi[:, 0]
    x1=roi[:, 1]
    y2=roi[:, 2]
    x2=roi[:, 3]
    area=(y2-y1)*(x2-x1)

    v,idx=score.sort(0)


    count=0
    while idx.numel() > 0:
        i=idx[-1]
        keep[count]=i
        count+=1
        if idx.size(0) == 1: 
            break
        idx=idx[:-1] 
        

        xx1=torch.index_select(x1, 0, idx)
        yy1=torch.index_select(y1, 0, idx)
        xx2=torch.index_select(x2, 0, idx)
        yy2=torch.index_select(y2, 0, idx)
        
        xx1=torch.clamp(xx1, min=float(x1[i]))
        yy1=torch.clamp(yy1, min=float(y1[i]))
        xx2=torch.clamp(xx2, max=float(x2[i]))
        yy2=torch.clamp(yy2, max=float(y2[i]))

        
        ww=xx2-xx1
        hh=yy2-yy1 
        ww=torch.clamp(ww, min=0.0)
        hh=torch.clamp(hh, min=0.0)
        inter = ww*hh
        
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas + area[i]- inter
        IoU = inter/union
        

        idx = idx[IoU.le(overlap)]
    return keep[:n_post_nms]


def l2b(src_bbox,loc):
    """
    Given the primitive bounding box [ymin,xmin,ymax,xmax] from src_bbox and transform\
    parameters [dy,dx,dh,dw] from loc. We can calculate the modified bounding box. Note\
    the lenght of src_bbox and loc should be matched.
    Returns:
       * **mod_bbox**: A Tensor each line of which is a modifed bounding box [ymin',xmin',
       ymax',xmax']
    """

    src_h=src_bbox[:,2]-src_bbox[:,0]
    src_w=src_bbox[:,3]-src_bbox[:,1]
    src_cy=(src_bbox[:,0]+src_bbox[:,2])/2
    src_cx=(src_bbox[:,1]+src_bbox[:,3])/2

    dy=loc[:,0]
    dx=loc[:,1]
    dh=loc[:,2]
    dw=loc[:,3]

    mod_cy=src_cy+dy*src_h
    mod_cx=src_cx+dx*src_w
    mod_h=src_h*torch.exp(dh)
    mod_w=src_w*torch.exp(dw)

    mod_ymin=mod_cy-0.5*mod_h
    mod_xmin=mod_cx-0.5*mod_w
    mod_ymax=mod_cy+0.5*mod_h
    mod_xmax=mod_cy+0.5*mod_w

    mod_bbox=torch.cat((mod_ymin.unsqueeze(1),mod_xmin.unsqueeze(1),mod_ymax.unsqueeze(1),
    mod_xmax.unsqueeze(1)),dim=1)


    return mod_bbox

def b2l(src_bbox,mod_bbox):
    """
    Given the primitive bounding box and the modified bounding box, we can calculate the
    parameters of transformation.
    Returns:
       * **loc**: A Tensor each line of which is a set of transform parameters [dy,dx,dh,dw]
    """
    src_h=src_bbox[:,2]-src_bbox[:,0]
    src_w=src_bbox[:,3]-src_bbox[:,1]
    src_cy=(src_bbox[:,0]+src_bbox[:,2])/2
    src_cx=(src_bbox[:,1]+src_bbox[:,3])/2

    mod_h=mod_bbox[:,2]-mod_bbox[:,0]
    mod_w=mod_bbox[:,3]-mod_bbox[:,1]
    mod_cy=(mod_bbox[:,0]+mod_bbox[:,2])/2
    mod_cx=(mod_bbox[:,1]+mod_bbox[:,3])/2

    dy=(mod_cy-src_cy)/src_h
    dx=(mod_cx-src_cx)/src_w
    dh=torch.log(mod_h / src_h)
    dw=torch.log(mod_w / src_w)
    loc=torch.cat((dy.unsqueeze(1),dx.unsqueeze(1),dh.unsqueeze(1),dw.unsqueeze(1)),dim=1)

    return loc