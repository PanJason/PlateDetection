import torch
import torch.nn as nn
from torch.nn import functional as F
from model.utils.widget import cal_iou
import numpy as np
def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)  # 计算交并比

    return iou
def gen_out(img_size,feat_ratio,bbox):
    THRES=0.75
    """
    Compute the ground truth of the feature map. Whether or not a plate should be in this grid.
    Returns:
    * **out**: Tensor representing the existence of the plate.
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out=torch.zeros((len(bbox),int(img_size[0]/feat_ratio),int(img_size[1]/feat_ratio))).to(device).long()
    for i in range(len(bbox)):
        ymin,xmin,ymax,xmax = bbox[i].cpu().numpy()
        width=xmax-xmin
        height=ymax-ymin
        for j in  range(int(img_size[0]/feat_ratio)):
            for k in range(int(img_size[1]/feat_ratio)):
                anchor=[j*feat_ratio-height/2,k*feat_ratio-width/2,j*feat_ratio+height/2,k*feat_ratio+width/2]
                anchor=np.array(anchor)
                if compute_iou(anchor,np.array([ymin,xmin,ymax,xmax])) >THRES: #Here the threshold is 0.75
                    out[i,j,k]=1
    return out

def gen_bbox(img_size,feat_ratio,bbox):
    """
    Compute the ground truth of affine parameters with respect to each grid in the feature map.
    Returns:
    * **out**: The ground truth affine parameters.
    """
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out=torch.zeros((len(bbox),4,int(img_size[0]/feat_ratio),int(img_size[1]/feat_ratio))).to(device)
    for i in range(len(bbox)):
        for j in  range(int(img_size[0]/feat_ratio)):
            for k in range(int(img_size[1]/feat_ratio)):
                ymin=bbox[i,0]/feat_ratio-j
                xmin=bbox[i,1]/feat_ratio-k
                ymax=bbox[i,2]/feat_ratio-j
                xmax=bbox[i,3]/feat_ratio-k
                out[i,0,j,k]=ymin
                out[i,1,j,k]=xmin
                out[i,2,j,k]=ymax
                out[i,3,j,k]=xmax
    return out

