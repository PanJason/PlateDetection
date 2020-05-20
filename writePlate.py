from data import dataset
from model import RegionProposalNetwork as RPN
from model.compactVGG16 import compactVGG16
from model.RCNNHeader import RCNNHeader
from model.RCNNHeader import ProposalTargetCreator
from model.utils import widget
from model.fasterRCNN_frame import FasterRCNN
import torch
import torch.nn.functional as F
from model.creator import AnchorTargetGenerator
from lossfunc.loss import loc_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from model.newModel import PlateDetector
from data.gen_out import gen_out,gen_bbox
from lossfunc.loss import affine_loss
import torch.nn as nn
import numpy as np
from data.gen_out import compute_iou
from xml.dom.minidom import Document
from xml.dom.minidom import parse
import os
import math


xml_pred = './Plate_dataset/AC/test/xml_pred'
PATH='./log'
f_img = '.\\Plate_dataset\\AC\\test\\jpeg'
writer=SummaryWriter(PATH)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plate_indx=[]
def WritePlate():
    """
    This function generate the predicted bbox using the platedetection network.
    Returns:
    * **bbox**: the predicted bbox of each plate.
    """
    bbox=[]
    for i in range(len(test_img)):
        xProb,xAffine=PD(test_img[i].unsqueeze(0))
        a,b,c,d=torch.where(xProb==torch.max(xProb[:,1,:,:]))
        affine=xAffine[0,:,c[0],d[0]]
        ymin=16*float((-0.5*affine[0]-0.5*affine[1]+affine[4])+c[0])
        xmin=16*float((-0.5*affine[3]-0.5*affine[2]+affine[5])+d[0])
        ymax=16*float((0.5*affine[0]+0.5*affine[1]+affine[4])+c[0])
        xmax=16*float((0.5*affine[3]+0.5*affine[2]+affine[5])+d[0])
        bbox.append([xmin,ymin,xmax,ymax])
    return bbox

if __name__ == "__main__":

    #Prepare the test data
    train_img,train_platetext,train_bbox,test_img,test_platetext,test_bbox=dataset.preprocess()

    #Import the trained model
    PD=PlateDetector()
    PD.to(device)
    PD.load_state_dict(torch.load('models/best_model.pt'))

    #Generate output file ID
    for file in os.listdir(f_img):
        imgID=file.split('.')[0]
        plate_indx.append(imgID)

    bbox=WritePlate()

    #Write to the xml file
    for i,plate in enumerate(bbox):
        anno_gt = parse(xml_pred+'/'+plate_indx[i]+'.xml')
        obj=anno_gt.getElementsByTagName("object")
        obj=obj[0]
        bbox= anno_gt.createElement("bndbox")
        obj.appendChild(bbox)
        xmin = anno_gt.createElement("xmin")
        ymin = anno_gt.createElement("ymin")
        xmax = anno_gt.createElement("xmax")
        ymax = anno_gt.createElement("ymax")
        bbox.appendChild(xmin)
        bbox.appendChild(ymin)
        bbox.appendChild(xmax)
        bbox.appendChild(ymax)
        xMin = anno_gt.createTextNode(str(math.ceil(plate[0])))
        yMin = anno_gt.createTextNode(str(math.ceil(plate[1])))
        xMax = anno_gt.createTextNode(str(math.ceil(plate[2])))
        yMax = anno_gt.createTextNode(str(math.ceil(plate[3])))
        xmin.appendChild(xMin)
        ymin.appendChild(yMin)
        xmax.appendChild(xMax)
        ymax.appendChild(yMax)
        filename = xml_pred+'/'+plate_indx[i]+'.xml'
        f = open(filename, "w")
        f.write(anno_gt.toprettyxml(indent="  "))
        f.close()