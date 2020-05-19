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
PATH='./log'
BATCHSIZE=16
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0005
EPOCH=150


def train(epoch):
    """
    This is the function to train the new PlateDetector Model
    """
    mean_ious=0
    avg_loss=0
    for j in range(epoch):
        for i in range(int(len(train_img)/BATCHSIZE)):
            optimizer.zero_grad()
            xProb,xAffine=PD(train_img[i*BATCHSIZE:(i+1)*BATCHSIZE])
            loss1=l1(xProb,train_label[i*BATCHSIZE:(i+1)*BATCHSIZE])
            loss2=affine_loss(xAffine,train_affine[i*BATCHSIZE:(i+1)*BATCHSIZE])
            loss=loss1+loss2
            print("loss1:%f"%(loss1.detach().cpu().item()))
            print("loss2:%f"%(loss2.detach().cpu().item()))
            print("max_prob:%f"%(torch.max(torch.exp(xProb)[:,1,:,:])))
            avg_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / int(len(train_img)/BATCHSIZE)
        print("Epoch:{}".format(j))
        print("Mean Trainning Loss:{:.4f}".format(avg_loss))
        writer.add_scalar('Train/Loss_NET%d'%(epoch), avg_loss, j)
        writer.flush()
        ious=validate()
        if ious>mean_ious:
            mean_ious=ious
            torch.save(PD.state_dict(), './models/test_model.pt')
            print("Best IoU:%f"%(mean_ious))

def validate():
    """
    This is the function to validate the trained model on the test dataset.
    """
    ious = []
    for i in range(len(test_img)):
        xProb,xAffine=PD(test_img[i].unsqueeze(0))
        a,b,c,d=torch.where(xProb==torch.max(xProb[:,1,:,:]))
        affine=xAffine[0,:,c[0],d[0]]
        ymin=16*float((-0.5*affine[0]-0.5*affine[1]+affine[4])+c[0])
        xmin=16*float((-0.5*affine[3]-0.5*affine[2]+affine[5])+d[0])
        ymax=16*float((0.5*affine[0]+0.5*affine[1]+affine[4])+c[0])
        xmax=16*float((0.5*affine[3]+0.5*affine[2]+affine[5])+d[0])
        bbox_pred=torch.tensor([ymin,xmin,ymax,xmax])
        iou = compute_iou(bbox_pred, test_bbox_T[i].cpu(), wh=False)
        ious.append(iou)
    print('所有样本的平均iou:{}'.format(np.mean(ious)))
    print('检测正确的样本数目:{}'.format(len([iou for iou in ious if iou > 0.5])))
    return np.mean(ious)

if __name__ == "__main__":
    writer=SummaryWriter(PATH)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_img,train_platetext,train_bbox,test_img,test_platetext,test_bbox=dataset.preprocess()
    train_bbox_T=torch.zeros((len(train_bbox),4)).to(device)
    train_bbox_T[:,0]=train_bbox[:,1]
    train_bbox_T[:,2]=train_bbox[:,3]
    train_bbox_T[:,1]=train_bbox[:,0]
    train_bbox_T[:,3]=train_bbox[:,2]
    test_bbox_T=torch.zeros((len(test_bbox),4)).to(device)
    test_bbox_T[:,0]=test_bbox[:,1]
    test_bbox_T[:,2]=test_bbox[:,3]
    test_bbox_T[:,1]=test_bbox[:,0]
    test_bbox_T[:,3]=test_bbox[:,2]

    train_label=gen_out((240,352),16,train_bbox_T)
    train_affine=gen_bbox((240,352),16,train_bbox_T)
    test_label=gen_out((240,352),16,test_bbox_T)
    test_affine=gen_bbox((240,352),16,test_bbox_T)

    PD=PlateDetector()
    PD.to(device)
    optimizer=optim.Adam(PD.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    
    l1=nn.NLLLoss()

    train(EPOCH)    