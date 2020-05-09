from PlateDetection.data import dataset
from PlateDetection.model import RegionProposalNetwork as RPN
from PlateDetection.model.compactVGG16 import compactVGG16
from PlateDetection.model.RCNNHeader import RCNNHeader
from PlateDetection.model.RCNNHeader import ProposalTargetCreator
from PlateDetection.model.utils import widget
from PlateDetection.model.fasterRCNN_frame import FasterRCNN
import torch
import torch.nn.functional as F
from PlateDetection.model.creator import AnchorTargetGenerator
from PlateDetection.lossfunc.loss import loc_loss
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

PATH='./log'
writer=SummaryWriter(PATH)

train_img,train_platetext,train_bbox,test_img,test_platetext,test_bbox=dataset.preprocess()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_bbox_T=torch.zeros((len(train_bbox),4)).to(device)
train_bbox_T[:,0]=train_bbox[:,1]
train_bbox_T[:,2]=train_bbox[:,3]
train_bbox_T[:,1]=train_bbox[:,0]
train_bbox_T[:,3]=train_bbox[:,2]
net=compactVGG16()
rpn=RPN.RegionProposalNetwork()
head=RCNNHeader(5,1./16)
net.to(device)
rpn.to(device)
head.to(device)
fasterRCNN=FasterRCNN(net,rpn,head)
optimizer=optim.Adam(fasterRCNN.parameters(),lr=0.01,weight_decay=0.0005)
ptc=ProposalTargetCreator()
atc=AnchorTargetGenerator()


def train(epoch):#kc1,kc2):
    avg_loss = 0
    for j in range(epoch):
        for i in range(len(train_img)):
            optimizer.zero_grad()
            xFeatured=fasterRCNN.extractor(train_img[i].unsqueeze(0))

            rpn_locs,rpn_scores,rois,roi_indices,anchor=fasterRCNN.rpn(xFeatured,(240,352))

            trueLoc,trueLabel=atc(anchor,train_bbox_T[i].unsqueeze(0),(240,352))
            sample_roi,sample_roi_indices,gt_loc,gt_label=ptc(rois,roi_indices,train_bbox_T[i].unsqueeze(0))

            try:
                roi_cls_locs,roi_scores=fasterRCNN.head(xFeatured,sample_roi,sample_roi_indices)
            except Exception as e:
                print(rois.size(),roi_indices.size())
            n_sample=roi_cls_locs.size()[0]
            roi_locs=roi_cls_locs.view(-1,2,4)
            roi_locs=roi_locs[torch.arange(0,n_sample),gt_label]

            loss0=loss=F.cross_entropy(rpn_scores[0], trueLabel, ignore_index=-1)
            loss1=F.cross_entropy(roi_scores,gt_label)
            loss2=loc_loss(rpn_locs[0],trueLoc,trueLabel.data,3)
            loss3=loc_loss(roi_locs[0],gt_loc,gt_label.data,1)

            loss=loss0+loss1+loss2+loss3

            avg_loss += loss.detach().cpu().item()
    
            loss.backward()
            optimizer.step()

        # 计算1个Epoch的平均 Training Loss
        avg_loss = avg_loss / len(train_img)

        print("Mean Trainning Loss:{:.4f}".format(avg_loss))
        writer.add_scalar('Train/Loss%d'%(epoch), avg_loss, j)
        writer.flush()
        #writer.add_scalar('Val/Loss%d'%(epoch),val_loss, j)
        #writer.add_scalar('Val/Accuracy%d'%(epoch), accuracy, j)
        #writer.flush()