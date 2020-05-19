import numpy as np
import os
import cv2
import torch 
import torch.nn as nn
import torch 
import torch.optim as optim
import torch.nn.functional as F
import math
from OCRmodel import CNN
train_path='.\\Chars_data'  
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_acc=0
BATCHSIZE=16
EPOCH=30
LEARNING_RATE=0.001
"""
This is a script to train the basic OCR model. Since the given chars_data set is very much different from those on the plate, we need another model trained directly on the ground truth of
the plate characters.
Returns:
    the basic OCR model which performs really bad on the training set.
"""

def data_augmentation(train_img,train_label):
    """
    This function augments the given chars data. Zooming, erasing, scattering are used here.
    Returns:
    * **train_img**: The augmented train image.
    * **train_label**: The augmented train label.
    """
    #zoom_in=np.random.randint(0,train_img.shape[0],size=(1,2000))
    zoom_out=np.random.randint(0,train_img.shape[0],size=(1,4000))
    scatter=np.random.randint(0,train_img.shape[0],size=(1,4000))
    erase=np.random.randint(0,train_img.shape[0],size=(1,4000))
    blur=np.random.randint(0,train_img.shape[0],size=(1,3000))
    sharpen=np.random.randint(0,train_img.shape[0],size=(1,3000))
    for i in scatter[0]:
        train_img = np.insert(train_img, train_img.shape[0], values=train_img[i], axis=0)
        train_label=np.insert(train_label,train_label.shape[0],values=train_label[i],axis=0)
        coord=np.random.randint(0,20,size=(50,2))
        for pos in coord:
            train_img[i,pos[0],pos[1]]=255
    for i in erase[0]:
        train_img = np.insert(train_img, train_img.shape[0], values=train_img[i], axis=0)
        train_label=np.insert(train_label,train_label.shape[0],values=train_label[i],axis=0)
        coord=np.random.randint(0,20,size=(50,2))
        for pos in coord:
            train_img[i,pos[0],pos[1]]=0
    for i in zoom_out[0]:
        train_img = np.insert(train_img, train_img.shape[0], values=train_img[i], axis=0)
        train_label=np.insert(train_label,train_label.shape[0],values=train_label[i],axis=0)
        resize_shape=np.random.randint(15,20)
        train_img_tmp=cv2.resize(train_img[i],(resize_shape,resize_shape))
        #print(train_img_tmp.shape)
        train_img[i]=cv2.copyMakeBorder(train_img_tmp,(20-resize_shape)//2,20-resize_shape-(20-resize_shape)//2,(20-resize_shape)//2,20-resize_shape-(20-resize_shape)//2,cv2.BORDER_CONSTANT,value=0)
        coord=np.random.randint(0,20,size=(25,2))
        for pos in coord:
            train_img[i,pos[0],pos[1]]=255
        coord1=np.random.randint(0,20,size=(25,2))
        for pos in coord1:
            train_img[i,pos[0],pos[1]]=0
    """
    for i in blur:
        train_img = np.insert(train_img, train_img.shape[0], values=train_img[i], axis=0)
        train_label=np.insert(train_label,train_label.shape[0],values=train_label[i],axis=0)
        train_img[i]=cv2.blur(train_img[i],ksize=(3,3))
    for i in sharpen:
        train_img = np.insert(train_img, train_img.shape[0], values=train_img[i], axis=0)
        train_label=np.insert(train_label,train_label.shape[0],values=train_label[i],axis=0)
        kernel1=np.array(
        [[0,-1,0],
        [-1,5,-1],
        [0,-1,0]]
        )
        train_img[i]=cv2.filter2D(train_img[i],-1,kernel1)
    """
    return train_img,train_label


def data_preparation(x,y,train_ratio,shuffle=True):
    """
    This function is to seperate the training dataset from the test dataset.
    Returns:
    * **training_img**
    * **training_labels**
    * **test_img**
    * **test_labels**
    """
    train_num=math.ceil(len(x)*train_ratio)
    test_num=len(x)-train_num
    indexes=list(range(len(x)))
    if shuffle:
        np.random.shuffle(indexes)
    indexes=np.array(indexes)
    train_index=indexes[:train_num]
    test_index=indexes[train_num:len(x)]
    train_index=torch.from_numpy(train_index).long()
    test_index=torch.from_numpy(test_index).long()
    return x[train_index],y[train_index],x[test_index],y[test_index]


def train(epoch):#kc1,kc2):
    avg_loss=0
    for j in range(epoch):
        for i in range(int(len(train_set)/BATCHSIZE)):
            optimizer.zero_grad()
            xProb=net(train_set[i*BATCHSIZE:(i+1)*BATCHSIZE])
            loss=criterion(xProb,train_gt[i*BATCHSIZE:(i+1)*BATCHSIZE])
            avg_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / int(len(train_img)/BATCHSIZE)

        print("Mean Trainning Loss:{:.4f}".format(avg_loss))
        #writer.add_scalar('Train/Loss_NET%d'%(epoch), avg_loss, j)
        #writer.flush()
        validate()

def validate():
    total_correct = 0
    avg_loss = 0.0
    global best_acc
    for i in range(len(test_set)):
        images = test_set[i].unsqueeze(0)
        labels = test_gt[i].unsqueeze(0)
        with torch.no_grad():
            output = net(images)
        avg_loss += criterion(output,labels)
        pred = output.detach().max(1)[1]  # detach cell from the model graph
        total_correct += pred.eq(labels.view_as(pred))

    avg_loss /= len(test_set)
    print('Validation Avg. Loss: %f, Accuracy: %f' % (
    avg_loss.detach().cpu().item(), float(total_correct) / len(test_set)))
    if float(total_correct) / len(test_set) > best_acc:
        best_acc = float(total_correct) / len(test_set)
        torch.save(net.state_dict(), 'best_OCR_model_test.pt')
    return avg_loss.detach().cpu().item(), float(total_correct) / len(test_set)

if __name__ == "__main__":
    #data preparation
    category=len(os.listdir(train_path))
    train_img=list()
    train_label=list()
    for i,dir in enumerate(os.listdir(train_path)):
        subpath=os.path.join(train_path, dir)
        for file in os.listdir(subpath):
            img=cv2.imread(os.path.join(subpath,file))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            train_img.append(img)
            train_label.append(i)
    train_img=np.array(train_img)
    train_label=np.array(train_label)   
    train_img,train_label=data_augmentation(train_img,train_label)
    train_img_t=torch.from_numpy(train_img)
    train_label_t=torch.from_numpy(train_label).long()
    train_img_t=train_img_t.unsqueeze(1).float()
    train_set,train_gt,test_set,test_gt=data_preparation(train_img_t,train_label_t,0.7)

    #Network definition and training 
    net=CNN()
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(net.parameters(),lr=LEARNING_RATE)
    train_set=train_set.to(device)
    train_gt=train_gt.to(device)
    test_set=test_set.to(device)
    test_gt=test_gt.to(device)
    net=net.to(device)
    train(EPOCH)
    