import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import os
from sklearn.cluster import KMeans
from OCRmodel import CNN, CNN_adv2
import math
import numpy as np
import torch
from trainOCRmodel import data_preparation
import torch.nn as nn
import torch.optim as optim


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
f_img = '.\\Plate_dataset\\AC\\train\\jpeg'
f_xml = '.\\Plate_dataset\\AC\\train\\xml'
BATCHSIZE=8
EPOCH=10
LEARNING_RATE=0.001
plates=[]
plates_gt=[]
train_path='.\\Chars_data'
char_map=dict()
char_map2=dict()
true=0.
all_plate = 0.
all_char = 0.
right_char = 0.
wrong_index = []
wrong_ones = []
right_five = 0.0
addition_train = []
addition_label = []

def find_waves(threshold, histogram):
    """
    This function is used to find peaks of the histogram in horizontal axis.
    Returns:
    * **wave_peaks**: each list in the wave_peak is in the format of [start_wave, end_wave]
    """
    up_point = -1
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks

def contain(bbox1,bbox2):
    """
    This function is to judge whether one box was in another.
    Returns:
    * **bool**: 1 if a bbox is in another.
    """
    x1,y1,w1,h1=bbox1
    x2,y2,w2,h2=bbox2
    if x1==x2 and y1==y2 and w1==w2 and h1==h2:
        return 0
    if x1<=x2 and y1<=y2 and x1+w1>=x2+w2 and y1+h1>=y2+h2:
        return 1
    if x1>=x2 and y1>=y2 and x1+w1<=x2+w2 and y1+h1<=y2+h2:
        return 1
    return 0


def valid_bbox(bboxes,thres=0.6,thres_up=1.5):
    """
    If there are bboxes whose height is less than 60% of the plate height, these bounding boxes are considered as invalid.
    If there are bboxes inside other bboexes, there are invalid as well.
    Returns:
    * **bool**: 1 if bbox is valid.
    """
    heightMax=np.max(bboxes[:,3])
    for b in bboxes:
        if b[3]<thres*heightMax or b[3]>thres_up*heightMax:
            return 0
    for b1 in bboxes:
        for b2 in bboxes:
            if contain(b1,b2):
                return 0
    return 1


def remove_plate_upanddown_border(plate_Arr):
    """
    This function is used to cut off the useless part of the plate and return a binary plate pic.
    Returns:
    * **plate_binary_img**: The Two value form of plate pic.
    * **plate_Arr**: The plate reshaped and croped in BGR format.
    """
    hh,ww,_=plate_Arr.shape
    plate_gray_Arr = cv2.cvtColor(plate_Arr[4:hh-4,3:ww-3], cv2.COLOR_BGR2GRAY)
    plate_gray_Arr = cv2.normalize(plate_gray_Arr, 0, 255, cv2.NORM_MINMAX)
    #plate_gray_Arr = cv2.medianBlur(plate_gray_Arr,5)
    #kernel=np.ones((3,3),np.float32)/9
    #plate_gray_Arr=cv2.filter2D(plate_gray_Arr,-1,kernel)
    #plate_gray_Arr=cv2.GaussianBlur(plate_gray_Arr,(3,3),2)
    #gray_lap = cv2.Laplacian(plate_gray_Arr,cv2.CV_16S,ksize = 3)
    #plate_gray_Arr = cv2.convertScaleAbs(gray_lap)
    ret, plate_binary_img = cv2.threshold( plate_gray_Arr, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )
    row_histogram = np.sum(plate_binary_img, axis=1)

    row_min = np.min( row_histogram )
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 3.2
    wave_peaks = find_waves(row_threshold, row_histogram)
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1]-wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    if wave_peaks==[]:
        ret, plate_binary_img = cv2.threshold( plate_gray_Arr, 78, 255, cv2.THRESH_BINARY_INV)
        row_histogram = np.sum(plate_binary_img, axis=1)

        row_min = np.min( row_histogram )
        row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
        row_threshold = (row_min + row_average) / 3
        wave_peaks = find_waves(row_threshold, row_histogram)
        wave_span = 0.0
        for wave_peak in wave_peaks:
            span = wave_peak[1]-wave_peak[0]
            if span > wave_span:
                wave_span = span
                selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    #this kernel sharpens the binary picture for better segmentation
    kernel1=np.array(
        [[0,-2,0],
        [-2,9,-2],
        [0,-2,0]]
    )
    plate_binary_img=cv2.filter2D(plate_binary_img,-1,kernel1)
    return  plate_binary_img,plate_Arr[selected_wave[0]:selected_wave[1],:]

def plate_number_bbox(plate_binary_img,method='cv2'):
    """
    This function returns six bounding boxes of characters on the binary img.
    Different methods including cv2.findcoutours, clusters and analysis of peaks are considered here.
    Returns:
    * **char_bbox**: The potential bbox of each character on the plate. Each one is in the format of [x,y,w,h].
    """

    if method=='cv2':
        contours, hierarchy = cv2.findContours(plate_binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        potential_bbox=[]
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w>=2.4*h:
                potential_bbox.append([x,y,w//4,h])
                potential_bbox.append([x+w//4,y,w-3*w//4,h])
                potential_bbox.append([x+2*w//4,y,w-2*w//4,h])
                potential_bbox.append([x+3*w//4,y,w-w//4,h])
                continue
            if w<2.4*h and w>=1.9*h:
                potential_bbox.append([x,y,w//3,h])
                potential_bbox.append([x+w//3,y,w-2*w//3,h])
                potential_bbox.append([x+2*w//3,y,w-w//3,h])
                potential_bbox.append([x+3*w//4,y,w-w//4,h])
                continue
            if w<2*h and w>h:
                potential_bbox.append([x,y,w//2,h])
                potential_bbox.append([x+w//2,y,w-w//2,h])
            else:
                if w<3:
                    continue
                potential_bbox.append([x,y,w,h])
        potential_bbox=np.array(potential_bbox)
        potential_bbox=potential_bbox[np.argsort(-potential_bbox[:,3])]
        potential_bbox=potential_bbox[:6,:]
        if valid_bbox(potential_bbox):
            return potential_bbox
        else:
            return plate_number_bbox(plate_binary_img,method="kmeans")
    if method=="kmeans":
        hh,ww=plate_binary_img.shape
        w_int=ww//6
        init_centroid=[]
        init_centroid.append([w_int/2,hh/2])
        init_centroid.append([w_int*3/2,hh/2])
        init_centroid.append([w_int*5/2+1,hh/2])
        init_centroid.append([w_int*7/2+2,hh/2])
        init_centroid.append([w_int*9/2+4,hh/2])
        init_centroid.append([w_int*11/2+5,hh/2])
        row_list,col_list = np.nonzero (  plate_binary_img >= 255 )
        dataArr = np.column_stack(( col_list,row_list))
        model=KMeans(n_clusters=6,init=np.array(init_centroid))
        y_pred=model.fit_predict(dataArr)
        d=dict()
        for i in range(6):
            d[i]=[]
        for i,label in enumerate(y_pred):
            d[label].append(list(dataArr[i]))
        #print(d)
        for i in range(6):
            d[i]=np.array(d[i])
        potential_bbox=[]
        for i in range(6):
            x,y,w,h = cv2.boundingRect(d[i])
            potential_bbox.append([x,y,w,h])
        potential_bbox=np.array(potential_bbox)
        if valid_bbox(potential_bbox,thres=0.75):
            return potential_bbox
        else:
            potential_bbox=[]
            hh,ww=plate_binary_img.shape
            w_int=ww//6
            potential_bbox.append([0,0,w_int,hh])
            potential_bbox.append([w_int,0,w_int,hh])
            potential_bbox.append([w_int*2,0,w_int+1,hh])
            potential_bbox.append([w_int*3+1,0,w_int+1,hh])
            potential_bbox.append([w_int*4+2,0,w_int+1,hh])
            potential_bbox.append([w_int*5+3,0,ww-(w_int*5+3),hh])
            return np.array(potential_bbox)

def pad_binary_char(plate_binary_img,char_bbox):
    """
    Returns a tensor in shape of 6*1*20*20 represents 6 bounded characters.
    Returns:
    * **char_tensor**: A tensor represent the character. Each one is in the shape of 6*1*20*20.
    """
    char_tensor=torch.zeros((6,1,20,20))
    char_bbox=char_bbox[np.argsort(char_bbox[:,0])]
    for i,bbox in enumerate(char_bbox):
        x,y,w,h=bbox
        character=cv2.resize(plate_binary_img[y:y+h,x:x+w],(math.ceil(w/h*18),18))
        if character.shape[1]>20:
            character=cv2.resize(character,(20,18))
        left_padding=(20-character.shape[1])//2
        right_padding=20-character.shape[1]-left_padding
        if left_padding>0:
            #print(character.shape)
            character=np.insert(character,0,values=np.zeros((left_padding,18)),axis=1)
        if right_padding>0:
            character=np.insert(character,character.shape[1],values=np.zeros((right_padding,18)),axis=1)
        character=np.insert(character,0,values=np.zeros((1,20)),axis=0)
        character=np.insert(character,character.shape[0],values=np.zeros((1,20)),axis=0)
        #fig=plt.figure()
        #plt.imshow(character)
        assert character.shape[1]==20
        char_tensor[i,0]=torch.from_numpy(character)
    return char_tensor


def predict_label(index):
    """
    This function returns the label predicted using heuristics.
    Returns:
    * **s**: A string representing the predicted license plate.
    """
    global char_map
    s=""
    for i in index:
        s+=char_map[int(i)]
    if s[2]=='G':
        s=s[0:2]+'6'+s[3:]
    if s[2]=='D':
        s=s[0:2]+'0'+s[3:]
    if s[2]=='B':
        s=s[0:2]+'8'+s[3:]
    """
    if s[2]=='A':
        s=s[0:2]+'4'+s[3:]
    if s[2]=='Z':
        s=s[0:2]+'7'+s[3:]
    if s[2]=='Q':
        s=s[0:2]+'0'+s[3:]
    """
    if s[3]=='G':
        s=s[0:3]+'6'+s[4:]
    if s[3]=='D':
        s=s[0:3]+'0'+s[4:]
    if s[3]=='B':
        s=s[0:3]+'8'+s[4:]
    """
    if s[3]=='A':
        s=s[0:3]+'4'+s[4:]
    if s[3]=='Z':
        s=s[0:3]+'7'+s[4:]
    if s[3]=='Q':
        s=s[0:3]+'0'+s[4:]
    """
    if s[0]>='A' and s[0]<='Z' and s[1]>='A' and s[1]<='Z':
        if s[4]=='G':
            s=s[0:4]+'6'+s[5:]
        if s[4]=='D':
            s=s[0:4]+'0'+s[5:]
        if s[4]=='B':
            s=s[0:4]+'8'+s[5:]
        if s[5]=='G':
            s=s[0:5]+'6'
        if s[5]=='D':
            s=s[0:5]+'0'
        if s[5]=='B':
            s=s[0:5]+'8'
    if s[4]>='A' and s[4]<='Z' and s[5]>='A' and s[5]<='Z':
        if s[0]=='G':
            s='6'+s[1:]
        if s[0]=='D':
            s='0'+s[1:]
        if s[0]=='B':
            s='8'+s[1:]
        if s[1]=='G':
            s=s[0:1]+'6'+s[2:]
        if s[1]=='D':
            s=s[0:1]+'0'+s[2:]
        if s[1]=='B':
            s=s[0:1]+'8'+s[2:]
    return s


best_acc=0
def train(epoch):#kc1,kc2):
    avg_loss=0
    for j in range(epoch):
        for i in range(int(len(train_set)/BATCHSIZE)):
            optimizer.zero_grad()
            xProb=net1(train_set[i*BATCHSIZE:(i+1)*BATCHSIZE])
            loss=criterion(xProb,train_gt[i*BATCHSIZE:(i+1)*BATCHSIZE])
            avg_loss += loss.detach().cpu().item()
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / int(len(train_set)/BATCHSIZE)

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
            output = net1(images)
        avg_loss += criterion(output,labels)
        pred = output.detach().max(1)[1]  # detach cell from the model graph
        total_correct += pred.eq(labels.view_as(pred))

    avg_loss /= len(test_set)
    print('Validation Avg. Loss: %f, Accuracy: %f' % (
    avg_loss.detach().cpu().item(), float(total_correct) / len(test_set)))
    if float(total_correct) / len(test_set) > best_acc:
        best_acc = float(total_correct) / len(test_set)
        torch.save(net1.state_dict(), 'best_OCR_model_CNN_net_adv2_test.pt')
    return avg_loss.detach().cpu().item(), float(total_correct) / len(test_set)



if __name__ == "__main__":

    #Load training data
    for file in os.listdir(f_img):

        img = cv2.imread(f_img+'\\'+file)
        imgID=file.split('.')[0]
        anno = ET.ElementTree(file=f_xml+'\\'+imgID+'.xml')
        label = anno.find('object').find('platetext').text
        xmin = anno.find('object').find('bndbox').find('xmin').text
        ymin = anno.find('object').find('bndbox').find('ymin').text
        xmax  = anno.find('object').find('bndbox').find('xmax').text
        ymax = anno.find('object').find('bndbox').find('ymax').text
        bbox = [xmin,ymin,xmax,ymax]
        bbox = [int(b)  for b in bbox]
        plate_img=img[int(ymin):int(ymax),int(xmin):int(xmax),:]
        plates.append(plate_img)
        plates_gt.append(label)


    #Create the map between character and the its indices.
    for i,dir in enumerate(os.listdir(train_path)):
        char_map[i]=dir
        char_map2[dir]=i

    
    #Load the pretrained OCR model.
    net=CNN_adv2()
    net.load_state_dict(torch.load('best_OCR_model_CNN_net_adv2_1.pt'))

    #Predict the label of the training plates
    for i, plate in enumerate(plates):
        plate_binary_img, plate_Arr = remove_plate_upanddown_border(plate)
        char_bbox = plate_number_bbox(plate_binary_img)
        try:
            char_tensor = pad_binary_char(plate_binary_img, char_bbox)
        except Exception:
            print(char_bbox)
            continue
        out1 = net(char_tensor)
        out2 = net(char_tensor)
        out3 = net(char_tensor)
        out4 = net(char_tensor)
        out5 = net(char_tensor)
        out6 = net(char_tensor)
        out = out1 + out2 + out3 + out4 + out5 + out6
        _, index = torch.max(out, 1)
        label = predict_label(index)
    #print(plates_gt[i],label)
        t = 1
        for j, s in enumerate(plates_gt[i]):
            right_char += (s == label[j])
            t -= (s != label[j])
            if s == label[j] or (s=="R" and (label[j]=="8" or label[j]=="B" or label[j]=="H") or (s=="B" and (label[j]=="8"))): #Here I mannually add some training data from the wrong judged.
                addition_train.append(char_tensor[j].numpy())
                addition_label.append(char_map2[s])
        all_char += len(label)
        true += (plates_gt[i] == label)
        if plates_gt[i] != label:
            wrong_index.append(i)
        if t >= 0:
            right_five += 1
        if t == 0:
            wrong_ones.append(i)
        all_plate += 1


    #Create additional training dataset
    addition_train1=torch.from_numpy(np.array(addition_train)).float()
    addition_label=torch.tensor(addition_label).long()
    train_set,train_gt,test_set,test_gt=data_preparation(addition_train1,addition_label,0.8)


    #Generate another classification model
    net1=CNN_adv2()
    criterion=nn.NLLLoss()
    optimizer=optim.Adam(net1.parameters(),lr=LEARNING_RATE)

    #Train the model.
    train_set=train_set.to(device)
    train_gt=train_gt.to(device)
    test_set=test_set.to(device)
    test_gt=test_gt.to(device)
    net1=net1.to(device)
    train(EPOCH)

