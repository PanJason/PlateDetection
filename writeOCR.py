from xml.dom.minidom import Document
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import os
from sklearn.cluster import KMeans
from runOCR import *
from OCRmodel import CNN_adv2

f_img = '.\\Plate_dataset\\AC\\test\\jpeg'
f_xml = '.\\Plate_dataset\\AC\\test\\xml'
xml_pred = './Plate_dataset/AC/test/xml_pred'
train_path='.\\Chars_data'
char_map=dict()
char_map2=dict()
plates=[]
plates_gt=[]
plate_indx=[]
true=0.
all_plate=0.
all_char=0.
right_char=0.
wrong_index=[]
wrong_ones=[]
right_five=0.0
addition_train=[]
addition_label=[]
plates_pred=[]
def predict_label(index):
    """
    This function returns the label predicted using heuristics.
    """
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


if __name__ == "__main__":
    for i,dir in enumerate(os.listdir(train_path)):
        char_map[i]=dir
        char_map2[dir]=i
    
    #print(char_map)
    #First generate ground truth
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
        plate_indx.append(imgID)

    #Then import the trained model
    net=CNN_adv2()
    net.load_state_dict(torch.load('best_OCR_model_CNN_net_adv2_2.pt'))

    #Using the trained model to predict the output
    for i,plate in enumerate(plates):
        plate_binary_img,plate_Arr=remove_plate_upanddown_border(plate)
        char_bbox=plate_number_bbox(plate_binary_img)
        try:
            char_tensor=pad_binary_char(plate_binary_img,char_bbox)
        except Exception:
            print(char_bbox)
            continue
        out1=net(char_tensor)
        out2=net(char_tensor)
        out3=net(char_tensor)
        out4=net(char_tensor)
        out5=net(char_tensor)
        #out6=net(char_tensor)
        out=out1+out2+out3+out4+out5#+out6
        _,index=torch.max(out,1)
        label=predict_label(index)
        plates_pred.append(label)
        #print(plates_gt[i],label)
        t=1
        for j,s in enumerate(plates_gt[i]):
            right_char+=(s==label[j])
            t-=(s!=label[j])
            if s==label[j] or (s=="R" and (label[j]=="8" or label[j]=="B" or label[j]=="H") or (s=="B" and (label[j]=="8"))):
                addition_train.append(char_tensor[j].numpy())
                addition_label.append(char_map2[s])
        all_char+=len(label)
        true+=(plates_gt[i]==label)
        if plates_gt[i]!=label:
            wrong_index.append(i)
        if t>=0:
            right_five+=1
        if t==0:
            wrong_ones.append(i)
        all_plate+=1

    #Write to the xml file
    for i,plate in enumerate(plates_pred):
        doc = Document()
        anno=doc.createElement("annotation")
        doc.appendChild(anno)
        obj = doc.createElement("object")
        anno.appendChild(obj)
        platetext = doc.createElement("platetext")
        obj.appendChild(platetext)
        text = doc.createTextNode(plate)
        platetext.appendChild(text)
        filename = xml_pred+'/'+plate_indx[i]+'.xml'
        f = open(filename, "w")
        f.write(doc.toprettyxml(indent="  "))
        f.close()