import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2
import torch 

IM_WIDTH=352
IM_HEIGHT=240


def parse_anno(anno):
    """
    Returns:
       * **label**: Plate text.
       * **xmin**: xmin of bbox.
       * **ymin**: ymin of bbox.
       * **xmax**: xmax of bbox.
       * **ymax**: ymax of bbox.
    """

    return anno.find('object').find('platetext').text, \
    int(anno.find('object').find('bndbox').find('xmin').text), \
    int(anno.find('object').find('bndbox').find('ymin').text), \
    int(anno.find('object').find('bndbox').find('xmax').text), \
    int(anno.find('object').find('bndbox').find('ymax').text)


def img_padding(img):
    """
    This function pads images to the same size for bacth processing.
    Returns:
       * **img**: Images padded by 0s to the positive direction of x axis and y axis. \
       Its shape is :math:`IM_HEIGHT \\times IM_WIDTH \\times Channels`.
    """
    img=np.pad(img,((0,IM_HEIGHT-img.shape[0]),(0,IM_WIDTH-img.shape[1]),(0,0)),'constant')

    return img


def preprocess():
    """
    This function is to load and process data from given data set. Images are loaded\
    and transformed into Tensors. XMLs are parsed and transformed into plate captions\
    and plate bounding boxes seperately.
    Returns:
       * **train_img**: Tensors in shape of :math:`train_set_size \\times channels \\times height \\times width`
       * **train_platetext**: Array of plate texts.
       * **train_bbox**:  Tensors represent the bounding box of the plate. Its shape is\
       :math:`(R`,4)`. Each line's format is :math:`[xmin,ymin,xmax,ymax]`
       * **test_img**: Tensors in shape of :math:`test_set_size \\times channels \\times height \\times width`
       * **test_platetext**: Array of plate texts.
       * **test_bbox**:  Tensors represent the bounding box of the plate. Its shape is\
       :math:`(R`,4)`. Each line's format is :math:`[xmin,ymin,xmax,ymax]` where x is in\
           the width axis and y is in the height axis.
    """
    train_img_path='.\\Plate_dataset\\AC\\train\\jpeg'
    train_xml_path='.\\Plate_dataset\\AC\\train\\xml'
    test_img_path='.\\Plate_dataset\\AC\\test\\jpeg'
    test_xml_path='.\\Plate_dataset\\AC\\test\\xml'
    
    train_set_size=len(os.listdir(train_img_path))
    test_set_size=len(os.listdir(test_img_path))
    
    train_img=list()
    train_platetext=list()
    train_bbox=list()
    test_img=list()
    test_platetext=list()
    test_bbox=list()

    for file in os.listdir(train_xml_path):
        anno = ET.ElementTree(file=os.path.join(train_xml_path, file))
        label,xmin,ymin,xmax,ymax=parse_anno(anno)
        bbox=[xmin,ymin,xmax,ymax]
        train_platetext.append(label)
        train_bbox.append(bbox)

    for file in os.listdir(test_xml_path):
        anno = ET.ElementTree(file=os.path.join(test_xml_path, file))
        label,xmin,ymin,xmax,ymax=parse_anno(anno)
        bbox=[xmin,ymin,xmax,ymax]
        test_platetext.append(label)
        test_bbox.append(bbox)

    for file in os.listdir(train_img_path):
        img=cv2.imread(os.path.join(train_img_path,file))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        padded_img=img_padding(img)
        transposed_img=np.transpose(padded_img,(2,0,1))
        train_img.append(transposed_img)
    
    for file in os.listdir(test_img_path):
        img=cv2.imread(os.path.join(test_img_path,file))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        padded_img=img_padding(img)
        transposed_img=np.transpose(padded_img,(2,0,1))
        test_img.append(transposed_img)
    
    train_img=torch.from_numpy(np.array(train_img)).float()
    train_bbox=torch.from_numpy(np.array(train_bbox)).float()
    test_img=torch.from_numpy(np.array(test_img)).float()
    test_bbox=torch.from_numpy(np.array(test_bbox)).float()

    assert train_set_size==len(train_img) and train_set_size==len(train_bbox), 'dimension mismatch!'
    assert test_set_size==len(test_img) and test_set_size==len(test_bbox), 'dimension mismatch!'

    return train_img,train_platetext,train_bbox,test_img,test_platetext,test_bbox
    

