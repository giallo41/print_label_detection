
import pickle
import cv2
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.utils import shuffle as sklearn_shuffle
from keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess

def extract_roi(image, outputs, pixels = 50):
    image_h, image_w, _ = image.shape
    
    roi_list = []
    for output in outputs:
        xmin, ymin, xmax, ymax = output['box_points']
        xmin = max(0, xmin-pixels)
        ymin = max(0, ymin-pixels)
        xmax = min(image_w, xmax+pixels)
        ymax = min(image_h, ymax+pixels)
        img = image[ymin:ymax,xmin:xmax,:]
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
        roi_list.append(mobile_preprocess(img))
    return np.array(roi_list)

def get_file_list(path):
    file_list = os.listdir(path)
    if '.ipynb_checkpoints' in file_list:
        idx = file_list.index('.ipynb_checkpoints')
        file_list.pop(idx)
        
    file_list = [f'{os.path.join(path,file)}' for file in file_list]
    return file_list

def get_image_value(path, dim, bw, model_type): 
    '''This function will read an image and convert to a specified version and resize depending on which algorithm is being used.  If edge is specified as true, it will pass the img array to get_edged which returns a filtered version of the img'''
    img = image.load_img(path, target_size = dim)
    img = image.img_to_array(img)
    if bw == True: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if model_type.upper() != 'Normal': 
            img = np.stack((img,)*3, axis =-1)
        else: 
            img = img.reshape(img.shape[0], img.shape[1],1)


    #if model_type.upper() == 'MOBILENET': 
    #    img = mobile_preprocess(img)
    #    return img
    #elif model_type.upper() == 'VGG16': 
    #    img = vgg16_preprocess(img) 
    #    return img
    return img/255.


def get_data(class_type):
    file_list = os.listdir(f'./data/images/syn/{class_type}/true')
    if '.ipynb_checkpoints' in file_list:
        idx = file_list.index('.ipynb_checkpoints')
        file_list.pop(idx)
    
    true_paths = [f'./data/images/syn/{class_type}/true/{file}' for file in file_list]
    true_labels = [1 for i in range(len(true_paths))]
    
    file_list = os.listdir(f'./data/images/syn/{class_type}/false')
    if '.ipynb_checkpoints' in file_list:
        idx = file_list.index('.ipynb_checkpoints')
        file_list.pop(idx)
    
    false_paths = [f'./data/images/syn/{class_type}/false/{file}' for file in file_list]
    false_labels = [0 for i in range(len(false_paths))]
    
    labels = np.array(true_labels + false_labels)
    print(f'{class_type.upper()} Value Counts')
    print(pd.Series(labels).value_counts())
    paths = np.array(true_paths + false_paths)
    #labels = to_categorical(labels)
    if class_type == 'train':
        paths, labels = sklearn_shuffle(paths, labels)
    return paths, labels

def get_bbox_image(path, train_test):
    
    img_path_list = []
    bbox_list = []
    with open(f"{path}/label/{train_test}.txt", 'r') as f:
        #for line in f.readline():
        lines = f.readlines()
        for line in lines:
            file, *bbox = line.strip().split( ',')
            img_path_list.append(file)
            b_list = []
            for item in bbox:
                b_list.append(float(item))
            bbox_list.append(b_list)
    
    img_path_list = [f"{path}/{train_test}/{file}" for file in img_path_list]
    
    return img_path_list, np.array(bbox_list)