# Object Detection : printed label detection 

------------------

### This project is for detecting the mis-printed label 

#### The model detecte the printed label in real time 
- using cv2 
- detected printed label using videocam 
- finetune the pretrained objecte detection model resnet50



## Sample images and Results

> Correctely printed label
<img src="./data/images/results/out01.png">

> Mis-printed Label
<img src="./data/images/results/out02.png">


## Model : 2 step:
- 1) Object detection 
    : Detect the printed label area using 
   > finetune model : ssd_resnet50_v1_fpn_640x640_coco17
- 2) Classification 
    : Classify the true / false labeled print 
   > finetune model : mobilenet 


## requirements 

- 1. install required package 
`pip install -r requirements.txt`

- 2. install imageai package
`pip install imageai --upgrade`

