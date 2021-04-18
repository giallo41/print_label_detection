import os
import cv2
import time
import numpy as np
import argparse
import tensorflow as tf

from src.data import extract_roi
from src.utils import cv_img_result

from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()

D_MODEL_PATH = f"./data/images/box/models/detection_model-ex-030--loss-0010.144.h5"
JSON_FILE_PATH = f"./data/images/box/json/detection_config.json"
C_MODEL_PATH = f"./data/models/mobilnet/mobilenet.h5"

def model_load(detection_model_path,
               json_path,
               cls_model_path,):
    
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    
    detector.setModelPath(detection_model_path) 
    detector.setJsonPath(json_path)
    detector.loadModel()
    
    model = tf.keras.models.load_model(cls_model_path)
    
    return detector, model

def main():
    import cv2
    #parser = argparse.ArgumentParser(description='Object detection')
    #parser.add_argument('json', type=str, default=JSON_FILE_PATH,
    #        help='json file')

    #parser.add_argument('detection_model', type=str, default=D_MODEL_PATH,
    #        help='Detection Model')

    #parser.add_argument('class_model', type=str, default=C_MODEL_PATH,
    #        help='classification Model')

    #args = parser.parse_args()

    json_path = JSON_FILE_PATH#args.json
    detection_model_path = D_MODEL_PATH#args.detection_model
    cls_model_path = C_MODEL_PATH#args.class_model

    detector, model = model_load(detection_model_path,
                                 json_path,
                                 cls_model_path)
    
    print (f"Sucessfully loaded models")

    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read()

        detections = []
        # Detect the object 
        _, detections = detector.detectObjectsFromImage(input_image=frame,
                                                        input_type='array',
                                                        output_type='array')

        if len(detections)>0:
            roi_list = extract_roi(frame, detections)
            # Classify the object
            pred = model.predict(np.array(roi_list))

            frame, result_class = cv_img_result(frame, pred, detections)

        # show the output image
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()