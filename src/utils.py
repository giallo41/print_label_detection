import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
#from IPython.display import display, Javascript
#from IPython.display import Image as IPyImage

#from object_detection.utils import visualization_utils as viz_utils

import sys
sys.path.append('/home/jupyter/git/models/research')

def cv_img_result(img, pred, detections, pixels=30):

    # return processed image and class list 
    result_class = [ 'true' if i>0.5  else 'false' for i in np.reshape(pred, (len(pred))) ]
    prob_list = [i for i in np.reshape(pred, (len(pred)))]
    
    image_h, image_w, _ = img.shape
    
    for txt, prob, bbox in zip(result_class, prob_list, detections):
        
        xmin, ymin, xmax, ymax = bbox['box_points']
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_w, xmax)
        ymax = min(image_h, ymax)
    
        if txt == 'true':
            rgb = (255,0,0) # Blue 
        else:
            rgb = (0,0,255) # Red
        txt = f'{txt} : {prob:.2f}'

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      rgb, 4)
        
        text_y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, txt, (xmin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
        
    return img, result_class


def show_result_img(img, pred, detections, pixels=30):
    # Plot processed figures and result class 
    plt.figure(figsize=(12, 10), dpi=80)
    result_class = [ 'true' if i>0.5  else 'false' for i in np.reshape(pred, (len(pred))) ]
    prob_list = [i for i in np.reshape(pred, (len(pred)))]
    
    image_h, image_w, _ = img.shape
    
    for txt, prob, bbox in zip(result_class, prob_list, detections):
        
        xmin, ymin, xmax, ymax = bbox['box_points']
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_w, xmax)
        ymax = min(image_h, ymax)
        
        if txt == 'true':
            rgb = (0,0,255)
        else:
            rgb = (255,0,0)
        txt = f'{txt} : {prob:.2f}'

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      rgb, 4)
        
        text_y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, txt, (xmin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
	# show the output image
    #cv2.imshow("Output", img)
    plt.imshow(img)
    return result_class
    

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
          and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
          this function assumes that the boxes to be plotted are groundtruth
          boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
          category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    
    image_np_with_annotations = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.8)
    
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    
    else:
        plt.imshow(image_np_with_annotations)

def load_img(image_path, plot=False):
    '''
    Import the image.
    
    Arguments:
    ---------
        image_path:
            Path to the image
        plot:
            Plot or not
            
    Returns:
    --------
        img:
            Array of type uint8 contains the RGB values of the image
    '''
    img_orig = cv2.imread(image_path,1)
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    if plot==True:
        plt.figure()
        plt.imshow(img)
    return img