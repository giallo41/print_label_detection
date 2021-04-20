import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image

from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
#from IPython.display import display, Javascript
#from IPython.display import Image as IPyImage

#from object_detection.utils import visualization_utils as viz_utils

import sys
sys.path.append('/home/jupyter/git/models/research')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def image_impose(img, heatmap, alpha=0.003):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    
    return superimposed_img

def disply_heatmap(img, model, last_conv_layer_name):
    if len(img.shape)<4:
        input_img = np.expand_dims(img, 0)
    else :
        input_img = img
        
    heatmap = make_gradcam_heatmap(input_img, model, last_conv_layer_name)
    
    output_img = image_impose(img, heatmap)
    return output_img

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
            prob = 1-prob
        txt = f'{txt} : {prob:.2f}'

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      rgb, 4)
        
        text_y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, txt, (xmin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
        
    return img, result_class


def show_result_img(img, pred, detections, pixels=30, img_show=True):
    # Plot processed figures and result class 
    
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
            prob = 1-prob
        txt = f'{txt} : {prob:.2f}'

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      rgb, 4)
        
        text_y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        cv2.putText(img, txt, (xmin, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 2)
	# show the output image
    #cv2.imshow("Output", img)
    if img_show:
        plt.figure(figsize=(6, 4), dpi=80)
        plt.imshow(img)
    return img, result_class
    

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