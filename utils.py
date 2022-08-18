from config import *
import numpy as np
import IPython.display as display
import PIL.Image
import tensorflow as tf
from os.path import join
from cv2 import resize

#Load the image and return it as a numpy array
def load_image() :
	return np.array(img)

#Display the image
def show(img_arry):
	PIL.Image.fromarray(np.array(img_arry)).show()

#Preprocess the image for the Inception model
def preprocess(img):
	return tf.keras.applications.inception_v3.preprocess_input(img)

def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def save_img(img_arry,name='test') :
	PIL.Image.fromarray(np.array(img_arry)).save(join(DATA_FOLDER,IMG_FLODER,name+".jpg"))

def build_model():
	base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	layers = [base_model.get_layer(name).output for name in names_layers]
	return tf.keras.Model(inputs=base_model.input,outputs=layers)


def cv2_clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

if __name__ == '__main__' :
	img = load_image()
	show(img)
	img_zoome = cv2_clipped_zoom(img,zoom_factor=1.05)
	show(img_zoome)