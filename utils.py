from config import *
import numpy as np
import IPython.display as display
import PIL.Image
import tensorflow as tf
from os.path import join


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

def save_img(img_arry,name='test.jpg') :
	PIL.Image.fromarray(np.array(img_arry)).save(join(DATA_FOLDER,name))

def build_model():
	base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	layers = [base_model.get_layer(name).output for name in names_layers]
	return tf.keras.Model(inputs=base_model.input,outputs=layers)

if __name__ == '__main__' :
	img = load_image()
	show(img)
