from config import *
import numpy as np
import IPython.display as display
import PIL.Image
import tensorflow as tf

#Load the image and return it as a numpy array
def load_image() :
	return np.array(img)

#Display the image
def show(img_arry):
	PIL.Image.fromarray(np.array(img_arry)).show()

#Preprocess the image for the Inception model
def preprocess_input(img):
    x /= 255.
    x -= 0.5
    x *= 2.
    return tf.cast(x,tf.uint8)


def build_model():
	base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	layers = [base_model.get_layer(name).output for name in names_layers]
	return tf.keras.Model(inputs=base_model.input,outputs=layers)

if __name__ == '__main__':
	img = load_image()
	show(img)
