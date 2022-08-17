from config import *
import numpy as np
import IPython.display as display
import PIL.Image

#Load the image and return it as a numpy array
def load_image() :
	return np.array(img)

#Display the image
def show(img_arry):
	PIL.Image.fromarray(np.array(img_arry)).show()


def process_img(img):
	pass

def load_model():
	pass




if __name__ == '__main__':
	img = load_image()
	show(img)
