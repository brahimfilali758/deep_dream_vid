#This file contains all the cofiguration variables that are shared between different modules

from PIL import Image
from os.path import join


img_path = join('data','img_input.jpg')
img = Image.open(img_path)
IMG_RES = img.size #resolution of input image
print(f"the image resolution is : {IMG_RES}")
