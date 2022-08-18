from config import *
from utils import build_model, load_image, preprocess,deprocess,save_img
from deep_dream import DeepDream
import numpy as np

def generate_frames(img,num_frames=10):
	model = build_model()
	deepdream = DeepDream(model)
	frames = []
	for i in range(num_frames):
		result_img = deepdream()
		save_img(np.array(result_img),name=f"img_{i}")
		deepdream.set_img(result_img)
		frames.append(np.array(result_img))
	return frames


def generate_vid(frames,resolution) :
	pass


if __name__ == '__main__' :
	generate_frames()




