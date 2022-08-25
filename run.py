from config import *
from utils import build_model, load_image, preprocess,deprocess,save_img,cv2_clipped_zoom
from deep_dream import DeepDream
import numpy as np
from cv2 import VideoWriter_fourcc,VideoWriter,cvtColor,COLOR_RGB2BGR

def generate_frames(num_frames=10):
	model = build_model()
	deepdream = DeepDream(model)
	frames = []
	for i in range(num_frames):
		result_img = deepdream()
		# save_img(np.array(result_img),name=f"img_{i}")
		nex_img = cv2_clipped_zoom(np.array(result_img),zoom_factor=1.05)
		deepdream.set_img(nex_img)
		frames.append(np.array(result_img))
	return frames


def generate_vid(frames,fps=3) :
	dream_name = 'dream_video'
	#Windows
	fourcc = VideoWriter_fourcc(*'XVID')
	# Linux:
	#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
	out = VideoWriter('{}.avi'.format(dream_name),fourcc, FPS, IMG_RES)
	for idx, frame in enumerate(frames):
		out.write(cvtColor(frame, COLOR_RGB2BGR))
		print(f"frame number {idx} written succefully. remaining : {len(frames)-idx}")
	out.release()

if __name__ == '__main__' :
	frames = generate_frames(num_frames=FRAMES)
	generate_vid(frames,fps=FPS)
