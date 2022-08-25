import tensorflow as tf
from utils import build_model, load_image, preprocess,deprocess,save_img
import numpy as np
from config import *

class DeepDream():
	def __init__(self,model, img=None,steps=None,step_size=None):
		self.model = model
		#self.img is a numpy arry of the image input to the deep dream algorithm
		if not img :
			self.img = load_image()
		else : self.img = img
		if not steps :
			self.steps = STEPS
		else : self.steps = steps
		if not step_size :
			self.step_size = 0.01
		else : self.step_size = step_size
		self.img = preprocess(self.img)
		self.img = tf.convert_to_tensor(self.img)
		print("The model summary is : \n")
		print(self.model.summary())

	def set_img(self,new_image):
		self.img = new_image

	def calc_loss(self):
		# Pass forward the image through the model to retrieve the activations.
		# Converts the image into a batch of size 1.
		img_batch = tf.expand_dims(self.img, axis=0)
		layer_activations = self.model(img_batch)
		if len(layer_activations) == 1:
			layer_activations = [layer_activations]

		losses = []
		for act in layer_activations:
			loss = tf.math.reduce_mean(act)
			losses.append(loss)

		return  tf.reduce_sum(losses)

	def optimize(self):
		loss = tf.constant(0.0)
		for n in tf.range(self.steps):
			print(f"Proceeding with step num : {n}")
			with tf.GradientTape() as tape:
				img_tensor = tf.convert_to_tensor(self.img)
				# `GradientTape` only watches `tf.Variable`s by default
				tape.watch(img_tensor)
				loss = self.calc_loss()
			print(f"loss calculated succefully, loss = {loss}")	
			# Calculate the gradient of the loss with respect to the pixels of the input image.
			gradients = tape.gradient(loss, img_tensor)
			print(gradients)
			# Normalize the gradients.
			gradients /= tf.math.reduce_std(gradients) + 1e-8 

			# In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
			# We can update the image by directly adding the gradients (because they're the same shape!)
			img = img_tensor + gradients*tf.convert_to_tensor(self.step_size)
			img = tf.clip_by_value(img, -1, 1)
			# print(f"Done with step num : {n}")	
		return loss, img

	def run_deep_dream(self):
		pass
		
	def __call__(self) :
		# img = preprocess(self.img)
		# img = tf.convert_to_tensor(img)
		# step_size = tf.convert_to_tensor(self.step_size)
		loss , img = self.optimize()
		result = deprocess(img)
		return result


if __name__ == '__main__' :
	model = build_model()
	deepdream = DeepDream(model)
	result_img = deepdream()
	save_img(np.array(result_img))