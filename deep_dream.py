import tensorflow as tf
from utils import build_model, load_image, preprocess,deprocess,save_img
import numpy as np

class DeepDream():
	def __init__(self,model, img=None,steps=None,step_size=None):
		self.model = model
		#self.img is a numpy arry of the image input to the deep dream algorithm
		if not img :
			self.img = load_image()
		else : self.img = img
		if not steps :
			self.steps = 100
		else : self.steps = steps
		if not step_size :
			self.step_size = 0.01
		else : self.step_size = step_size

		print("The model summary is : \n")
		print(self.model.summary())


	def calc_loss(self,img, model):
		# Pass forward the image through the model to retrieve the activations.
		# Converts the image into a batch of size 1.
		img_batch = tf.expand_dims(img, axis=0)
		layer_activations = model(img_batch)
		if len(layer_activations) == 1:
			layer_activations = [layer_activations]

		losses = []
		for act in layer_activations:
			loss = tf.math.reduce_mean(act)
			losses.append(loss)

		return  tf.reduce_sum(losses)

	def optimize(self,img, steps, step_size):
		loss = tf.constant(0.0)
		for n in tf.range(self.steps):
			print(f"Proceeding with step num : {n}")
			with tf.GradientTape() as tape:
				# `GradientTape` only watches `tf.Variable`s by default
				tape.watch(img)
				loss = self.calc_loss(img, self.model)
			print(f"loss calculated succefully, loss = {loss}")	
			# Calculate the gradient of the loss with respect to the pixels of the input image.
			gradients = tape.gradient(loss, img)

			# Normalize the gradients.
			gradients /= tf.math.reduce_std(gradients) + 1e-8 

			# In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
			# You can update the image by directly adding the gradients (because they're the same shape!)
			img = img + gradients*self.step_size
			img = tf.clip_by_value(img, -1, 1)
			print(f"Done with step num : {n}")	
		return loss, img

	def __call__(self) :
		img = preprocess(self.img)
		img = tf.convert_to_tensor(img)
		step_size = tf.convert_to_tensor(self.step_size)
		loss , img = self.optimize(img,self.steps,step_size)
		result = deprocess(img)
		return result


if __name__ == '__main__' :
	model = build_model()
	deepdream = DeepDream(model)
	result_img = deepdream()
	save_img(np.array(result_img))