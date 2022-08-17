import tensorflow as tf
from utils import build_model, load_image


class DeepDream():
	def __init__(self,model, img=None):
		self.model = model
		if not img :
			self.img = load_image()
		else : self.img = img
		print("The model summary is : \n")
		print(self.model.summary())


	def calc_loss(img, model):
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

		def optimize(self):
			pass



if __name__ == '__main__' :
	model = build_model()
	deepdream = DeepDream(model)