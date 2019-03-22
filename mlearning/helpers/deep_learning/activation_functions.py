"""
	This moddule contains  activation functions that are
	used to give the output of the network from the biased inputs.
"""

import numpy as np

class Rectified_Linear_Units:
	"""
		A rectified linear units activation function
	"""
	def __call__(self, x):
		"""
			Runs a ReLu function for given input and
			returs the ouput
		"""
		return np.where(x >= 0, x, 0)

	def find_grad(self):
		"""Calculates the gradient (ReLu : 0 < x 1)

		"""
		return np.where(x >= 0, 1, 0)

