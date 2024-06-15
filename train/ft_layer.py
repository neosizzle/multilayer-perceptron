import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import logging
import numpy as np
import ft_math

class Ft_layer:
	def __init__(self,
				type,
				node_count_lhs,
				node_count_rhs,
				seed,
				init_weights=None,
				init_bias=None,
				activation_fn="sigmoid"
			  ):
		# rng = np.random.default_rng()
		rng = np.random.default_rng(seed)
		
		if init_bias != None:
			self.bias = np.array(init_bias)
		else:
			# self.bias = rng.normal(size=(node_count_rhs, 1))
			self.bias = np.zeros((node_count_rhs, 1))
		
		if init_weights != None:
			self.weights = np.array(init_weights)
		else:
			self.weights = rng.normal(size=(node_count_rhs, node_count_lhs))
			# logging.info(rng.random((node_count_rhs, node_count_lhs)))
		logging.debug(f"{type} layer init with bias {self.bias} and weight {self.weights} {node_count_rhs}x{node_count_lhs}")

		self.type = type
		self.activation_fn = activation_fn
		self.lhs_activation = None
		self.rhs_activation = None

		# This is to store the loss values for RHS during back propagation 
		# NOTE: This is a np.array
		self.rhs_loss = None

		# This is to store the weight derivatives during back propagation 
		# NOTE: This is a np.array
		self.pending_weights_derivatives = np.zeros(self.weights.shape)

		# This is to store the bias derivatives during back propagation 
		# NOTE: This is a np.array
		self.pending_bias_derivatives = np.zeros(self.bias.shape)

		# This is to store the previous weight derivatives for momentum calculation 
		# NOTE: This is a np.array
		self.weights_velocity = np.zeros(self.weights.shape)

		# This is to store the previous bias derivatives for momentum calculation 
		# NOTE: This is a np.array
		self.bias_velocity = np.zeros(self.bias.shape)

		# This is to store the previous squared weight derivatives for RMS propagation 
		# NOTE: This is a np.array
		self.s_weights = np.zeros(self.weights.shape)

		# This is to store the previous squared bias derivatives for RMS propagation 
		# NOTE: This is a np.array
		self.s_bias = np.zeros(self.bias.shape)

		# This is to store the projected weight derivatives for momentum calculation 
		# NOTE: This is a np.array
		# self.projected_weights_derivatives = np.zeros(self.weights.shape)

	# runs activation functions for current layer, and sets the next layers activation function output
	def run_activation(self):
		x_values = np.add(np.dot(self.weights, self.lhs_activation), self.bias)
		activated_output = None
		if self.activation_fn == "softplus":
			activated_output = np.array(ft_math.softplus(x_values))
		elif self.activation_fn == "sigmoid":
			activated_output = np.array(ft_math.sigmoid(x_values))
		elif self.activation_fn == "softmax":
			if self.type != "output":
				raise ValueError("Layer is not output type but softmax is requested")
			activated_output = np.array(ft_math.softmax(x_values))
		else :
			raise ValueError(f"Invalid activation function {self.activation_fn}")
		# logging.info(f"\n\tweight\n{self.weights}\n\tlhs\n\t{self.lhs_activation}\n\tbias\n{self.bias}\n\tx_values\n{x_values}\n\trhs\n{new_rhs}")
		self.rhs_activation = activated_output
		
		# logging.info(f"\n{self.weights}\n*\n{self.lhs_activation}\n+\n{self.bias}\n=\n{new_rhs}")