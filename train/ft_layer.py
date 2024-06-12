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
		
		# Stores 1 layer of softmax input for RHS only
		# This is valid only if the activation value is softmax. Its needed because softmax is irreversible
		self.last_softmax_input = None

		# This is to store the loss values for RHS during back propagation 
		# NOTE: This is a np.array
		self.rhs_loss = None

		# This is to store the weight derivatives during back propagation 
		# NOTE: This is a np.array
		self.pending_weights_derivatives = np.zeros(self.weights.shape)

		# This is to store the bias derivatives during back propagation 
		# NOTE: This is a np.array
		self.pending_bias_derivatives = np.zeros(self.bias.shape)

		# This is to store the softmax output for output layers
		# NOTE: This is a np.array
		self.softmax_output = None

	# runs activation functions for current layer, and sets the next layers activation function output
	def run_activation(self):
		x_values = np.add(np.matmul(self.weights, self.lhs_activation), self.bias)
		col_scalar = ft_math.single_column_to_scalar(x_values)
		activated_output = []
		# NOTE: x_values goes big here and col_scalar sums to 0
		# logging.info(f"col_scalar {col_scalar} x_values {x_values} mean : {ft_math.mean(col_scalar)}")
		if self.activation_fn == "softplus":
			for x_value in col_scalar:
				activated_output.append(ft_math.softplus(x_value))
		if self.activation_fn == "sigmoid":
			for x_value in col_scalar:
				activated_output.append(ft_math.sigmoid(x_value))
		if activated_output == []:
			raise ValueError(f"Invalid activation function {self.activation_fn}")

		new_rhs = ft_math.scalar_to_single_column(activated_output)
		# logging.info(f"\n\tweight\n{self.weights}\n\tlhs\n\t{self.lhs_activation}\n\tbias\n{self.bias}\n\tx_values\n{x_values}\n\trhs\n{new_rhs}")
		self.rhs_activation = new_rhs
		
		# logging.info(f"\n{self.weights}\n*\n{self.lhs_activation}\n+\n{self.bias}\n=\n{new_rhs}")