import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import numpy as np
import logging
import ft_model, ft_layer, ft_math
import random

class Ft_perceptron:
	def __init__(self,
			  hidden_layers,
			  epoch_count,
			  output_loss_type,
			  batch_size,
			  learning_rate,
			  dataset_train,
			  dataset_test,
			  momentum_decay = 0.5,
			  rmsprop_ratio = 0.999,
			  rmsprop_stabilizer = 10 ** -8,
			  ):
		self.epoch_count = epoch_count
		self.output_loss_type = output_loss_type
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dataset_train = dataset_train
		self.dataset_test = dataset_test
		self.enum_models = ft_model.get_enumerable_models()
		self.momentum_decay = momentum_decay
		self.rmsprop_ratio = rmsprop_ratio
		self.rmsprop_stabilizer = rmsprop_stabilizer

		self.layers = []
		
		# generate input layer
		self.layers.append(ft_layer.Ft_layer(
			type="input",
			node_count_lhs=len(self.enum_models),
			node_count_rhs=hidden_layers[0],
			seed=132
		))

		# generate hidden layers
		i = 0
		while i < len(hidden_layers) - 1 :
			self.layers.append(ft_layer.Ft_layer(
				type="hidden",
				node_count_lhs=hidden_layers[i],
				node_count_rhs=hidden_layers[i + 1],
				seed= i + 1 * 2
			))
			i += 1

		# generate output layer
		self.layers.append(ft_layer.Ft_layer(
			type="output",
			node_count_lhs=hidden_layers[-1],
			node_count_rhs=len(ft_model.DIAGNOSIS),
			activation_fn="softmax",
			seed=i * 2
		))

		# generate true values for each training data
		train_truth = []
		for data in dataset_train:
			truth_matrix = []
			truth_name = ft_model.TRUTH_MODEL["name"]
			truth_value = data.get_feature(truth_name)
			for cmp in ft_model.DIAGNOSIS:
				if cmp == truth_value:
					truth_matrix.append(1)
				else:
					truth_matrix.append(0)
			train_truth.append(truth_matrix)

		self.train_truth = np.array(train_truth).T

		# generate true valeus for each test data
		test_truth = []
		for data in dataset_test:
			truth_matrix = []
			truth_name = ft_model.TRUTH_MODEL["name"]
			truth_value = data.get_feature(truth_name)
			for cmp in ft_model.DIAGNOSIS:
				if cmp == truth_value:
					truth_matrix.append(1)
				else:
					truth_matrix.append(0)
			test_truth.append(truth_matrix)

		self.test_truth = np.array(test_truth).T

		# store historic data here?
		logging.info("Perceptron initialized")

	def generate_input_matrix(self, dataset):
		# fill input data
		inputs = []
		for model in self.enum_models:
			inputs.append([])

		for idx,model in enumerate(self.enum_models):
			feature_name = model["name"]
			for entry in dataset:
				feature_value = entry.get_feature(feature_name)
				inputs[idx].append(feature_value)

		return np.array(inputs)

	def begin_train_alt(self):
		logging.info("Alt Training begin")

		# fill input data
		inputs = []
		for model in self.enum_models:
			inputs.append([])

		for idx,model in enumerate(self.enum_models):
			feature_name = model["name"]
			for entry in self.dataset_train:
				feature_value = entry.get_feature(feature_name)
				inputs[idx].append(feature_value)


		self.layers[0].lhs_activation = np.array(inputs)

		print(f"inputs is {inputs}")

		# get truth data
		truth = self.train_truth

		learning_rate = 0.314

		# epoch loop
		epochs = 70
		for i in range(epochs):
			# TODO: use my own functions in this - samson zhang
			# fw prop
			input_activation = self.layers[0].lhs_activation
			weights_input_to_hidden = self.layers[0].weights
			bias_input_to_hidden = self.layers[0].bias
			weights_hidden_to_output = self.layers[1].weights
			bias_hidden_to_output = self.layers[1].bias

			z1 = np.add(np.dot(weights_input_to_hidden, input_activation), bias_input_to_hidden)
			a1 = np.array(ft_math.sigmoid(z1))

			z2 = np.add(np.dot(weights_hidden_to_output, a1), bias_hidden_to_output)
			a2 = np.array(ft_math.softmax(z2))
			# logging.info(a2)
			# bw prop

			# this is straight up error function??? no. this is softmax derivatinve wrt its input with bonary cross entropy included
			# dz2 = a2 - truth
			dz2 = ft_math.dcost_dz_output_np(a2, truth)
			# logging.info(f"\ndz2_theirs\n{dz2}\ndz2_mine\n{ft_math.dcost_dz_output_np(a2, truth)}\npredicted\n{np.array(a2)}\ntruth {truth}")

			# dw2 = dz2.dot(a1.T)
			dw2 = ft_math.dcost_dw_output_np(dz2, a1)
			# logging.info(f"\ndww_theirs\n{dw2}\ndw2_mine\n{ft_math.dcost_dw_output_np(dz2, a1)}\npredicted\n{np.array(a2)}\ntruth {truth}")


			# db2 = np.sum(dz2)
			db2 = ft_math.dcost_db_output_np(dz2)
			# logging.info(f"\ndb2_theirs\n{db2}\ndb2_mine\n{ft_math.dcost_dw_output_np(dz2, a1)}\npredicted\n{np.array(a2)}\ntruth {truth}")

			# ft_math.binary_cross_entropy_np(np.array(a2), truth)
			# logging.info(a1)
			# logging.info(f"dz2_theirs {dz2}, dz2_mine = {ft_math.binary_cross_entropy_deriv_activation_np(np.array(a2), truth)}, pred {np.array(a2)} truth {truth}")

			# dz_dalhs * dc_da * da_dz
			# dz1 = weights_hidden_to_output.T.dot(dz2) * ft_math.sigmoid_deriv(ft_math.sigmoid_reverse(a1))
			# print(f"weights_hidden_to_output {weights_hidden_to_output.shape} dz2 {dz2.shape} a1 {a1.shape}")
			dz1 = ft_math.dcost_dz_hidden_np(weights_hidden_to_output, dz2, a1)
			# logging.info(f"dz1_theirs {dz1}, dz1_mine = {ft_math.dcost_dz_hidden_np(weights_hidden_to_output, dz2, a1)}, pred {np.array(a2)} truth {truth}")

			# dw1 = dz1.dot(input_activation.T)
			dw1 = ft_math.dcost_dw_hidden_np(dz1, input_activation)

			db1 = np.sum(dz1)
			# db1 = dz1

			# logging.info(f"\npredicted\n{a2},\ntruth\n{truth}")
			
			
			self.layers[0].weights = self.layers[0].weights - learning_rate * dw1
			self.layers[0].bias = self.layers[0].bias - learning_rate * db1
			self.layers[1].weights = self.layers[1].weights - learning_rate * dw2
			self.layers[1].bias = self.layers[1].bias - learning_rate * db2

	def begin_train(self):
		truth_train = self.train_truth
		truth_test = self.test_truth
		warmup_threshold = 128 # how many epochs to run before we check for early stopping?
		last_test_accuracy = 0

		logging.info(f"Training begin with {warmup_threshold} early stop warmup epochs")
		random.seed(69)
		for i in range(self.epoch_count):
			# TODO: implement Recall metric alongside loss
			# generate batch randomly based on batch size
			indices = None
			if self.batch_size < 0:
				indices = range(0, len(self.dataset_train))
			else :
				indices = random.sample(population=range(0, len(self.dataset_train)), k=self.batch_size)
			train_batches = list(map(lambda x: self.dataset_train[x], indices))
			logging.debug(f"mini-batch indices {indices}")
			
			truth_batch_scalar = []
			for col_idx, col in enumerate(truth_train[0]):
				if col_idx not in indices:
					continue
				truth_1 = self.train_truth[0][col_idx]
				truth_2 = self.train_truth[1][col_idx]
				truth_batch_scalar.append([truth_1, truth_2])
			train_batch_truths = np.array(truth_batch_scalar).T

			# forwardfeed and backpropagate
			self.layers[0].lhs_activation = self.generate_input_matrix(train_batches)
			last_layer_error_train = self.feed_forward_and_backprop_train(train_batch_truths)
			accuracy_train = ft_math.get_accuracy(self.layers[-1].rhs_activation, train_batch_truths)

			# # forwardfeed and backpropagate test set
			self.layers[0].lhs_activation = self.generate_input_matrix(self.dataset_test)
			last_layer_error_test = self.feed_forward_test(truth_test)
			accuracy_test = ft_math.get_accuracy(self.layers[-1].rhs_activation, truth_test)

			test_error = np.abs(np.mean(last_layer_error_test))
			train_error = np.abs(np.mean(last_layer_error_train))
			# check for early stopping
			if i >= warmup_threshold :
				if accuracy_test < last_test_accuracy:
					logging.info(f"Epoch {i} finished; train loss {round(train_error, 3)}, validation loss {round(test_error, 3)} validation acc {accuracy_test} < {last_test_accuracy}, early stopping condition met, stopping.")
					break
				last_test_accuracy = accuracy_test

			# apply weight changes
			self.apply_derivatives_reset_cache()
			logging.info(f"Epoch {i} finished; train loss {round(train_error, 3)}, validation loss {round(test_error, 3)} validation acc {round(accuracy_test, 3)}")

	def feed_forward(self):
		# iterate through all layers (assume input layer LHS is already set)
		for layer_idx, layer in enumerate(self.layers):			
			# run activation function, this would populate the current layer.rhs
			layer.run_activation()

			# set next layer lhs depends on current layer type
			if layer.type != "output":
				self.layers[layer_idx + 1].lhs_activation = layer.rhs_activation

			logging.debug(f"[feed_forward]\n{layer.type} layer {layer_idx} activation\n{layer.rhs_activation}")
		last_layer_activation = self.layers[-1].rhs_activation
		return last_layer_activation

	def backprop(self, truth):
		# this value will change to store the current layers dz value
		# for the previous layer to access
		last_dz = None

		# this value will change to store the current layers weights
		# for the previous layer to access
		last_layer_weights = None

		for layer_idx, layer in enumerate(reversed(self.layers)):
			# logging.info(f"running backprop for layer {layer.type} @ {idx}")
			if layer.type == "output":
				# test = ft_math.softmax
				dz = ft_math.dcost_dz_output_np(layer.rhs_activation, truth, layer.pre_softmax_x_values, self.output_loss_type)
				dw = ft_math.dcost_dw_output_np(dz, layer.lhs_activation)
				db = ft_math.dcost_db_output_np(dz)
				layer.pending_weights_derivatives = dw
				layer.pending_bias_derivatives = db
				last_dz = dz
				last_layer_weights = layer.weights
				logging.debug(f"[backprop]\n{layer.type} layer {len(self.layers) - layer_idx} dz\n{dz}\ndw\n{dw}\ndb\n{db}")
			else :
				# logging.info(f"weights_hidden_to_output {last_layer_weights.shape} dz2 {last_dz.shape} a1 {layer.rhs_activation.shape}")
				dz = ft_math.dcost_dz_hidden_np(last_layer_weights, last_dz, layer.rhs_activation)
				dw = ft_math.dcost_dw_hidden_np(dz, layer.lhs_activation)
				db = ft_math.dcost_db_hidden_np(dz)
				layer.pending_weights_derivatives = dw
				layer.pending_bias_derivatives = db
				last_dz = dz
				last_layer_weights = layer.weights
				logging.debug(f"[backprop]\n{layer.type} layer {len(self.layers) - layer_idx} dz\n{dz}\ndw\n{dw}\ndb\n{db}")

	# feeds forward and backpropagates for 1 example, appending the required changes in the
	# layers themselves
	def feed_forward_and_backprop_train(self, truth):
		# run feed forward as usual
		last_layer_activation = self.feed_forward()
		# logging.info(f"activation {ft_math.single_columADD this n_to_scalar(last_layer_activation)} truth {ft_math.single_column_to_scalar(truth)}")
		last_layer_error = None
		if self.output_loss_type == "binaryCrossEntropy":
			last_layer_error = ft_math.binary_cross_entropy(
						last_layer_activation,
						truth
						)
		elif self.output_loss_type == "MSE":
			last_layer_error = ft_math.squared_err(
						last_layer_activation,
						truth
						)
		else:
			raise ValueError("Invalid loss detected in feed_forward_and_backprop_train")
		# logging.info(f"truth {ft_math.single_column_to_scalar(truth)}")
		# logging.info(f"raw entropy for entry {entry.get_feature('Id')} : {last_layer_entropy}")

		# run back propagation to get derivatives for all weights and biases
		self.backprop(truth)
		return last_layer_error

	def feed_forward_test(self, truth):
		last_layer_activation = self.feed_forward()
		if self.output_loss_type == "binaryCrossEntropy":
			last_layer_error = ft_math.binary_cross_entropy(
						last_layer_activation,
						truth
						)
		elif self.output_loss_type == "MSE":
			last_layer_error = ft_math.squared_err(
						last_layer_activation,
						truth
						)
		else:
			raise ValueError("Invalid loss detected in feed_forward_and_backprop_train")
		return last_layer_error

	def apply_derivatives_reset_cache(self):
		for idx, layer in enumerate(self.layers):
			# apply rmsprop
			new_s_dw = (self.rmsprop_ratio * layer.s_weights) + ((1 - self.rmsprop_ratio) * np.square(layer.pending_weights_derivatives))
			rms_w_derivatives = layer.pending_weights_derivatives / np.sqrt(new_s_dw + self.rmsprop_stabilizer)
			layer.s_weights = new_s_dw

			new_s_db = (self.rmsprop_ratio * layer.s_bias) + ((1 - self.rmsprop_ratio) * np.square(layer.pending_bias_derivatives))
			rms_b_derivatives = layer.pending_bias_derivatives / np.sqrt(new_s_db + self.rmsprop_stabilizer)
			layer.s_bias = new_s_db

			# apply momentum
			new_weights_velocity = self.momentum_decay * layer.weights_velocity - (self.learning_rate * rms_w_derivatives)
			layer.weights_velocity = new_weights_velocity

			new_bias_velocity = self.momentum_decay * layer.bias_velocity - (self.learning_rate * rms_b_derivatives)
			layer.bias_velocity = new_bias_velocity

			layer.weights = layer.weights + new_weights_velocity
			layer.bias = layer.bias + new_bias_velocity

			# clear cache matrix
			layer.pending_weights_derivatives = np.zeros(layer.weights.shape)
			layer.pending_bias_derivatives = np.zeros(layer.bias.shape)