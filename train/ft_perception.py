import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import numpy as np
import logging
import ft_model, ft_layer, ft_math

class Ft_perceptron:
	def __init__(self,
			  hidden_layers,
			  epoch_count,
			  output_loss_type,
			  batch_size,
			  learning_rate,
			  dataset_train,
			  dataset_test,
			  ):
		self.epoch_count = epoch_count
		self.output_loss_type = output_loss_type
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.dataset_train = dataset_train
		self.dataset_test = dataset_test
		self.enum_models = ft_model.get_enumerable_models()

		self.layers = []
		
		# generate input layer
		self.layers.append(ft_layer.Ft_layer(
			type="input",
			node_count_lhs=len(self.enum_models),
			node_count_rhs=hidden_layers[0],
			seed=1323
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

	def begin_train_alt(self):
		logging.info("Alt Training begin")

		# fll input data
		inputs = []
		for model in self.enum_models:
			inputs.append([])

		for idx,model in enumerate(self.enum_models):
			feature_name = model["name"]
			feature_value = self.dataset_train[0].get_feature(feature_name)
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

			# this is straight up error function??? no. this is softmax derivatinve wrt its input
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
			dz1 = ft_math.dcost_dz_hidden_np(weights_hidden_to_output, dz2, a1)
			# logging.info(f"dz1_theirs {dz1}, dz1_mine = {ft_math.dcost_dz_hidden_np(weights_hidden_to_output, dz2, a1)}, pred {np.array(a2)} truth {truth}")

			# dw1 = dz1.dot(input_activation.T)
			dw1 = ft_math.dcost_dw_hidden_np(dz1, input_activation)

			db1 = np.sum(dz1)
			# db1 = dz1

			logging.info(f"\npredicted\n{a2},\ntruth\n{truth}")
			
			self.layers[0].weights = self.layers[0].weights - learning_rate * dw1
			self.layers[0].bias = self.layers[0].bias - learning_rate * db1
			self.layers[1].weights = self.layers[1].weights - learning_rate * dw2
			self.layers[1].bias = self.layers[1].bias - learning_rate * db2

	def begin_train(self):
		logging.info("Training begin")
		# run procedure function epoch_count times
		for i in range(self.epoch_count):

			# TODO: implement Recall metric alongside loss
			epoch_loss = [0] * len(ft_model.DIAGNOSIS)

			# TODO batch GD here?
			for data_idx, data in enumerate(self.dataset_train):
				# for each entry, call feed_forward and store predictions
				# assume length of train set is = to the length of truth because they 
				# are derived from same csv
				example_loss = self.feed_forward_and_backprop(data, self.train_truth[data_idx], data_idx)

				# append loss to entropy losses
				for idx, loss in enumerate(example_loss) :
					epoch_loss[idx] += loss
				
				# TODO: remove testing code 
				# break

			# TODO store visualize data here (along with test data)? 

			# apply weight and bias changes
			self.apply_derivatives_reset_cache()
			total_loss = None
			if self.output_loss_type == "binaryCrossEntropy":
				total_loss = sum(epoch_loss) * (-1 * (1 / len(self.dataset_train)))
			else:
				total_loss = sum(epoch_loss) / len(self.dataset_train)
			logging.info(f"Epoch {i} finished; train loss {total_loss}, validation loss TODO")

	def feed_forward(self, data, data_idx):
		# iterate through all layers
		for layer_idx, layer in enumerate(self.layers):
			# if the current layer is the first layer, set the 
			# lhs activations to input entry
			if layer.type == "input" :
				inputs = []
				for model in self.enum_models:
					feature_name = model["name"]
					feature_value = data.get_feature(feature_name)
					inputs.append([feature_value])
				layer.lhs_activation = np.array(inputs)
			else :
				prev_rhs = self.layers[layer_idx - 1].rhs_activation
				if prev_rhs is None:
					raise ValueError("Fill forward previous rhs is none")
				layer.lhs_activation = prev_rhs
			
			# run activation function, this would populate the current layer.rhs
			# logging.info(f".........run_activation layer {layer_idx} @ {data.get_feature('Id')} .......")
			# logging.info(f"\n\tweight\n{layer.weights}\n\tlhs\n\t{layer.lhs_activation}\n\tbias\n{layer.bias}")
			layer.run_activation()
			# if layer.type == "output":
			# 	logging.info(f"rhs\n{layer.rhs_activation}, truth : {self.train_truth[data_idx]}")
			# else:
			# 	logging.info(f"rhs\n{layer.rhs_activation}")

	
		last_layer_activation = self.layers[-1].rhs_activation
		return last_layer_activation

	def backprop(self, last_layer_entropy, truth, last_layer_activation):
		# set curr_loss at last layer and start backprop
		curr_loss = last_layer_entropy

		for idx, layer in enumerate(reversed(self.layers)):
			# TODO: remove test code
			# if len(self.layers) - 1 - idx != 3:
			# 	continue
			# logging.info(f".........run backprop layer {len(self.layers) - 1 - idx}.......")
			weight_deriv_matrix = np.empty(layer.weights.shape)
			bias_deriv_matrix = np.empty(layer.bias.shape)
			layer.rhs_loss = curr_loss.copy()
			next_loss = [0] * len(layer.lhs_activation)
			# NOTE: row_idx corresponds to rhs nodes, and col idx corresponds to lhs nodes.
			for row_idx, row in enumerate(layer.weights):
				for col_idx, col in enumerate(row):
					# logging.info(f"processing [{row_idx}][{col_idx}] {layer.lhs_activation}")
					loss_type = self.output_loss_type
					activation_type = layer.activation_fn
					lhs_activation = layer.lhs_activation[col_idx][0]
					# need to add a pseudo activation for input layer here
					# because ft_math assumes all input is validated activation
					# but our csv had input 0, which is not valid to do reverse softmax.
					# if layer.type == "input":
					# 	if activation_type == "sigmoid":
					# 		lhs_activation = ft_math.sigmoid(lhs_activation)
					# 	elif activation_type == "softplus":
					# 		lhs_activation = ft_math.softplus(lhs_activation)
					# 	else:
					# 		raise ValueError("Invalid activation type for input layer found in backprop")

					prev_softmax_value = None
					if layer.last_softmax_input is not None:
						prev_softmax_value = layer.last_softmax_input[row_idx][0]
					weight = col

					# logging.info(f"procesising {entry.get_feature('Id')} [{row_idx}][{col_idx}] = {col}.")
					if layer.type == "output":
						predicted = last_layer_activation[row_idx][0]
						true_value = truth[row_idx][0]
						weight_derivative = ft_math.dcost_dweight_output(
							loss_type,
							activation_type,
							predicted,
							true_value,
							lhs_activation,
							prev_softmax_value,
						)
						bias_derivative = ft_math.dcost_dbias_output(
							loss_type,
							activation_type,
							predicted,
							true_value,
							lhs_activation,
							prev_softmax_value,
						)
						next_loss_derivative = ft_math.dcost_dalhs_output(
							loss_type,
							activation_type,
							predicted,
							true_value,
							lhs_activation,
							weight,
							prev_softmax_value,
						)
						# logging.info(f"out - [{row_idx}][{col_idx}] weight_derivative {weight_derivative} @ {layer.weights[row_idx][col_idx]}, bias_derivative {bias_derivative}, next_loss_derivative {next_loss_derivative}, pred {predicted}, true {true_value}")
						# logging.info(f"{idx} adding {weight_derivative} to [{row_idx}][{col_idx}]")
						weight_deriv_matrix[row_idx][col_idx] = weight_derivative				
						bias_deriv_matrix[row_idx][0] = bias_derivative
						next_loss[col_idx] += next_loss_derivative
						# logging.info(f"......{predicted} {true_value} - [{row_idx}][{col_idx}]......")
					else:
						rhs_deriv_cost = layer.rhs_loss[row_idx]
						# logging.info(f"rhs_deriv_cost {rhs_deriv_cost}")
						# if idx == 3:
						# 	logging.info(f"{idx} act: {activation_type}, lhs: {lhs_activation}, rhs_cost: {rhs_deriv_cost}, prev_sm: {prev_softmax_value}")
						weight_derivative = ft_math.dcost_dweight_hidden(
							activation_type,
							lhs_activation,
							rhs_deriv_cost,
							prev_softmax_value
						)
						bias_derivative = ft_math.dcost_dbias_hidden(
							activation_type,
							lhs_activation,
							rhs_deriv_cost,
							prev_softmax_value
						)
						next_loss_derivative = ft_math.dcost_dalhs_hidden(
							activation_type,
							lhs_activation,
							rhs_deriv_cost,
							weight,
							prev_softmax_value
						)
						# logging.info(f"\n\tlhs idx {col_idx}\n\trhs idx {row_idx} \n\trhs_deriv_cost {rhs_deriv_cost}\n\tlhs_activation {lhs_activation}\n\tprev_softmax_value {prev_softmax_value}\n\tweight_derivative {weight_derivative}\n\tbias_derivative {bias_derivative}")
						# logging.info(f"hidden - [{row_idx}][{col_idx}] weight_derivative {weight_derivative} @ {layer.weights[row_idx][col_idx]}, bias_derivative {bias_derivative}, next_loss_derivative {next_loss_derivative}, pred {predicted}, true {true_value}")
						# if idx == 3:
						# logging.info(f"{idx} adding {weight_derivative} to [{row_idx}][{col_idx}]")
						weight_deriv_matrix[row_idx][col_idx] = weight_derivative				
						bias_deriv_matrix[row_idx][0] = bias_derivative
						next_loss[col_idx] += next_loss_derivative
			
			curr_loss = next_loss
			layer.pending_weights_derivatives += weight_deriv_matrix
			# logging.info(f"{len(self.layers) - 1 - idx} bp layer.weight_deriv_matrix \n{weight_deriv_matrix}")
			# logging.info(f"{len(self.layers) - 1 - idx} bp next_loss {next_loss}")
			# logging.info(f"{len(self.layers) - 1 - idx} bp layer.pending_weights_derivatives after add \n{layer.pending_weights_derivatives}")
			layer.pending_bias_derivatives += bias_deriv_matrix

	# feeds forward and backpropagates for 1 example, appending the required changes in the
	# layers themselves
	def feed_forward_and_backprop(self, entry, truth, data_idx):
		# run feed forward as usual
		last_layer_activation = self.feed_forward(entry, data_idx)

		# NOTE: HARDCODED CROSS ENTROPY
		# NOTE: sometimes activation is 1 here
		# logging.info(f"activation {ft_math.single_columADD this n_to_scalar(last_layer_activation)} truth {ft_math.single_column_to_scalar(truth)}")
		last_layer_error = None
		if self.output_loss_type == "binaryCrossEntropy":
			last_layer_error = ft_math.binary_cross_entropy(
						ft_math.single_column_to_scalar(last_layer_activation),
						ft_math.single_column_to_scalar(truth)
						)
		elif self.output_loss_type == "MSE":
			last_layer_error = ft_math.squared_err(
						ft_math.single_column_to_scalar(last_layer_activation),
						ft_math.single_column_to_scalar(truth)
						)
		else:
			raise ValueError("Invalid loss detected in feed_forward_and_backprop")
		# logging.info(f"truth {ft_math.single_column_to_scalar(truth)}")
		# logging.info(f"raw entropy for entry {entry.get_feature('Id')} : {last_layer_entropy}")

		# run back propagation to get derivatives for all weights and biases
		self.backprop(last_layer_error, truth, last_layer_activation)
		return last_layer_error

	def apply_derivatives_reset_cache(self):
		for idx, layer in enumerate(self.layers):
			# TODO: remove testing code
			# if idx != 3:
			# 	continue
			if layer.weights.shape != layer.pending_weights_derivatives.shape:
				raise ValueError("Weight derivatives dimensions dot not match value dimensions")
			if layer.bias.shape != layer.pending_bias_derivatives.shape:
				raise ValueError("Bias derivatives dimensions dot not match value dimensions")
			
			# apply changes to weights
			# logging.info(f"{idx} layer.weights before \n{layer.weights}")
			# logging.info(f"{idx} layer.pending_weights_derivatives \n{layer.pending_weights_derivatives}")
			stepsize_matrix = layer.pending_weights_derivatives * self.learning_rate
			layer.weights = layer.weights - stepsize_matrix
			logging.info(f"{idx} layer.weights stepsizes \n{stepsize_matrix}")
			# logging.info(f"{idx} layer.weights now \n{layer.weights}")
			
			# TODO ADD this back in
			# apply changes to biases
			# stepsize_matrix = layer.pending_bias_derivatives * self.learning_rate
			# layer.bias = layer.bias - stepsize_matrix

			# clear cache matrix
			layer.pending_weights_derivatives = np.zeros(layer.weights.shape)
			layer.pending_bias_derivatives = np.zeros(layer.bias.shape)