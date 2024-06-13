import math
import numpy as np

# np.seterr(divide='ignore')

def mean(values) :
	return sum(values) / len(values)

def sigmoid(z) :
	# print(z)
	return 1 / (1 + np.exp(z * -1))

def sigmoid_reverse(z) :
	return np.log(z / (1 - z))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

def linear(m, x, c):
	return (x * m) + c

def softplus(x):
	return np.log(1 + np.exp(x))

def softplus_reverse(x):
	return np.log(np.exp(x) - 1)

def softplus_deriv(x):
	return sigmoid(x)

def softmax(values):
	res = []
	
	total = 0
	for value in values:
		total += np.exp(value)

	for value in values:
		res.append(np.exp(value) / total)

	return res

def softmax_deriv(x):
	return x * (1 - x)

# https://math.stackexchange.com/questions/2786600/invert-the-softmax-function
# def softmax_reverse(values):
# 	res = []
# 	values_product = math.prod(values)
# 	c = ((1 - math.log(values_product)) / len(values))
# 	for value in values:
# 		res.append(math.log(value) + c)
# 	return res

# binary cross entropy loss function. more sensitive than MSE
def binary_cross_entropy(softmaxed_values, true_values):
	return true_values * np.log(softmaxed_values) + (1 - true_values) * np.log(1 - softmaxed_values)

def squared_err(softmaxed_values, true_values):
	result = np.empty(softmaxed_values.shape)
	for row_idx, row in enumerate(softmaxed_values):
		for col_idx, col in enumerate(row):
			result[row_idx][col_idx] = math.pow(true_values[row_idx][col_idx] - col, 2)
	return result

def mean_squared_err(softmaxed_values, true_values):
	return squared_err(softmaxed_values, true_values) / len(true_values)

def mean_sqaured_err_deriv(predicted, actual):
	return -2 * (predicted - actual)

# https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
# this is the derivative of binary cross -> softmax with respect to its input 
def dcost_dz_output_np(predicted, actual):
	return predicted - actual

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# partial derivation of cost function loss against weight of output layer
# dcost / dw = dz/dw (lhs activation) * dc/dz
# w = weight param for last layer
# z = pre-activated input for node at last layer
def dcost_dw_output_np(dcost_dz, activation_lhs):
	return dcost_dz.dot(activation_lhs.T)

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# partial derivation of cost function against the bias of a hidden layer
# dc/dw = dc/dz * dz/db -> 1
def dcost_db_output_np(dcost_dz):
	return np.sum(dcost_dz * 1)

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# partial derivation of cost function against the unactivated output of a hidden layer
# dcost/dz hidden = dcost/darhs = (dz/dalhs * dcost/dznext = next weight) * da/dz (derive sigmoid)
def dcost_dz_hidden_np(weight, next_dcost_dz, lhs_activation):
	return weight.T.dot(next_dcost_dz) * sigmoid_deriv(sigmoid_reverse(lhs_activation))

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# partial derivation of cost function against the weight of a hidden layer
# dc/dw = dc/dz * dz/dw -> lhs activation
def dcost_dw_hidden_np(dcost_dz, activation_lhs):
	return dcost_dz.dot(activation_lhs.T)

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# partial derivation of cost function against the bias of a hidden layer
# dc/dw = dc/dz * dz/db -> 1
def dcost_db_hidden_np(dcost_dz):
	return np.sum(dcost_dz * 1)

def get_accuracy(predicted, actual):
	if predicted.shape != actual.shape:
		raise ValueError("predicted and actual shape is not equal")
	total_elements = predicted.size
	correct_elements = 0
	for row_idx, row in enumerate(predicted):
		for col_idx, col in enumerate(row):
			actual_elem = actual[row_idx][col_idx]
			if actual_elem == 1 and col >= 0.5:
				correct_elements += 1
			if actual_elem == 0 and col < 0.5:
				correct_elements += 1
	return correct_elements / total_elements

###### NUMPY UTILS #######
def single_column_to_scalar(np_arr):
	res = []
	if np_arr.shape[1] != 1:
		raise ValueError(f"Incorrent dimensions to convert to scalar {np_arr.shape}")

	for i in np_arr:
		res.append(i[0])
	return res
	
def scalar_to_single_column(arr):
	res = []
	for i in arr:
		res.append([i])
	return np.array(res)