import math
import numpy as np
import logging

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
	return math.log(1 + math.exp(x))

def softplus_reverse(x):
	return math.log(math.exp(x) - 1)

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

# https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
def softmax_deriv_2(predicted, actual):
	# print(f"predicted {predicted}")
	# print(f"actual {actual}")
	# print(f"minus {predicted - actual}")
	return predicted - actual

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
	results = []
	for (elem_idx, elem) in enumerate(true_values): 
		pred = softmaxed_values[elem_idx]
		if pred == 0 or pred == 1:
			logging.warn(f"Numerical overflow/underflow at value {pred}, might be overfitting.")
			if pred == 0:
				pred = 0.00001
			if pred == 1:
				pred = 0.99999
		results.append(elem * math.log(pred) + (1 - elem) * math.log(1 - pred))
	return results

def binary_cross_entropy_deriv_activation(predicted, actual):
	return  (actual / predicted - ((1 - actual) / (1 - predicted)))

def binary_cross_entropy_deriv_activation_np(predicted, actual):
	res = []

	for row in predicted:
		res.append([])

	for row_idx, row in enumerate(predicted):
		for col_idx, col in enumerate(row):
			pred_value = col
			actual_value = actual[row_idx][col_idx]
			if actual_value == 1:
				res[row_idx].append(-1 / pred_value)
			else:
				res[row_idx].append(1 / (1 - pred_value))
	return np.array(res)

# flip to positive when displaying
def binary_cross_entropy_np(pred, actual):
	res = []
	for idx, elem in enumerate(actual):
		if elem == 1:
			res.append(np.log(pred[idx]))
		else:
			res.append(np.log(1 - pred[idx]))
	return np.array(res)

def squared_err(softmaxed_values, true_values):
	results = []
	for (elem_idx, elem) in enumerate(true_values): 
		results.append(math.pow(elem - softmaxed_values[elem_idx], 2))
	return results

def mean_squared_err(softmaxed_values, true_values):
	return squared_err(softmaxed_values, true_values) / len(true_values)

def mean_sqaured_err_deriv(predicted, actual):
	return -2 * (predicted - actual)

# partial derivation of cost function loss against weight of output layer
# dcost / dw = dz/dw * da/dz * dc/da
# w = weight param for last layer
# a = activated value for node at last layer
# z = pre-activated input for node at last layer
def dcost_dweight_output(loss_type, activation_type, predicted, actual, lhs_activation, prev_softmax_value=None):
	activation_deriv = None
	if loss_type == "binaryCrossEntropy":
		activation_deriv = binary_cross_entropy_deriv_activation(predicted, actual)
	elif loss_type == "MSE":
		activation_deriv = mean_sqaured_err_deriv(predicted, actual)
	else:
		raise ValueError(f"Invalid loss_type, options are binaryCrossEntropy and MSE")

	da_dz = None
	dz_dw = lhs_activation
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dc_da = activation_deriv
	# print(f"| dc_da {dc_da} da_dz {da_dz} dz_dw {dz_dw}|")
	return dz_dw * da_dz * dc_da

def dcost_dw_output_np(predicted, actual, lhs_activation):
	# print(f"| predicted {predicted} actual {actual} lhs_activation {lhs_activation}| pre_Activate {pre_activate}")
	dc_da = binary_cross_entropy_deriv_activation_np(predicted, actual)
	da_dz = softmax_deriv_2(predicted, actual)
	dz_dw = lhs_activation
	# print(f"| dc_da {dc_da} da_dz {da_dz} dz_dw {dz_dw}|")
	return (dc_da.dot(da_dz.T)).dot(dz_dw)

# partial derivation of cost function loss against bias of output layer
# dcost/ dbias = dz/dbias * da/dz * dc/da
# bias = bias param for last layer
# a = activated value for node at last layer
# z = pre-activated input for node at last layer
def dcost_dbias_output(loss_type, activation_type, predicted, actual, lhs_activation, prev_softmax_value=None):
	activation_deriv = None
	if loss_type == "binaryCrossEntropy":
		activation_deriv = binary_cross_entropy_deriv_activation(predicted, actual)
	elif loss_type == "MSE":
		activation_deriv = mean_sqaured_err_deriv(predicted, actual)
	else:
		raise ValueError(f"Invalid loss_type, options are binaryCrossEntropy and MSE")

	dz_db = 1
	da_dz = None
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dc_da = activation_deriv

	return dz_db * da_dz * dc_da

# partial derivation of cost function loss against input activation of output layer
# dcost / da(lhs) = dz/da(lhs) * da/dz * dc/da
# a(lhs) = activation value for the previous layer
# a = activated value for node at last layer
# z = pre-activated input for node at last layer
def dcost_dalhs_output(loss_type, activation_type, predicted, actual, lhs_activation, weight, prev_softmax_value=None):
	activation_deriv = None
	if loss_type == "binaryCrossEntropy":
		activation_deriv = binary_cross_entropy_deriv_activation(predicted, actual)
	elif loss_type == "MSE":
		activation_deriv = mean_sqaured_err_deriv(predicted, actual)
	else:
		raise ValueError(f"Invalid loss_type, options are binaryCrossEntropy and MSE")

	dz_dalhs = weight
	da_dz = None
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dc_da = activation_deriv

	# print(f"dz_dalhs {dz_dalhs} | da_dz {da_dz} | dc_da {dc_da} | pred {predicted} | actual {actual}")
	return dz_dalhs * da_dz * dc_da

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
def dcost_dz_output_np(predicted, actual):
	return predicted - actual

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
def dcost_dw_output_np(dcost_dz, activation_lhs):
	return dcost_dz.dot(activation_lhs.T)

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
def dcost_db_output_np(dcost_dz):
	return dcost_dz * 1

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# dcost/dz hidden = dcost/darhs = (dz/dalhs * dcost/dznext = weight) * da/dz (derive sigmoid)
def dcost_dz_hidden_np(weight, next_dcost_dz, lhs_activation):
	return weight.T.dot(next_dcost_dz) * sigmoid_deriv(sigmoid_reverse(lhs_activation))

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# dc/dw = dc/dz * dz/dw -> lhs activation
def dcost_dw_hidden_np(dcost_dz, activation_lhs):
	return dcost_dz.dot(activation_lhs.T)

# https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a
# dc/dw = dc/dz * dz/db -> 1
def dcost_db_hidden_np(dcost_dz):
	return dcost_dz * 1

# partial derivation of cost function against the weight of a hidden layer
# dcost/dw(l) = dz(L) / dw(L) * da(L) / dz(L) * dcost / da(L)
# the dcost / da(L) or rhs_deriv_cost is obtained from the next layer via backprop.
def dcost_dweight_hidden(activation_type, lhs_activation, rhs_deriv_cost, prev_softmax_value=None):
	dz_dw = lhs_activation
	da_dz = None
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dcost_da = rhs_deriv_cost
	
	# print(f"dz_dw {dz_dw} | da_dz {da_dz} | dcost_da {dcost_da} {activation_type}")
	return dz_dw * da_dz * dcost_da

# partial derivation of cost function against the weight of a hidden layer
# dcost/dalhs(l) = dz(L) / dalhs(L) * da(L) / dz(L) * dcost / da(L)
# the dcost / da(L) or rhs_deriv_cost is obtained from the next layer via backprop.
def dcost_dalhs_hidden(activation_type, lhs_activation, rhs_deriv_cost, weight, prev_softmax_value=None):
	dz_alhs = weight
	da_dz = None
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dcost_da = rhs_deriv_cost

	return dz_alhs * da_dz * dcost_da

# partial derivation of cost function against the bias of a hidden layer
# dcost/dbias(l) = dz(L) / dbias(L) * da(L) / dz(L) * dcost / da(L)
# the dcost / da(L) or rhs_deriv_cost is obtained from the next layer via backprop.
def dcost_dbias_hidden(activation_type, lhs_activation, rhs_deriv_cost, prev_softmax_value=None):
	dz_dbias = 1
	da_dz = None
	if activation_type == "softplus":
		da_dz = softplus_deriv(softplus_reverse(lhs_activation))
	elif activation_type == "sigmoid":
		da_dz = sigmoid_deriv(sigmoid_reverse(lhs_activation))
	else :
		raise ValueError(f"Invalid activation_type, options are sigmoid, softmax and softplus")
	dcost_da = rhs_deriv_cost

	return dz_dbias * da_dz * dcost_da

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