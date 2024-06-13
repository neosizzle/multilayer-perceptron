import sys
import os
import traceback

sys.path.insert(0, os.path.abspath('ft_model'))

import argparse
import coloredlogs, logging
import ft_preprocess
import ft_perception

import math
import ft_math

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	# TODO: make sure hidden laters >= 3
	parser.add_argument('-l', '--layer', help="Speficy number of nodes for each layer", nargs='*', type=int, default=[13, 13, 13])
	parser.add_argument('-e', '--epochs', help="Speficy number of Epochs to run", type=int, default=10)
	# parser.add_argument('-l', '--layer', help="Speficy number of nodes for each layer", nargs='*', type=int, default=[24, 24, 24])
	# parser.add_argument('-e', '--epochs', help="Speficy number of Epochs to run", type=int, default=84)
	parser.add_argument('-L', '--loss', help="Speficy type of loss function used at output layer", type=str, default='binaryCrossEntropy', choices=['binaryCrossEntropy', 'MSE'])
	parser.add_argument('-b', '--batch_size', help="Speficy number of nodes in 1 batch for mini batch GD", type=int, default=8)
	parser.add_argument('-a', '--learning_rate', help="Speficy the learning rate for GD", type=float, default=0.0314)
	parser.add_argument('train_dataset', nargs=1)
	parser.add_argument('test_dataset', nargs=1)
	return parser.parse_args()

def main():
	# get args and initialize logging
	coloredlogs.install()
	args = get_args()
	if args.verbose:
		coloredlogs.set_level(logging.DEBUG)
	
	raw_data_train = []
	raw_data_test = []
	try:
		raw_data_train = ft_preprocess.read_csv(args.train_dataset[0])
		raw_data_test = ft_preprocess.read_csv(args.test_dataset[0])
		logging.info("Train and test dataset read")
	except Exception as e:
		logging.error(f"read_csv: {e}")
		logging.error(traceback.format_exc())
		return
	min_max_weights_train = ft_preprocess.normalize_features(raw_data_train)
	min_max_weights_test = ft_preprocess.normalize_features(raw_data_test)
	logging.info("Train and test dataset normalized")
	logging.debug(f"min_max_weights_train : {min_max_weights_train}")
	logging.debug(f"min_max_weights_test : {min_max_weights_test}")
	
	mean_and_stddev_train = ft_preprocess.standardize_features(raw_data_train)
	mean_and_stddev_test = ft_preprocess.standardize_features(raw_data_test)
	logging.info("Train and test dataset standatdized")
	logging.debug(f"mean_and_stddev_train : {mean_and_stddev_train}")
	logging.debug(f"mean_and_stddev_test : {mean_and_stddev_test}")


	perceptron = ft_perception.Ft_perceptron(
		args.layer,
		args.epochs,
		args.loss,
		args.batch_size,
		args.learning_rate,
		raw_data_train,
		raw_data_test
	)

	try:
		# i = 0
		# predicted = 0.99
		# actual = 0
		# while i < 100 :
		# 	loss = ft_math.binary_cross_entropy([predicted], [actual])
		# 	# 
		# 	logging.info(f"pred {predicted} actual {actual} loss: {loss} deriv {ft_math.binary_cross_entropy_deriv_activation(predicted, actual)}")
		# 	predicted -= 0.01
		# 	i += 1
		perceptron.begin_train()
		# perceptron.begin_train_alt()

	except Exception as e:
		logging.error(f"begin_train: {e}")
		logging.error(traceback.format_exc())
		return
main()