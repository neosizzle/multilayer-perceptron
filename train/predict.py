import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import argparse
import coloredlogs, logging
import traceback
import numpy as np
import json

import ft_preprocess
import ft_perception
import ft_reporter
import ft_math
import math

INVALID = "INVALID"

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	# TODO: make sure hidden layers >= 3
	parser.add_argument('layer_weights', nargs=1)
	parser.add_argument('layer_bias', nargs=1)
	parser.add_argument('config', nargs=1)
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
	init_weights_list = []
	init_bias_list = []
	configuration = None
	try:
		raw_data_test = ft_preprocess.read_csv(args.test_dataset[0])

		loaded_weights = list(dict(np.ndenumerate(np.load(args.layer_weights[0], allow_pickle=True)['arr_0'])).values())[0]
		for weight in loaded_weights:
			init_weights_list.append(loaded_weights[weight])

		loaded_bias = list(dict(np.ndenumerate(np.load(args.layer_bias[0], allow_pickle=True)['arr_0'])).values())[0]
		for bias in loaded_bias:
			init_bias_list.append(loaded_bias[bias])

		with open(args.config[0]) as f:
			configuration = json.load(f)

		logging.info("test dataset and weights read")
	except Exception as e:
		logging.error(f"read_csv: {e}")
		logging.error(traceback.format_exc())
		return

	ft_preprocess.normalize_wtith_weights(raw_data_test, configuration['norm_weights'])
	logging.info("test dataset normalized")
	
	ft_preprocess.standardize_with_weights(raw_data_test, configuration['stand_weights'])
	logging.info("test dataset standardize")

	reporter = ft_reporter.Ft_reporter("", "")

	# logging.info(init_weights_list)

	perceptron = ft_perception.Ft_perceptron(
		configuration["hidden_layers"],
		0,
		configuration["loss_type"],
		INVALID,
		INVALID,
		INVALID,
		configuration['norm_weights'],
		configuration['stand_weights'],
		raw_data_train,
		raw_data_test,
		reporter,
		init_weights_list=init_weights_list,
		init_bias_list=init_bias_list,
	)

	try:
		perceptron.begin_predict()
	except Exception as e:
		logging.error(f"begin_predict: {e}")
		logging.error(traceback.format_exc())
		return
main()