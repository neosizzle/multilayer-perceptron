import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import argparse
import coloredlogs, logging
import traceback
import time

import ft_preprocess
import ft_perception
import ft_reporter

def get_args():
	script_dir = os.path.dirname(os.path.abspath(__file__))

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-l', '--layer', help="Speficy number of nodes for each layer", nargs='*', type=int, default=[10, 10, 10])
	parser.add_argument('-e', '--epochs', help="Speficy number of Epochs to run", type=int, default=10)
	parser.add_argument('-L', '--loss', help="Speficy type of loss function used at output layer", type=str, default='binaryCrossEntropy', choices=['binaryCrossEntropy', 'MSE'])
	parser.add_argument('-H', '--historic_path', help="Speficy the path of the folder to store historics of this training session", type=str, default=f'{script_dir}/historics/{round(time.time() * 1000)}')
	parser.add_argument('-Hn', '--historic_name', help="Speficy the name of the historic files of this training session", type=str, default=f'')
	parser.add_argument('-o', '--output', help="Speficy the path of the folder to store weights of this training session", type=str, default=f'{script_dir}/weights/')
	parser.add_argument('-b', '--batch_size', help="Speficy number of nodes in 1 batch for mini batch GD", type=int, default=-1)
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
	ft_preprocess.normalize_wtith_weights(raw_data_test, min_max_weights_train)
	logging.info("Train and test dataset normalized")
	logging.debug(f"min_max_weights_train : {min_max_weights_train}")
	
	mean_and_stddev_train = ft_preprocess.standardize_features(raw_data_train)
	ft_preprocess.standardize_with_weights(raw_data_test, mean_and_stddev_train)
	logging.info("Train and test dataset standardize")
	logging.debug(f"mean_and_stddev_train : {mean_and_stddev_train}")

	reporter = ft_reporter.Ft_reporter(args.historic_path, args.historic_name)

	perceptron = ft_perception.Ft_perceptron(
		args.layer,
		args.epochs,
		args.loss,
		args.batch_size,
		args.learning_rate,
		args.output,
		min_max_weights_train,
		mean_and_stddev_train,
		raw_data_train,
		raw_data_test,
		reporter
	)

	try:
		perceptron.begin_train()

	except Exception as e:
		logging.error(f"begin_train: {e}")
		# logging.error(traceback.format_exc())
		return
main()