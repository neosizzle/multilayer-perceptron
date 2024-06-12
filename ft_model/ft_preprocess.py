import sys
import csv

import ft_model

def stddev(data, ddof=0):
    # Calculate the mean of the data
    mean_data = sum(data) / len(data)

    # Calculate squared differences for each data point and mean
    squared_diffs = [(x - mean_data) ** 2 for x in data]

    # Calculate the average of the squared differences
    variance = sum(squared_diffs) / (len(squared_diffs) - ddof)

    # Calculate the square root of the variance
    stddev = variance ** 0.5
    return stddev

def mean(set):
	data_len = len(set)
	total = 0
	for i in set :
		total += i
	return total / data_len


# read and parse csv
def read_csv(filepath) :
	res = []
	file=open(filepath, "r")
	reader = csv.reader(file)
	for line in reader:
		t=line

		entry = ft_model.Model()
		for column in ft_model.DATA_MODEL:
			index = column["idx"]
			feature = column["name"]
			data = ft_model.match_types(line[index], column["type"])
			entry.set_feature(feature, data)
		res.append(entry)
	return res

# given raw data from read_csv, normalize the faetures and return the min max weights
def normalize_features(raw_data):
	feature_arr = []
	min_max_arr = []

	for feature in ft_model.DATA_MODEL:
		feature_arr.append([])
		min_max_arr.append([])

	for entry in raw_data:
		for feature in ft_model.DATA_MODEL:
			feature_idx = feature["idx"]
			feature_name = feature["name"]
			feature_arr[feature_idx].append(entry.get_feature(feature_name))
	
	for (feat_arr_idx, features) in enumerate(feature_arr):
		if ft_model.DATA_MODEL[feat_arr_idx]["type"] != "float":
			min_max_arr[feat_arr_idx].append(None)
			min_max_arr[feat_arr_idx].append(None)
			continue
		min_data = min(features)
		max_data = max(features)
		
		for (value_idx, value) in enumerate(features):
			new_val = (value - min_data) / (max_data - min_data)
			features[value_idx] = new_val

		min_max_arr[feat_arr_idx].append(min_data)
		min_max_arr[feat_arr_idx].append(max_data)
	
	for (entry_idx, entry) in enumerate(raw_data):
		for feature in ft_model.DATA_MODEL:
			feature_idx = feature["idx"]
			feature_name = feature["name"]
			new_feat_value = feature_arr[feature_idx][entry_idx]
			entry.set_feature(feature_name, new_feat_value)

	return min_max_arr
	
# returns [means all features, standard deviations all features]
def standardize_features(raw_data):
	feature_arr = []
	mean_arr = []
	stddev_arr = []

	for feature in ft_model.DATA_MODEL:
		feature_arr.append([])
		mean_arr.append([])
		stddev_arr.append([])

	# get original features
	for entry in raw_data:
		for feature in ft_model.DATA_MODEL:
			feature_idx = feature["idx"]
			feature_name = feature["name"]
			feature_arr[feature_idx].append(entry.get_feature(feature_name))
	
	# populate mean and sttdev arr
	for feat_arr_idx, feature_values in enumerate(feature_arr):
		if ft_model.DATA_MODEL[feat_arr_idx]["type"] != "float":
			stddev_arr[feat_arr_idx] = None
			mean_arr[feat_arr_idx] = None
		else :
			mean_arr[feat_arr_idx] = mean(feature_values)
			stddev_arr[feat_arr_idx] = stddev(feature_values)
	
	# calculate new feature values
	for feat_idx, features in enumerate(feature_arr):
		if ft_model.DATA_MODEL[feat_idx]["type"] != "float":
			continue
		for entry_idx, entry_feature in enumerate(features):
			mean_val = mean_arr[feat_idx]
			dev = stddev_arr[feat_idx]
			feature_arr[feat_idx][entry_idx] = (entry_feature - mean_val) / dev

	# set feature values
	for (entry_idx, entry) in enumerate(raw_data):
		for feature in ft_model.DATA_MODEL:
			feature_idx = feature["idx"]
			feature_name = feature["name"]
			new_feat_value = feature_arr[feature_idx][entry_idx]
			entry.set_feature(feature_name, new_feat_value)

	return [mean_arr, stddev_arr]