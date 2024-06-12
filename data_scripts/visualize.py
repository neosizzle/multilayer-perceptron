import sys
import os
sys.path.insert(0, os.path.abspath('ft_model'))

import ft_preprocess
import ft_model
import csv
import matplotlib.pyplot as plt
import numpy as np

ENUMERABLE_FEATURE_OFFSET = 2
ENUMERABLE_FEATURE_NUM = 32 - ENUMERABLE_FEATURE_OFFSET
CLASS_FEATURE = "Diagnosis"

def sum(data) :
	total = 0
	for val in data:
		total += val
	return total

def mean(values) :
	return sum(values) / len(values)

def count(values) :
	return len(values)

def stddev(data, ddof=0):
    mean_data = mean(data)

    squared_diffs = [(x - mean_data) ** 2 for x in data]

    # Calculate the average of the squared differences
    variance = sum(squared_diffs) / (len(squared_diffs) - ddof)

    stddev = variance ** 0.5
    return stddev

def min(data):
	min = sys.maxsize
	for val in data :
		if val < min :
			min = val
	return min

def max(data):
	max = -sys.maxsize - 1
	for val in data :
		if val > max :
			max = val
	return max

# https://www.dummies.com/article/academics-the-arts/math/statistics/how-to-calculate-percentiles-in-statistics-169783/
def percentile(data, percent):
	sorted_data = sorted(data)
	total_len = len(data)
	percentile_idx = percent * total_len
	
	# determine if number is whole number
	if percentile_idx == int(percentile_idx) :
		data_0 = sorted_data[int(percentile_idx)]
		data_1 = sorted_data[int(percentile_idx) + 1]
		return (data_0 + data_1) / 2
	else:
		return sorted_data[int(percentile_idx)]

def generate_historgram(data) :
	# populate bins -  hardcoded as length 10
	bin_len = 10
	bins = []
	sorted_data = sorted(data)
	step = (sorted_data[-1] - sorted_data[0]) / bin_len
	curr_bin = sorted_data[0]

	for i in range(bin_len + 1) :
		bins.append((curr_bin + (i * step)))

	hist = []
	for i in range(len(bins) - 1):
		min = bins[i]
		max = bins[i + 1]
		num_elements = len(list(filter(lambda x: x >= min and x < max, sorted_data)))
		hist.append(num_elements)
	
	# fix for last element
	max = bins[-1]
	min = bins[-2]
	hist[-1] = len(list(filter(lambda x: x == max or x > min, sorted_data)))

	return (hist, bins)

def get_centers(data) :
	res = []
	for i in range(len(data) - 1):
		min = data[i]
		max = data[i + 1]
		res.append((min + max) / 2)
	return res

# match models and return appropriate type
def match_types(data, type) :
	if type == "float":
		return float(data)
	if type == "int":
		return int(data)
	return data

def get_house_color(house) :
	if house == "B":
		return "blue"
	if house == "M":
		return "red"

# read and parse csv
# [[indices], [ft_model.DIAGNOSIS], [name] ... ]
def read_csv(filepath) :
	res = []
	for type in ft_model.DATA_MODEL :
		res.append([])
	file=open(filepath, "r")
	reader = csv.reader(file)
	skip_first_line = True
	for line in reader:
		t=line

		if skip_first_line:
			skip_first_line = False
			continue

		data_row = []
		for model in ft_model.DATA_MODEL:
			idx = model["idx"]
			type = model["type"]
			try:
				data = match_types(line[idx], type)
				data_row.append(data)
			except:
				continue

		if len(data_row) == len(ft_model.DATA_MODEL):
			for idx, col in enumerate(data_row):
				res[idx].append(col)				
	return res

def plot_scatter(data, ax):
	features_matrix = []
	for i in range(ENUMERABLE_FEATURE_NUM):
		single_feature_row = []
		for j in range(ENUMERABLE_FEATURE_NUM):
			single_feature_row.append([[], []]) # [x_values, y_values]
		features_matrix.append(single_feature_row)
	
	for entry in data:
		for x in ft_model.DATA_MODEL:
			if x['idx'] < ENUMERABLE_FEATURE_OFFSET:
				continue
			for y in ft_model.DATA_MODEL:
				if y['idx'] < ENUMERABLE_FEATURE_OFFSET:
					continue
				actual_y_idx = y['idx'] - ENUMERABLE_FEATURE_OFFSET
				feature_y = entry.get_feature(y['name'])
				actual_x_idx = x['idx'] - ENUMERABLE_FEATURE_OFFSET
				feature_x = entry.get_feature(x['name'])
				class_entry = entry.get_feature(CLASS_FEATURE)
				features_matrix[actual_y_idx][actual_x_idx][0].append((feature_x, class_entry))
				features_matrix[actual_y_idx][actual_x_idx][1].append((feature_y, class_entry))
		# break

	for (row_idx, row_data) in enumerate(features_matrix):
		for (col_idx, col_data) in enumerate(row_data):
			y_feature_name = ft_model.DATA_MODEL[row_idx + ENUMERABLE_FEATURE_OFFSET]['name']
			x_feature_name = ft_model.DATA_MODEL[col_idx + ENUMERABLE_FEATURE_OFFSET]['name']
			print(f"{y_feature_name} vs {x_feature_name}") 
			x_values = list(map(lambda x: x[0], col_data[0]))
			y_values = list(map(lambda x: x[0], col_data[1]))
			
			# x and y should have same color here
			colors = list(map(lambda x: get_house_color(x[1]), col_data[0]))

			# print(colors)
			ax[row_idx][col_idx].scatter(x_values, y_values, c=colors, alpha=0.6)
			ax[row_idx][col_idx].set_xlabel(x_feature_name)
			ax[row_idx][col_idx].set_ylabel(y_feature_name)


def main():
	if len(sys.argv) != 2:
		print("Usage: python visualize.py [filename]")
		return
	
	# init matplotlib
	plt.style.use('_mpl-gallery')
	fig, ax = plt.subplots(ENUMERABLE_FEATURE_NUM, ENUMERABLE_FEATURE_NUM, figsize=(105, 105))
	plt.tight_layout(pad=5)


	data = ft_preprocess.read_csv(sys.argv[1])
	plot_scatter(data, ax)
	# print(data)

	# export matplotlib as png
	plt.savefig(f"{sys.argv[0]}.png")
main()