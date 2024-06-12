import sys
import os
sys.path.insert(0, os.path.abspath('ft_model'))

import random
import csv
import ft_preprocess

def write_to_csv(train, destination, name):
	with open(f"{destination}{name}", "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for data in train:
			writer.writerow(data.to_csv_row())

def divide_with_random_idx(data, indices):
	data_one = []
	data_two = []

	for (index, data) in enumerate(data):
		if index in indices:
			data_one.append(data)
		else:
			data_two.append(data)
	return (data_one, data_two)

def seperate_and_write_data(raw_data, destination) :
	mal_entries = [x for x in raw_data if x.get_feature("Diagnosis") == "M"]
	ben_entries = [x for x in raw_data if x.get_feature("Diagnosis") == "B"]
	mal_count_new = len(mal_entries) // 2
	ben_count_new = len(ben_entries) // 2
	mal_indices = random.sample(range(0, len(mal_entries)), mal_count_new)
	ben_indices = random.sample(range(0, len(ben_entries)), ben_count_new)

	(training_mal, test_mal) = divide_with_random_idx(mal_entries, mal_indices)
	(training_ben, test_ben) = divide_with_random_idx(ben_entries, ben_indices)

	training_set = training_mal + training_ben
	test_set = test_mal + test_ben

	random.shuffle(training_set)
	random.shuffle(test_set)

	write_to_csv(training_set, destination, "data_train.csv")
	write_to_csv(test_set, destination, "data_test.csv")

def main():
	if len(sys.argv) != 3:
		print("Usage: python seperate_data.py [filename] [datasets_destination]")
		return
	raw_data = ft_preprocess.read_csv(sys.argv[1])
	seperate_and_write_data(raw_data, sys.argv[2])
	print("OK")

main()