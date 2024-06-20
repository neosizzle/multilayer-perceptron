import sys
import os

sys.path.insert(0, os.path.abspath('ft_model'))

import logging
import time
import numpy as np
import ft_math

# reporter class to record and retreive historics
# events are logged in the following format:
# [loss, accuracy, recall]
# 1 row represents 1 event. The total historic may contain many rows
class Ft_reporter:
	def __init__(self,
				historic_path,
				historic_name
			  ):
		self.historic_path = historic_path
		self.historic_name = historic_name
		self.train_events = None
		self.test_events = None

	def report_event(self, event_type, loss, acc, recall):
		new_event = np.array([[loss, acc, recall]])

		if event_type == "train":
			if self.train_events is None :
				self.train_events = new_event
			else :
				self.train_events = np.append(self.train_events, new_event, axis=0)
		elif event_type == "test":
			if self.test_events is None :
				self.test_events = new_event
			else :
				self.test_events = np.append(self.test_events, new_event, axis=0)

	def generate_report(self):
		logging.debug(f"train_events {self.train_events}")
		logging.debug(f"test_events {self.test_events}")
		logging.debug(f"historic path {self.historic_path}")

		if not os.path.isdir(self.historic_path):
			os.mkdir(self.historic_path)
	
		name = round(time.time() * 1000)
		if self.historic_name != "" :
			name = self.historic_name
		np.save(f"{self.historic_path}/{name}_train.npy", self.train_events)
		np.save(f"{self.historic_path}/{name}_test.npy", self.test_events)