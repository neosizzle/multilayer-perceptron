import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.layout_engine as layout

import ft_event

COLORS = [
	"red",
	"cyan",
	"blue",
	"green",
	"pink",
	"orange",
	"yellow",
	"lime",
]

class Ft_plotter:
	def __init__(self, historic_path):
		if not (os.path.exists(historic_path) and os.path.isdir(historic_path)) :
			raise ValueError("Invalid path provided during init")

		plt.style.use('_mpl-gallery')
		plt.rcParams['font.size'] = 12

		layout_engine = layout.TightLayoutEngine(pad=2.0)

		loss_fig = plt.figure("Loss")
		loss_fig.suptitle("Loss")
		loss_fig.set_layout_engine(layout=layout_engine)
		loss_fig.set_size_inches(20, 10)

		acc_fig = plt.figure("Accuracy")
		acc_fig.suptitle("Accuracy")
		acc_fig.set_layout_engine(layout=layout_engine)
		acc_fig.set_size_inches(20, 10)

		recall_fig = plt.figure("Recall")
		recall_fig.suptitle("Recall")
		recall_fig.set_layout_engine(layout=layout_engine)
		recall_fig.set_size_inches(20, 10)

		self.loss_fig = loss_fig
		self.acc_fig = acc_fig
		self.recall_fig = recall_fig
		self.historic_path = historic_path
		self.entry_table = {
			"train_events": [],
			"test_events": [],
		}

	def read_values(self):
		for entry in os.listdir(self.historic_path):
			file_path = os.path.join(self.historic_path, entry)
			if os.path.isfile(file_path):
				raw_data = np.load(file_path).T
				loss = raw_data[0]
				acc = raw_data[1]
				recall = raw_data[2]
				if file_path.endswith("_test.npy"):
					self.entry_table["test_events"].append(ft_event.Ft_event(file_path[:file_path.find("_test.npy")], loss, acc, recall))
				elif file_path.endswith("_train.npy"):
					self.entry_table["train_events"].append(ft_event.Ft_event(file_path[:file_path.find("_train.npy")], loss, acc, recall))
		self.entry_table["test_events"].sort(key=lambda s: s.name)
		self.entry_table["train_events"].sort(key=lambda s: s.name)

	def plot_subplots(self, figure):
		train_plot, test_plot = figure.subplots(1, 2)

		train_plot.set_title("Train")
		test_plot.set_title("Test")
		
		train_plot.set_xlabel("Epochs")
		train_plot.set_ylabel("Value")
		train_plot.set_ylim(0, 1.1)

		test_plot.set_xlabel("Epochs")
		test_plot.set_ylabel("Value")
		test_plot.set_ylim(0, 1.1)

		for idx, train_event in enumerate(self.entry_table["train_events"]):
			y_values = None

			if figure.get_suptitle() == "Loss":
				y_values = train_event.losses
			elif figure.get_suptitle() == "Accuracy":
				y_values = train_event.accs
			elif figure.get_suptitle() == "Recall":
				y_values = train_event.recalls
			x_values = list(range(len(y_values)))
			train_plot.plot(x_values, y_values, label=train_event.name, alpha=0.7, c=COLORS[idx])
		train_plot.legend(fontsize='large')

		for idx, test_event in enumerate(self.entry_table["test_events"]):
			y_values = None

			if figure.get_suptitle() == "Loss":
				y_values = test_event.losses
			elif figure.get_suptitle() == "Accuracy":
				y_values = test_event.accs
			elif figure.get_suptitle() == "Recall":
				y_values = test_event.recalls
			x_values = list(range(len(y_values)))
			test_plot.plot(x_values, y_values, label=test_event.name, alpha=0.7, c=COLORS[idx])
		test_plot.legend(fontsize='large')

	def plot_diagrams(self):
		self.plot_subplots(self.loss_fig)
		self.plot_subplots(self.recall_fig)
		self.plot_subplots(self.acc_fig)

		self.loss_fig.savefig(f"loss.png")
		self.recall_fig.savefig(f"recall.png")
		self.acc_fig.savefig(f"acc.png")