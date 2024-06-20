import os

class Ft_event:
	def __init__(self, name, loss, accuracy, recall):
		self.name = name
		self.losses = loss
		self.accs = accuracy
		self.recalls = recall