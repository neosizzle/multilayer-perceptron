# DATA_MODEL = [
# 	{"name": "Id", "idx": 0, "type": "int"},
# 	{"name": "Diagnosis", "idx": 1, "type": "string"},
# 	{"name": "Radius_N1", "idx": 2, "type": "float"},
# 	{"name": "Texture_N1", "idx": 3, "type": "float"},
# 	{"name": "Perimeter_N1", "idx": 4, "type": "float"},
# 	{"name": "Area_N1", "idx": 5, "type": "float"},
# 	{"name": "Smoothness_N1", "idx": 6, "type": "float"},
# 	{"name": "Compactness_N1", "idx": 7, "type": "float"},
# 	{"name": "Concavity_N1", "idx": 8, "type": "float"},
# 	{"name": "Concave points_N1", "idx": 9, "type": "float"},
# 	{"name": "Symmetry_N1", "idx": 10, "type": "float"},
# 	{"name": "Fractal dimension_N1", "idx": 11, "type": "float"},
	
# 	{"name": "Radius_N2", "idx": 12, "type": "float"},
# 	{"name": "Texture_N2", "idx": 13, "type": "float"},
# 	{"name": "Perimeter_N2", "idx": 14, "type": "float"},
# 	{"name": "Area_N2", "idx": 15, "type": "float"},
# 	{"name": "Smoothness_N2", "idx": 16, "type": "float"},
# 	{"name": "Compactness_N2", "idx": 17, "type": "float"},
# 	{"name": "Concavity_N2", "idx": 18, "type": "float"},
# 	{"name": "Concave points_N2", "idx": 19, "type": "float"},
# 	{"name": "Symmetry_N2", "idx": 20, "type": "float"},
# 	{"name": "Fractal dimension_N2", "idx": 21, "type": "float"},

# 	{"name": "Radius_N3", "idx": 22, "type": "float"},
# 	{"name": "Texture_N3", "idx": 23, "type": "float"},
# 	{"name": "Perimeter_N3", "idx": 24, "type": "float"},
# 	{"name": "Area_N3", "idx": 25, "type": "float"},
# 	{"name": "Smoothness_N3", "idx": 26, "type": "float"},
# 	{"name": "Compactness_N3", "idx": 27, "type": "float"},
# 	{"name": "Concavity_N3", "idx": 28, "type": "float"},
# 	{"name": "Concave points_N3", "idx": 29, "type": "float"},
# 	{"name": "Symmetry_N3", "idx": 30, "type": "float"},
# 	{"name": "Fractal dimension_N3", "idx": 31, "type": "float"},
# ]

DATA_MODEL = [
	{"name": "Id", "idx": 0, "type": "int"},
	{"name": "Diagnosis", "idx": 1, "type": "string"},
	{"name": "Radius_N1", "idx": 2, "type": "float"},
	{"name": "Texture_N1", "idx": 3, "type": "float"},
]

TRUTH_MODEL = {"name": "Diagnosis", "idx": 1, "type": "string"}

DIAGNOSIS = [
	"B",
	"M",
]

def get_enumerable_models():
	return list(filter(lambda x: x["type"] == "float", DATA_MODEL))

# match models and return appropriate type from string data
def match_types(data, type) :
	if data == '':
		return None
	if type == "float":
		return float(data)
	if type == "int":
		return int(data)
	return data


class Model:
	def __init__(self):
		self.features = {}

	def __repr__(self):
		return f"{self.features.__repr__()}\n"

	def __str__(self):
		return self.features.__str__()

	def set_feature(self, feature, value):
		self.features[feature] = value

	def get_feature(self, feature):
		try:
			return self.features[feature]
		except:
			return None

	def get_all_features(self):
		return self.features
	
	def to_csv_row(self):
		res = []
		for feature in DATA_MODEL:
			value = self.get_feature(feature["name"])
			res.append(str(value))
		return res