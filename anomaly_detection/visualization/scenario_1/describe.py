from utils.preprocessing import Preprocessing

preprocessing = Preprocessing()

original_dataset = preprocessing.read_file("../../../datasets/scenario_1.binetflow")

print(original_dataset.describe())