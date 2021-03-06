from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from utils.preprocessing import Preprocessing

preprocessing = Preprocessing()

input_file_name = "../../anomaly_detection/capture20110818-2.binetflow"
# input_file_name = "../../anomaly_detection/samples/small_sample.csv"

print("[" + datetime.now().__str__() + "]: " + 'Script starting')

cols = [
        "Dur",
        "Proto",
        "Sport",
        "Dir",
        "Dport",
        "State",
        "TotPkts",
        "TotBytes",
        "SrcBytes"
]

cols_with_labels = cols[:]
cols_with_labels.append('attack')

data = pd.read_csv(input_file_name)

preprocessing.filter_by_column(data, "Dir", ["<?>", "<->", "<-"])
preprocessing.transform_labels(data)
# preprocessing.transform_non_numerical_column(data, "Dport")
# preprocessing.transform_non_numerical_column(data, "Sport")
# preprocessing.transform_non_numerical_column(data, "Proto")
# preprocessing.transform_non_numerical_column(data, "Dir")
preprocessing.transform_non_numerical_column(data, "State")

preprocessing.normalize_columns(data, cols)
data = preprocessing.filter_by_column(data, "attack", [1])


y = data['attack']          # Split off classifications
X = data[cols] # Split off features

# Select features to include in the plot
plot_feat = cols[:]


print("[" + datetime.now().__str__() + "]: " + 'Drawing starts')

# Perform parallel coordinate plot
X.plot.hist()
fig = plt.gcf()
fig.set_size_inches(19.2, 10.8)
fig.savefig("capture20110818-2.binetflow-boxplot-anomalies_only.png", dpi=100, bbox_inches='tight')

print("[" + datetime.now().__str__() + "]: " + 'Drawing ended')
