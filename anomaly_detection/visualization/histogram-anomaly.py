from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.preprocessing import Preprocessing

preprocessing = Preprocessing()

input_file_name = "../../anomaly_detection/capture20110818-2.binetflow"
# input_file_name = "../../anomaly_detection/samples/small_sample.csv"

analyzed_parameter = 'SrcBytes'

print("[" + datetime.now().__str__() + "]: " + 'Script starting')

data = pd.read_csv(input_file_name)
preprocessing.transform_labels(data)

smallDataset = data[[analyzed_parameter, 'attack']]
# preprocessing.transform_non_numerical_column(smallDataset, 'State')

normal = smallDataset[:]
anomaly = smallDataset[:]

normal = preprocessing.filter_by_column(normal, "attack", [1])
anomaly = preprocessing.filter_by_column(anomaly, "attack", [1])

normal = normal[[analyzed_parameter]]
anomaly = anomaly[[analyzed_parameter]]

# y = data['attack']          # Split off classifications
X = smallDataset[analyzed_parameter] # Split off features

# Select features to include in the plot
plot_feat = list(X)


print("[" + datetime.now().__str__() + "]: " + 'Drawing starts')

# Perform parallel coordinate plot
# X.value_counts().plot(kind='barh', logx=True)

# fig, (normal_ax, anomaly_ax) = plt.subplots(2)
#

fig, (normal_ax_0, normal_ax_1,normal_ax_2) = plt.subplots(nrows=3, figsize=(6,12))
# normal = normal[(normal < 4000)]

sns.distplot(normal, kde=False, color="r", ax=normal_ax_0, bins=100)
sns.distplot((normal[(normal > 100) & (normal < 10000)].dropna(axis=1, how='all')), kde=False, color="r", ax=normal_ax_1, bins=100, hist_kws={"range": [100, 10000]})
sns.distplot((normal[(normal > 900) & (normal < 2500)].dropna(axis=1, how='all')), kde=False, color="r", ax=normal_ax_2, bins=50, hist_kws={"range": [900, 2500]})
# sns.distplot((normal[(normal > 1000) & (normal < 1100)].dropna(axis=1, how='all')), kde=False, color="r", ax=normal_ax_3, bins=50, hist_kws={"range": [1000, 1100]})
# sns.distplot((normal[(normal < 750)].dropna(axis=1, how='all')), kde=False, color="r", ax=normal_ax_4, bins=50, hist_kws={"range": [0, 750]})
# sns.distplot((normal[(normal < 100) & (normal > 50)].dropna(axis=1, how='all')), kde=False, color="r", ax=normal_ax_5, bins=50, hist_kws={"range": [50, 100]})
# sns.distplot(anomaly, kde=False, color="r", ax=anomaly_ax, bins=50)

# anomaly_ax.set_xscale("log")
# anomaly_ax.set_yscale("log")
#
normal_ax_0.set_xscale("log")
normal_ax_0.set_yscale("log")
normal_ax_0.set_xlabel("SrcBytes")
normal_ax_0.set_xlabel("SrcBytes")
normal_ax_1.set_xscale("log")
normal_ax_1.set_yscale("log")
normal_ax_1.set_xlabel("SrcBytes")
normal_ax_1.set_ylabel("Frequency")
normal_ax_2.set_xlabel("SrcBytes")
normal_ax_2.set_ylabel("Frequency")
# normal_ax_3.set_xlabel("SrcBytes")
# normal_ax_3.set_ylabel("Frequency")
# normal_ax_4.set_xlabel("SrcBytes")
# normal_ax_4.set_ylabel("Frequency")
# normal_ax_5.set_xlabel("SrcBytes")
# normal_ax_5.set_ylabel("Frequency")

fig.savefig("capture20110818-2.binetflow-histogram-" + analyzed_parameter + ".png", dpi=100, bbox_inches='tight')

print("[" + datetime.now().__str__() + "]: " + 'Drawing ended')
