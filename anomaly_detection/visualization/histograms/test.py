from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.preprocessing import Preprocessing
from utils.logger import Logger

preprocessing = Preprocessing()
datasets_path = "../../../datasets/"

filenames = ["scenario_1.binetflow", "scenario_2.binetflow", "scenario_6.binetflow", "scenario_8.binetflow", "scenario_9.binetflow"]

analyzed_parameter = 'SrcBytes'
analyzed_file = filenames[0]

logger = Logger("../output/histogram/hist-" + analyzed_file + "-" + analyzed_parameter + ".log")

logger.log("Script starts")

original_dataset = pd.read_csv(datasets_path + analyzed_file)

X = original_dataset[:]

preprocessing.transform_labels_text_detailed(X)

X_normal = preprocessing.filter_by_column(X, column="Label", values=["Normal"])[analyzed_parameter]
X_bg = preprocessing.filter_by_column(X, column="Label", values=["Background"])[analyzed_parameter]
X_anomaly = preprocessing.filter_by_column(X, column="Label", values=["Anomaly"])[analyzed_parameter]

logger.log("Normal:\n" + X_normal.describe().__str__())
logger.log("90%:\n" + X_normal.quantile([0.9]).__str__())

logger.log("Background:\n" + X_bg.describe().__str__())
logger.log("90%:\n" + X_bg.quantile([0.9]).__str__())

logger.log("Anomaly:\n" + X_anomaly.describe().__str__())
logger.log("90%:\n" + X_anomaly.quantile([0.9]).__str__())

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(12,24))

ax1.hist([X_normal, X_bg, X_anomaly], color=['g', 'b', 'r'], alpha=0.5, bins=100, bottom=1)
ax2.hist([(X_normal[(X_normal < 100000)]), (X_bg[(X_bg< 100000)]), (X_anomaly[(X_anomaly < 100000)])], color=['g', 'b', 'r'], alpha=0.5, bins=100, bottom=1)
ax3.hist([(X_normal[(X_normal < 10000) & (X_normal > 100)]), (X_bg[(X_bg< 10000) & (X_bg > 100)]), (X_anomaly[(X_anomaly < 10000) & (X_anomaly > 100)])], color=['g', 'b', 'r'], alpha=0.5, bins=100, bottom=1)
ax4.hist([(X_normal[(X_normal < 2000) & (X_normal > 100)]), (X_bg[(X_bg< 2000) & (X_bg > 100)]), (X_anomaly[(X_anomaly < 2000) & (X_anomaly > 100)])], color=['g', 'b', 'r'], alpha=0.5, bins=100, bottom=10)

ax1.set_yscale("log")
ax1.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax3.set_yscale("log")
ax4.set_yscale("log")
ax1.set_xlabel("SrcBytes")
ax1.set_ylabel("Frequency")
ax2.set_xlabel("SrcBytes")
ax2.set_ylabel("Frequency")
ax3.set_xlabel("SrcBytes")
ax3.set_ylabel("Frequency")
ax4.set_xlabel("SrcBytes")
ax4.set_ylabel("Frequency")

fig.savefig("../output/histogram/hist-" + analyzed_file + "-" + analyzed_parameter + ".png", dpi=100, bbox_inches='tight')

logger.log("Drawing finished")

