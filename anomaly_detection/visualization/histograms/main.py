from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.preprocessing import Preprocessing
from utils.logger import Logger
import matplotlib.patches as mpatches

preprocessing = Preprocessing()
datasets_path = "../../../datasets/"

filenames = ["scenario_1.binetflow", "scenario_2.binetflow", "scenario_6.binetflow", "scenario_8.binetflow", "scenario_9.binetflow"]
# filenames = ["small_sample1.csv", "small_sample2.csv"]

analyzed_parameter = 'SrcBytes'

for i in range(0, filenames.__len__()):
    analyzed_file = filenames[i]

    logger = Logger("../output/histogram/hist-" + analyzed_file + "-" + analyzed_parameter + ".log")
    logger.log("Script starts drawing histogram for file " + analyzed_file)

    # reading in data
    original_dataset = pd.read_csv(datasets_path + analyzed_file)

    X = original_dataset[:]
    preprocessing.transform_labels_text_detailed(X)

    # splitting whole dataset into parts
    X_normal = preprocessing.filter_by_column(X, column="Label", values=["Normal"])[analyzed_parameter]
    X_bg = preprocessing.filter_by_column(X, column="Label", values=["Background"])[analyzed_parameter]
    X_anomaly = preprocessing.filter_by_column(X, column="Label", values=["Anomaly"])[analyzed_parameter]

    quantile_X_normal = X_normal.quantile([0.1, 0.9])
    quantile_X_bg = X_bg.quantile([0.1, 0.9])
    quantile_X_anomaly = X_anomaly.quantile([0.1, 0.9])

    logger.log("\n\nNormal:\n" + X_normal.describe().__str__())
    logger.log("Quantile boundaries:\n" + quantile_X_normal.__str__())

    logger.log("\n\nBackground:\n" + X_bg.describe().__str__())
    logger.log("Quantile boundaries:\n" + quantile_X_bg.__str__())

    logger.log("\n\nAnomalies:\n" + X_anomaly.describe().__str__())
    logger.log("Quantile boundaries:\n" + quantile_X_anomaly.__str__())

    anomaly_patch = mpatches.Patch(color='r', label='Anomalies')
    normal_patch = mpatches.Patch(color='g', label='Normal')
    bg_patch = mpatches.Patch(color='b', label='Background')

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(12, 24))
    ax1.legend(handles=[anomaly_patch, normal_patch, bg_patch])
    ax2.legend(handles=[anomaly_patch, normal_patch, bg_patch])
    ax3.legend(handles=[anomaly_patch, normal_patch, bg_patch])
    ax4.legend(handles=[anomaly_patch, normal_patch, bg_patch])

    # rough visualization, showing anything that doesn't go into 80% of observations
    ax1.hist([X_normal, X_bg, X_anomaly], color=['g', 'b', 'r'], alpha=0.5, bins=100, bottom=1)

    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlabel(analyzed_parameter)
    ax1.set_ylabel("Frequency")
    ax1.title.set_text('Whole dataset')

    # visualizing 80% of regular "Normal observations"
    range_start = quantile_X_normal.iloc[0]
    range_end = quantile_X_normal.iloc[1]

    ax2.hist([(X_normal[(X_normal <= range_end) & (X_normal >= range_start)]), (X_bg[(X_bg <= range_end) & (X_bg >= range_start)]),
              (X_anomaly[(X_anomaly <= range_end) & (X_anomaly >= range_start)])], color=['g', 'b', 'r'], alpha=0.5, bins=20,
             bottom=1)

    # figuring out the scales
    x_logscale = (range_end - range_start) > 10000

    if x_logscale:
        ax2.set_xscale("log")

    ax2.title.set_text('80% of records labeled as "Normal" traffic')
    ax2.set_yscale("log")
    ax2.set_xlabel(analyzed_parameter)
    ax2.set_ylabel("Frequency")

    # visualizing 80% of regular "Background observations"
    range_start = quantile_X_bg.iloc[0]
    range_end = quantile_X_bg.iloc[1]

    ax3.hist([(X_normal[(X_normal <= range_end) & (X_normal >= range_start)]), (X_bg[(X_bg <= range_end) & (X_bg >= range_start)]),
              (X_anomaly[(X_anomaly <= range_end) & (X_anomaly >= range_start)])], color=['g', 'b', 'r'], alpha=0.5, bins=20,
             bottom=1)

    # figuring out the scales
    x_logscale = (range_end - range_start) > 10000

    if x_logscale:
        ax3.set_xscale("log")

    ax3.title.set_text('80% of records labeled as "Background" traffic')
    ax3.set_yscale("log")
    ax3.set_xlabel(analyzed_parameter)
    ax3.set_ylabel("Frequency")

    # visualizing 80% of regular "Anomaly observations"
    range_start = quantile_X_anomaly.iloc[0]
    range_end = quantile_X_anomaly.iloc[1]

    ax4.hist([(X_normal[(X_normal <= range_end) & (X_normal >= range_start)]), (X_bg[(X_bg <= range_end) & (X_bg >= range_start)]),
              (X_anomaly[(X_anomaly <= range_end) & (X_anomaly >= range_start)])], color=['g', 'b', 'r'], alpha=0.5, bins=20,
             bottom=1)

    # figuring out the scales
    x_logscale = (range_end - range_start) > 10000

    ax4.title.set_text('80% of records labeled as "Anomaly" traffic')
    if x_logscale:
        ax4.set_xscale("log")

    ax4.set_yscale("log")
    ax4.set_xlabel(analyzed_parameter)
    ax4.set_ylabel("Frequency")

    fig.savefig("../output/histogram/hist-" + analyzed_file + "-" + analyzed_parameter + ".png", dpi=100, bbox_inches='tight')

    logger.log("Drawing finished")
