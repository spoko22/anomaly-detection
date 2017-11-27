import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from pandas.tools.plotting import parallel_coordinates

preprocessing = Preprocessing()

datasets_path = "../../../datasets/"

filenames = ["scenario_1.binetflow", "scenario_2.binetflow", "scenario_6.binetflow", "scenario_8.binetflow", "scenario_9.binetflow"]

analyzed_features = [
    # "StartTime", # timestamp, all will be unique, no point in analyzing that
    "Dur",
    "Proto", # transform to numerical, not all describe feature are useful
    # "SrcAddr", # probably not useful
    "Sport", # transform to numerical, not all describe feature are useful
    "Dir", # transform to numerical, not all describe feature are useful
    # "DstAddr", # probably not useful
    "Dport", # transform to numerical, not all describe feature are useful
    "State", # transform to numerical, not all describe feature are useful
    "sTos",
    "dTos",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "attack"
]
logger = Logger("..\output\parallel-coordinates\pc-" + datetime.now().strftime("%Y-%m-%d_%H-%M").__str__() + ".log")

for i in range(0, filenames.__len__()):

    logger.log("Drawing " + (i+1).__str__() + "/" + filenames.__len__().__str__() + " sample")
    logger.log("Data transforming starts")

    original_dataset = preprocessing.read_file(datasets_path + filenames[i])

    # target dataset
    X = original_dataset[:]

    preprocessing.filter_by_column(X, "Dir", ["<?>", "<->", "<-"])

    preprocessing.transform_labels(X)
    preprocessing.transform_non_numerical_column(X, "Dport")
    preprocessing.transform_non_numerical_column(X, "Sport")
    preprocessing.transform_non_numerical_column(X, "Proto")
    preprocessing.transform_non_numerical_column(X, "Dir")
    preprocessing.transform_non_numerical_column(X, "State")

    preprocessing.normalize_columns(X, analyzed_features)  # features normalized to range [0,1], so they may be plotted

    X_normal = X[:]
    X_anomaly = X[:]

    # filter out anomalies to have two distinct plots
    X_normal = preprocessing.filter_by_column(X_normal, "attack", [-1])
    X_anomaly = preprocessing.filter_by_column(X_anomaly, "attack", [1])

    # let's leave only relevant features
    X_normal = X_normal[analyzed_features]
    X_anomaly = X_anomaly[analyzed_features]

    logger.log("Data transforming finished, drawing anomaly plot first")

    parallel_coordinates(X_anomaly, 'attack')
    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8)
    fig.savefig("parallel-coordinates_" + filenames[i] +"_anomaly.png", dpi=100, bbox_inches='tight')

    logger.log("Anomalies drawn, drawing regular data plot")

    parallel_coordinates(X_normal, 'attack')
    fig = plt.gcf()
    fig.set_size_inches(19.2, 10.8)
    fig.savefig("parallel-coordinates_" + filenames[i] +"_regular.png", dpi=100, bbox_inches='tight')

    logger.log("Drawing sample " + (i+1).__str__() + " finished")
