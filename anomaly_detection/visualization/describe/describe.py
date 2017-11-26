from utils.preprocessing import Preprocessing
from utils.logger import Logger

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
    "attack" # needs to be transformed from Label first
]

for i in range(0, filenames.__len__()):
    logger = Logger("..\output\describe\describe-" + filenames[i] + ".log")
    logger.log("Analyzing " + (i+1).__str__() + "/" + filenames.__len__().__str__() + " sample")

    original_dataset = preprocessing.read_file(datasets_path + filenames[i])

    # target dataset
    X = original_dataset[:]

    preprocessing.transform_non_numerical_column(X, "Proto")
    preprocessing.transform_non_numerical_column(X, "Sport")
    preprocessing.transform_non_numerical_column(X, "Dir")
    preprocessing.transform_non_numerical_column(X, "Dport")
    preprocessing.transform_non_numerical_column(X, "State")

    preprocessing.transform_labels(X)


    # only relevant features should be analyzed
    X = X[analyzed_features]

    logger.log("\n" + X.describe().__str__())

    logger.log("File " + filenames[i] + " described")

