from datetime import datetime

import pandas as pd
from sklearn import metrics
from sklearn import svm
from utils.logger import Logger
from utils.preprocessing import Preprocessing

logger = Logger("logs\logs" + datetime.now().strftime("%Y-%m-%d_%H-%M").__str__() + ".txt")
# logger = Logger()
preprocessing = Preprocessing()

#    0           1        2       3           4      5         6          7       8      9       10        11          12          13         14
# StartTime,    Dur,    Proto,  SrcAddr,    Sport,  Dir,    DstAddr,    Dport,  State,  sTos,   dTos,   TotPkts,    TotBytes,   SrcBytes,   Label

# input_file_name = "samples/small_sample.csv"
# input_file_name = "capture20110815.binetflow" #scenario 4
input_file_name = "capture20110818-2.binetflow"
# input_file_name = "capture20110818-22.binetflow"

for i in range(1, 2):
    logger.log("Try nr: " + i.__str__())
    relevant_features = [
        "Dur",
        "SrcBytes",
        "Dport",
        "Sport",
        "Dir",
        "State"
    ]

    logger.log("Script starts")

    logger.log("Reading data from file " + input_file_name + " started")
    original_dataset = pd.read_csv(input_file_name, low_memory=False)
    preprocessing.filter_by_column(original_dataset, "Dir", ["<?>", "<->", "<-"])
    logger.log("Reading data from file finished")

    logger.log("Replacing labels")
    preprocessing.transform_labels(original_dataset)

    relevant_features_with_labels = relevant_features[:]
    relevant_features_with_labels.append('attack')

    logger.log("Using only relevant features: " + relevant_features.__str__())
    dataset = original_dataset[relevant_features_with_labels]

    preprocessing.transform_non_numerical_column(dataset, "Dport")
    preprocessing.transform_non_numerical_column(dataset, "Sport")
    preprocessing.transform_non_numerical_column(dataset, "State")
    preprocessing.transform_non_numerical_column(dataset, "Dir")
    # preprocessing.transform_non_numerical_column(dataset, "SrcAddr")
    # preprocessing.transform_non_numerical_column(dataset, "DstAddr")

    target = original_dataset['attack']
    preprocessing.normalize_columns(dataset, relevant_features)

    # train_data, test_data, train_target, test_target = train_test_split(dataset, target, train_size = 0.8)

    dataset_size = len(dataset.index)

    train_size = round(0.8 * dataset_size)

    X_train = dataset[:]
    X_test = dataset[:]

    X_train = preprocessing.filter_by_column(X_train, "attack", [-1])
    train_data = X_train[:train_size]
    train_data = train_data[relevant_features]
    test_data = preprocessing.get_not_present_values(train_data, X_test)
    test_target = test_data['attack']
    test_data = test_data[relevant_features]

    logger.log("Training starts")
    model = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma=0.00005)
    model.fit(train_data)
    logger.log("Training finished")

    logger.log("Predicting starts")
    preds = model.predict(test_data)
    logger.log("Predicting finished")

    targs = test_target

    logger.log("Assessment:")
    logger.log("Accuracy: " + metrics.accuracy_score(targs, preds).__str__())
    logger.log("Precision: " + metrics.precision_score(targs, preds).__str__())
    logger.log("Recall: " + metrics.recall_score(targs, preds).__str__())
    logger.log("F1: " + metrics.f1_score(targs, preds).__str__())
    logger.log("Area under curve (auc): " + metrics.roc_auc_score(targs, preds).__str__())
    logger.log("\n\n")
