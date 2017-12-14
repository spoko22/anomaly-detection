from datetime import datetime

import pandas as pd
import os
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from utils.sample_selector import SampleSelector
from utils.dataframe_utils import DataframeUtils
from sklearn.externals import joblib
from multiprocessing import Pool
import itertools

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm

execution_version = "1.2.1"

preprocessing = Preprocessing()
datasets_path = "../../../datasets/"
dfu = DataframeUtils()

filenames = ["scenario_6.binetflow", "scenario_2.binetflow", "scenario_9.binetflow", "scenario_1.binetflow", "scenario_8.binetflow"]
# filenames = ["small_sample1.csv", "small_sample2.csv"]

# 'Dur' 'SrcBytes' 'TotBytes' 'Dport' 'Sport' - tests suggest these params seem to be slightly better
# than the others, so they will be used

numerical_features = [
    "Dur",
    "SrcBytes",
    # "TotPkts",
    "TotBytes"
]

categorical_features = [
    # "Proto",
    "Dport",
    "Sport",
    # "Dir",
    # "State"
]

relevant_features = numerical_features[:]
relevant_features.extend(categorical_features)

gamma = [0.1, 0.2, 0.5]
nu = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]

tuned_parameters = []

for g in range(0, gamma.__len__()):
    for n in range(0, nu.__len__()):
        tuned_parameters.append((gamma[g], nu[n]))


def tune_osvm(filename):
    analyzed_file = filename

    date = datetime.now().strftime("%Y-%m-%d_%H-%M").__str__()
    directory = "../output/tuning/" + date + '-' + execution_version + '-' + analyzed_file

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = Logger(directory + "/" + "osvm-" + analyzed_file + ".log")

    logger.log("Tuning OSVM starts. File: " + analyzed_file)
    logger.log("Features used: " + relevant_features.__str__())

    # reading in data
    logger.log("Reading in data from file: " + analyzed_file)
    original_dataset = pd.read_csv(datasets_path + analyzed_file)
    logger.log("Dataset read")

    X = original_dataset[:]

    # transforming labels
    logger.log("Transforming labels")
    preprocessing.transform_labels(X)

    # transforming categorical data into numerical data
    for f_c in range(0, categorical_features.__len__()):
        feature = categorical_features[f_c]
        logger.log("Transforming categorical feature: " + feature)
        preprocessing.transform_non_numerical_column(X, feature)

    # TODO: scaling/normalizing/standarizing numerical features

    sel = SampleSelector(X)
    logger.log("Splitting dataset")
    X_train, X_cv, X_test = sel.novelty_detection_random(train_size=200000, test_size=50000)
    X_train.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-train.csv")
    X_cv.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-cv.csv")
    X_test.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-test.csv")

    # extracting labels to a separate dataframe
    Y_train = X_train['inlier']
    Y_test = X_test['inlier']

    # removing all unnecessary features, as well as labels
    X_train = X_train[relevant_features]
    X_test = X_test[relevant_features]

    for p in range(0, tuned_parameters.__len__()):
        parameters_pair = tuned_parameters[p]
        logger.log("Tuning starts for parameters:")
        logger.log("nu: %s" % parameters_pair[1].__str__())
        logger.log("gamma: %s" % parameters_pair[0].__str__())
        osvm = svm.OneClassSVM(kernel='rbf', nu=parameters_pair[1], gamma=parameters_pair[0])

        logger.log("Model starts to learn")
        model = osvm.fit(X_train)

        logger.log("Learning finished")

        # dumping taught model to file, so it may be retrieved later, if it was needed for further examination
        model_file = directory + "/osvm-" + analyzed_file + ".model"
        logger.log("Dumping model to file: " + model_file)
        joblib.dump(model, model_file)

        logger.log("Data dumped, now predicting. Sanity check with training data first:")
        Y_pred_train = model.predict(X_train)

        logger.log("Predictions for train dataset finished, saving datasets for later analysis")
        df_train = dfu.merge_results(X_train, Y_train, Y_pred_train)
        df_train.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-train-with_prediction.csv")

        logger.log("Assessment:")
        logger.log("Accuracy: " + metrics.accuracy_score(Y_train, Y_pred_train).__str__())
        logger.log("Precision: " + metrics.precision_score(Y_train, Y_pred_train).__str__())
        logger.log("Recall: " + metrics.recall_score(Y_train, Y_pred_train).__str__())
        logger.log("F1: " + metrics.f1_score(Y_train, Y_pred_train).__str__())
        # logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_train, X_pred_train).__str__())

        tn, fp, fn, tp = metrics.confusion_matrix(Y_train, Y_pred_train).ravel()
        logger.log("TP: " + tp.__str__())
        logger.log("TN: " + tn.__str__())
        logger.log("FP: " + fp.__str__())
        logger.log("FN: " + fn.__str__())

        logger.log("Checking model parameters with test dataset:")
        Y_pred_test = model.predict(X_test)

        logger.log("Predictions for test dataset finished, saving datasets for later analysis")
        df_test = dfu.merge_results(X_test, Y_test, Y_pred_test)
        df_test.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-test-with_prediction.csv")

        logger.log("Assessment:")
        logger.log("Accuracy: " + metrics.accuracy_score(Y_test, Y_pred_test).__str__())
        logger.log("Precision: " + metrics.precision_score(Y_test, Y_pred_test).__str__())
        logger.log("Recall: " + metrics.recall_score(Y_test, Y_pred_test).__str__())
        logger.log("F1: " + metrics.f1_score(Y_test, Y_pred_test).__str__())
        logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_test, Y_pred_test).__str__())

        tn, fp, fn, tp = metrics.confusion_matrix(Y_test, Y_pred_test).ravel()
        logger.log("TP: " + tp.__str__())
        logger.log("TN: " + tn.__str__())
        logger.log("FP: " + fp.__str__())
        logger.log("FN: " + fn.__str__())

    logger.log("Working on file [" + analyzed_file + "] finished.")


if __name__ == '__main__':
    pool = Pool()
    pool.map(tune_osvm, filenames)
    pool.close()
    pool.join()









