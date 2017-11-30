from datetime import datetime

import pandas as pd
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from utils.sample_selector import SampleSelector
from sklearn.externals import joblib

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm


preprocessing = Preprocessing()
datasets_path = "../../../datasets/"

filenames = ["scenario_6.binetflow", "scenario_2.binetflow", "scenario_9.binetflow", "scenario_1.binetflow", "scenario_8.binetflow"]
# filenames = ["small_sample1.csv", "small_sample2.csv"]

numerical_features = [
    "Dur",
    "SrcBytes"
]

categorical_features = [
    "Dport",
    "Sport",
    "Dir",
    "State"
]

relevant_features = numerical_features[:]
relevant_features.extend(categorical_features)


for i in range(0, filenames.__len__()):
    analyzed_file = filenames[i]

    # logger = Logger("../output/taught-models/" + datetime.now().strftime("%Y-%m-%d_%H-%M").__str__() + "-osvm-"
    #                 + analyzed_file + ".log")

    logger = Logger()

    logger.log("Prediction script using One Class SVM starts. File: " + analyzed_file)
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
    # for f_c in range(0, categorical_features.__len__()):
    #     feature = categorical_features[f_c]
    #     logger.log("Transforming categorical feature: " + feature)
    #     preprocessing.transform_non_numerical_column(X, feature)

    # TODO: scaling/normalizing/standarizing numerical features

    sel = SampleSelector(X)
    logger.log("Splitting dataset")
    X_train, X_cv, X_test = sel.strategy_25min()

    # osvm = svm.OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
    #
    # # extracting labels to a separate dataframe
    # Y_train = X_train['attack']
    # Y_test = X_test['attack']
    #
    # # removing all unnecessary features, as well as labels
    # X_train = X_train[relevant_features]
    # X_test = X_test[relevant_features]
    #
    # logger.log("Model starts to learn")
    # model = osvm.fit(X_train)
    #
    # logger.log("Learning finished")
    #
    # # dumping taught model to file, so it may be retrieved later, if it was needed for further examination
    # model_file = "../output/taught-models/" + datetime.now().strftime("%Y-%m-%d_%H-%M").__str__() + "-osvm-" + analyzed_file + ".model"
    # logger.log("Dumping model to file: " + model_file)
    # joblib.dump(model, model_file)
    #
    # logger.log("Data dumped, now predicting. Sanity check with training data first:")
    # X_pred_train = model.predict(X_train)
    #
    # logger.log("Assessment:")
    # logger.log("Accuracy: " + metrics.accuracy_score(Y_train, X_pred_train).__str__())
    # logger.log("Precision: " + metrics.precision_score(Y_train, X_pred_train).__str__())
    # logger.log("Recall: " + metrics.recall_score(Y_train, X_pred_train).__str__())
    # logger.log("F1: " + metrics.f1_score(Y_train, X_pred_train).__str__())
    # logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_train, X_pred_train).__str__())
    #
    # logger.log("Checking model parameters with test dataset:")
    # X_pred_test = model.predict(X_test)
    # logger.log("Assessment:")
    # logger.log("Accuracy: " + metrics.accuracy_score(Y_test, X_pred_test).__str__())
    # logger.log("Precision: " + metrics.precision_score(Y_test, X_pred_test).__str__())
    # logger.log("Recall: " + metrics.recall_score(Y_test, X_pred_test).__str__())
    # logger.log("F1: " + metrics.f1_score(Y_test, X_pred_test).__str__())
    # logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_test, X_pred_test).__str__())












