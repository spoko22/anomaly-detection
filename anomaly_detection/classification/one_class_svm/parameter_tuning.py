from datetime import datetime

import pandas as pd
import os
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from utils.sample_selector import SampleSelector
from utils.dataframe_utils import DataframeUtils
from sklearn.externals import joblib
from multiprocessing import Pool

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

execution_version = "1.1.3"

preprocessing = Preprocessing()
datasets_path = "../../../datasets/"
dfu = DataframeUtils()

filenames = ["scenario_6.binetflow", "scenario_2.binetflow", "scenario_9.binetflow", "scenario_1.binetflow", "scenario_8.binetflow"]
# filenames = ["small_sample1.csv", "small_sample2.csv"]

numerical_features = [
    "Dur",
    "SrcBytes",
    "TotPkts",
    "TotBytes"
]

categorical_features = [
    "Proto",
    "Dport",
    "Sport",
    "Dir",
    "State"
]

relevant_features = numerical_features[:]
relevant_features.extend(categorical_features)

def perform_tuning_using_file(filename):
    analyzed_file = filename
    date = datetime.now().strftime("%Y-%m-%d_%H-%M").__str__()
    directory = "../output/tuning/" + date + '-' + execution_version + '-' + analyzed_file

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = Logger(directory + "/" + "osvm-" + analyzed_file + ".log")

    logger.log("Prediction script using One Class SVM starts. File: " + analyzed_file)
    logger.log("Features available: " + relevant_features.__str__())

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

    sel = SampleSelector(X)
    logger.log("Splitting dataset")
    X_train, X_cv, X_test = sel.novelty_detection_random(train_size=20000, test_size=5000, cv_size=5000)
    X_train.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-train.csv")
    X_cv.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-cv.csv")
    X_test.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-test.csv")

    # -------------------------------------------------------------------------------------------------------------------

    y_train, y_test, y_cv = X_train['inlier'], X_test['inlier'], X_cv['inlier']

    X_train = X_train[relevant_features]
    X_cv = X_cv[relevant_features]
    X_test = X_test[relevant_features]

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'],
                         'gamma': [0.1, 1e-2],
                         'nu': [0.1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5]}]

    scores = ['accuracy', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(svm.OneClassSVM(), tuned_parameters, scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


perform_tuning_using_file(filenames[0])