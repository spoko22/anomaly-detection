from datetime import datetime

import pandas as pd
import os
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from utils.sample_selector import SampleSelector
from utils.dataframe_utils import DataframeUtils
from sklearn.externals import joblib
from multiprocessing import Pool
from sklearn.decomposition import PCA
from skmca.skmca.skmca import MCA
import numpy as np
import gc

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from feature_engineering.freq import FrequencyIndicator
from feature_engineering.technical import TechnicalFeatures

execution_version = "1.4.1"

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
    # "SrcAddr",
    # "DstAddr",
    "Dport",
    "Sport",
    "Proto",
    "Dir",
    "State"
]

categorical_features_to_freq = [
    # "SrcAddr",
    # "DstAddr",
    # "Dport",
    # "Sport",
    # "Proto",
    # "Dir",
    # "State"
]

binary_features = [

]

categorical_features_to_dummies = []

def add_technical_features(dataset, numerical_features, categorical_features, binary_features):
    tech = TechnicalFeatures()
    is_http = "is_http"
    is_email = "is_email"
    is_irc = "is_irc"
    tech.add_is_http(dataset, is_http)
    binary_features.append(is_http)
    tech.add_is_email(dataset, is_email)
    binary_features.append(is_email)
    tech.add_is_irc(dataset, is_irc)
    binary_features.append(is_irc)
    dataset = tech.rates(dataset, "TotBytes", "TotBytesRate")
    dataset = tech.rates(dataset, "TotPkts", "TotPktsRate")
    dataset = tech.rates(dataset, "SrcBytes", "SrcBytesRate")
    dataset = tech.rates(dataset, "TotBytes", "TotBytesPerPacket", duration_col="TotPkts")
    dataset = tech.rates(dataset, "SrcBytes", "SrcBytesPerPacket", duration_col="TotPkts")
    dataset = tech.subtraction(dataset, "TotBytes", "SrcBytes", "PacketOverhead")
    numerical_features.append("TotBytesRate")
    numerical_features.append("TotPktsRate")
    numerical_features.append("SrcBytesRate")
    numerical_features.append("TotBytesPerPacket")
    numerical_features.append("SrcBytesPerPacket")
    numerical_features.append("PacketOverhead")

    return dataset


def perform_osvm(filename):
    analyzed_file = filename
    date = datetime.now().strftime("%Y-%m-%d_%H-%M").__str__()
    directory = "../output/taught-models/" + date + '-' + execution_version + '-' + analyzed_file

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = Logger(directory + "/" + "osvm-" + analyzed_file + ".log")

    logger.log("Prediction script using One Class SVM starts. File: " + analyzed_file)

    # reading in data
    logger.log("Reading in data from file: " + analyzed_file)
    original_dataset = pd.read_csv(datasets_path + analyzed_file)
    logger.log("Dataset read")

    # feature engineering on features that are true for every single row, so it should be done before other
    original_dataset = add_technical_features(original_dataset, numerical_features, categorical_features, binary_features)

    for d in range(0, categorical_features_to_dummies.__len__()):
        dummy = categorical_features_to_dummies[d]
        dummies = pd.get_dummies(original_dataset[dummy])
        for dummy_id in range(0, list(dummies).__len__()):
            header = list(dummies)[dummy_id]
            original_dataset[header] = dummies[header]
            binary_features.append(header)

    for features_number in range(1, 10):
        X = original_dataset[:]
        engineered_features = []
        relevant_features = numerical_features[:]
        relevant_features.extend(categorical_features)
        relevant_features.extend(binary_features)

        # transforming labels
        logger.log("Transforming labels")
        preprocessing.transform_labels_with_normals(X)

        # transforming categorical data into numerical data
        for f_c in range(0, categorical_features.__len__()):
            feature = categorical_features[f_c]
            logger.log("Transforming categorical feature: " + feature)
            preprocessing.transform_non_numerical_column(X, feature)

        clear_distinction = preprocessing.filter_by_column(X, 'inlier', [-1, 1])
        preprocessing.transform_labels(X)

        maxc = 7
        acc = np.array([])
        precision = np.array([])
        recall = np.array([])
        f1 = np.array([])
        auc = np.array([])
        sel = SampleSelector(X)
        for i in range(1, maxc+1):
            logger.log('Loop ' + i.__str__() + "/" + maxc.__str__())
            logger.log("Splitting dataset")
            X_train, X_cv, X_test = sel.novelty_detection_random(train_size=100000, test_size=100000)
            X_train.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-train.csv")
            X_cv.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-cv.csv")
            X_test.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-test.csv")
            # preprocessing.transform_labels(X)
            # preprocessing.transform_labels(X_train)
            # preprocessing.transform_labels(X_cv)
            # preprocessing.transform_labels(X_test)

            X_non_tested_regularities = X[:]
            # everything then is based on idea "at the time of learning we only have regularities", thus anomalies should be
            # filtered out to make sure classifier learns only what it should. Standarization performed on data with anomalies
            # may be wrong
            X_non_tested_regularities = preprocessing.filter_by_column(X_non_tested_regularities, 'inlier', [1])
            X_regularities = X_non_tested_regularities[:]
            X_non_tested_regularities = preprocessing.get_not_present_values(X_non_tested_regularities, X_test)
            X_non_tested_regularities = preprocessing.get_not_present_values(X_non_tested_regularities, X_cv)
            freq = FrequencyIndicator(reference_set=X_regularities)

            original_target = X['inlier']
            X_train['inlier'] = original_target

            # feature engineering
            for f_c in range(0, categorical_features_to_freq.__len__()):
                feature = categorical_features_to_freq[f_c]
                new_feature = feature + "_freq"
                logger.log("\"Frequency\" feature engineering performed on: " + feature)
                X_non_tested_regularities = freq.using_mean(X_non_tested_regularities, feature, new_column=new_feature)
                X_train = freq.using_mean(X_train, feature, new_column=new_feature)
                X_test = freq.using_mean(X_test, feature, new_column=new_feature)
                clear_distinction = freq.using_mean(clear_distinction, feature, new_column=new_feature)
                engineered_features.append(new_feature)
                relevant_features.append(new_feature)

            X_chosen = preprocessing.feature_selection_chi2(clear_distinction[relevant_features],
                                                            clear_distinction['inlier'], features_number)

            chosen_features = list(X_chosen.columns.values)

            logger.log("Features available: " + relevant_features.__str__())
            logger.log("Features used: " + chosen_features.__str__())

            # standarization, normalization etc
            for f_n in range(0, numerical_features.__len__()):
                feature = numerical_features[f_n]
                if feature in chosen_features:
                    logger.log("Quantile standarization of feature: " + feature)
                    preprocessing.quantile_standarization(X_non_tested_regularities, feature)
                    preprocessing.quantile_standarization(X_train, feature)
                    preprocessing.quantile_standarization(X_test, feature)
                    # preprocessing.quantile_standarization(X_cv, feature)

            for f_e in range(0, engineered_features.__len__()):
                feature = engineered_features[f_e]
                if feature in chosen_features:
                    logger.log("Removing mean and scaling to unit variance for feature: " + feature)
                    preprocessing.quantile_standarization(X_non_tested_regularities, feature)
                    preprocessing.quantile_standarization(X_train, feature)
                    preprocessing.quantile_standarization(X_test, feature)
                    # preprocessing.quantile_standarization(X_cv, feature)

            del X_non_tested_regularities
            gc.collect()
            osvm = svm.OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)

            # extracting labels to a separate dataframe
            Y_train = X_train['inlier']
            Y_test = X_test['inlier']

            # removing all unnecessary features, as well as labels
            X_train = X_train[chosen_features]
            X_test = X_test[chosen_features]

            chosen_numerical = []
            chosen_categorical = []

            for c in range(0, chosen_features.__len__()):
                chosen = chosen_features[c]
                if chosen in categorical_features:
                    chosen_categorical.append(chosen)
                if chosen in numerical_features:
                    chosen_numerical.append(chosen)

            if chosen_numerical.__len__() > 0 :
                pca = PCA(whiten=False).fit(X_train[chosen_numerical])

                X_train_num = pca.transform(X_train[chosen_numerical])
                X_test_num = pca.transform(X_test[chosen_numerical])

                X_train = X_train[chosen_categorical]
                X_train = pd.concat([X_train, pd.DataFrame(data=X_train_num, index=X_train.index.values)], axis=1)
                X_test = X_test[chosen_categorical]
                X_test = pd.concat([X_test, pd.DataFrame(data=X_test_num, index=X_test.index.values)], axis=1)

            logger.log("Model starts to learn")
            model = osvm.fit(X_train)

            logger.log("Learning finished")

            # dumping taught model to file, so it may be retrieved later, if it was needed for further examination
            model_file = directory + "/osvm-" + analyzed_file + "-fn-" + features_number.__str__() + ".model"
            logger.log("Dumping model to file: " + model_file)
            joblib.dump(model, model_file)

            logger.log("Data dumped, now predicting. Sanity check with training data first:")
            X_pred_train = model.predict(X_train)

            logger.log("Predictions for train dataset finished, saving datasets for later analysis")
            df_train = dfu.merge_results(X_train, Y_train, X_pred_train)
            df_train.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-fn-" + features_number.__str__() + "-train-with_prediction.csv")
            logger.log("Features used: " + chosen_features.__str__())
            logger.log("Assessment:")
            logger.log("Accuracy: " + metrics.accuracy_score(Y_train, X_pred_train).__str__())
            logger.log("Precision: " + metrics.precision_score(Y_train, X_pred_train).__str__())
            logger.log("Recall: " + metrics.recall_score(Y_train, X_pred_train).__str__())
            logger.log("F1: " + metrics.f1_score(Y_train, X_pred_train).__str__())
            # logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_train, X_pred_train).__str__())
            tn, fp, fn, tp = metrics.confusion_matrix(Y_train, X_pred_train).ravel()
            logger.log("TP: " + tp.__str__())
            logger.log("TN: " + tn.__str__())
            logger.log("FP: " + fp.__str__())
            logger.log("FN: " + fn.__str__())

            logger.log("Checking model parameters with test dataset:")
            X_pred_test = model.predict(X_test)

            logger.log("Predictions for test dataset finished, saving datasets for later analysis")
            df_test = dfu.merge_results(X_test, Y_test, X_pred_test)
            df_test.to_csv(path_or_buf=directory + "/" + "osvm-" + analyzed_file + "-fn-" + features_number.__str__() + "-test-with_prediction.csv")

            logger.log("Assessment:")
            logger.log("Accuracy: " + metrics.accuracy_score(Y_test, X_pred_test).__str__())
            logger.log("Precision: " + metrics.precision_score(Y_test, X_pred_test).__str__())
            logger.log("Recall: " + metrics.recall_score(Y_test, X_pred_test).__str__())
            logger.log("F1: " + metrics.f1_score(Y_test, X_pred_test).__str__())
            logger.log("Area under curve (auc): " + metrics.roc_auc_score(Y_test, X_pred_test).__str__())
            tn, fp, fn, tp = metrics.confusion_matrix(Y_test, X_pred_test).ravel()
            logger.log("TP: " + tp.__str__())
            logger.log("TN: " + tn.__str__())
            logger.log("FP: " + fp.__str__())
            logger.log("FN: " + fn.__str__())

            acc = np.append(acc, metrics.accuracy_score(Y_test, X_pred_test))
            precision = np.append(precision, metrics.precision_score(Y_test, X_pred_test))
            recall = np.append(recall, metrics.recall_score(Y_test, X_pred_test))
            f1 = np.append(f1, metrics.f1_score(Y_test, X_pred_test))
            auc = np.append(auc, metrics.roc_auc_score(Y_test, X_pred_test))

            del X_train, X_test, Y_test, Y_train, X_pred_train, X_pred_test
            gc.collect()

            logger.log('ACC, PRECISION, RECALL, F1, AUC MEAN\n' + np.mean(acc).__str__() + '\n' + np.mean(precision).__str__() + '\n' + np.mean(recall).__str__() + '\n' + np.mean(f1).__str__() + '\n' + np.mean(auc).__str__())
            logger.log('ACC, PRECISION, RECALL, F1, AUC STD\n' + np.std(acc).__str__() + '\n' + np.std(precision).__str__() + '\n' + np.std(recall).__str__() + '\n' + np.std(f1).__str__() + '\n' + np.std(auc).__str__())

    logger.log("Working on file [" + analyzed_file + "] finished.")

if __name__ == '__main__':
    pool = Pool()
    pool.map(perform_osvm, filenames)
    pool.close()
    pool.join()









