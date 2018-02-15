from datetime import datetime

import pandas as pd
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from utils.sample_selector import SampleSelector
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import metrics
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score, precision_score,
                             precision_recall_fscore_support, accuracy_score)
from keras.losses import binary_crossentropy
import numpy as np
import os
import matplotlib.pyplot as plt
from feature_engineering.technical import TechnicalFeatures

class Autoencoder:
    datasets_path = "../../../datasets/"
    X = None
    numerical_features = []
    categorical_features = []
    binary_features = []
    execution_version = None
    logger = None
    preprocessing = Preprocessing()
    tech = TechnicalFeatures

    def log(self, msg):
        self.logger.log(msg)

    def __init__(self, file, version="test", numerical_features=[], categorical_features=[], binary_features=[]):
        original_dataset = pd.read_csv(self.datasets_path + file)
        self.X = original_dataset[:]
        self.execution_version = version

        analyzed_file = file
        date = datetime.now().strftime("%Y-%m-%d_%H-%M").__str__()
        directory = "../output-autoencoders/taught-models/" + date + '-' + self.execution_version + '-' + analyzed_file

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger = Logger(directory + "/" + "autoencoder-" + analyzed_file + ".log")

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.binary_features = []
        self.relevant_features = numerical_features[:]
        self.relevant_features.extend(categorical_features)
        self.relevant_features.extend(binary_features)

        self.X = self.__add_technical_features__(self.X, self.numerical_features, self.categorical_features, self.binary_features)

    def __add_technical_features__(self, dataset, numerical_features, categorical_features, binary_features):
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

    def __transform_categories_to_numbers__(self):
        # transforming categorical data into numerical data
        for f_c in range(0, self.categorical_features.__len__()):
            feature = self.categorical_features[f_c]
            self.logger.log("Transforming categorical feature: " + feature)
            self.preprocessing.transform_non_numerical_column(self.X, feature)

    def __replace_labels__(self):
        self.preprocessing.transform_labels(self.X)


    def __split_datasets__(self, train_size=400000, test_size=100000):
        sel = SampleSelector(self.X)
        X_train, X_cv, X_test = sel.novelty_detection_random(train_size=train_size, test_size=test_size)
        return X_train, X_cv, X_test

    def perform_ae(self, nb_epoch=100, batch_size= 128, train_size=400000, test_size=100000):
        self.__replace_labels__()
        self.__transform_categories_to_numbers__()

        X_train, X_cv, X_test = self.__split_datasets__(train_size=train_size, test_size=test_size)

        self.preprocessing.transform_labels_invert(X_train)
        self.preprocessing.transform_labels_invert(X_test)

        y_train = X_train['inlier']
        X_train = X_train[self.relevant_features]

        # X_cv = X_cv[self.relevant_features]
        y_test = X_test['inlier']
        X_test = X_test[self.relevant_features]

        # # standarization, normalization etc
        # for f_n in range(0, self.numerical_features.__len__()):
        #     feature = self.numerical_features[f_n]
        #     if feature in self.relevant_features:
        #         self.logger.log("Quantile standarization of feature: " + feature)
        #         # preprocessing.quantile_standarization(X_train, feature)
        #         self.preprocessing.quantile_standarization(X_train, feature)
        #         self.preprocessing.quantile_standarization(X_test, feature)
        #         # preprocessing.quantile_standarization(X_cv, feature)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        input_dim = X_train.shape[1]
        encoding_dim = 14

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh",
                        activity_regularizer=regularizers.l1(10e-5))(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
        decoder = Dense(input_dim, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        autoencoder.compile(optimizer='adam',
                            loss='mse',
                            metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath="model.h5",
                                       verbose=0,
                                       save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)
        history = autoencoder.fit(X_train, X_train,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  validation_split=0.2,
                                  validation_data=(X_test, X_test),
                                  verbose=1,
                                  callbacks=[checkpointer, tensorboard]).history

        predictions = autoencoder.predict(X_test)

        train_pred = autoencoder.predict(X_train)

        mse_train = np.mean(np.power(X_train - train_pred, 2), axis=1)
        mse = np.mean(np.power(X_test - predictions, 2), axis=1)

        # bc = binary_crossentropy(y_test, predictions)
        error_df = pd.DataFrame({'reconstruction_error': mse,
                                 'true_class': y_test})
        error_df.describe()

        fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
        roc_auc = auc(fpr, tpr)

        threshold = error_df.loc[error_df.true_class == 1].reconstruction_error.quantile(q=0.01)

        y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
        error_df['predictions'] = y_pred

        y_test_inliers_true = [-1 if tc == 1 else 1 for tc in error_df.true_class.values]
        y_test_inliers_pred = [-1 if tc == 1 else 1 for tc in y_pred]

        tn, fp, fn, tp = confusion_matrix(y_test_inliers_true, y_test_inliers_pred).ravel()

        acc = accuracy_score(y_test_inliers_true, y_test_inliers_pred)
        precision = precision_score(y_test_inliers_true, y_test_inliers_pred)
        recall = recall_score(y_test_inliers_true, y_test_inliers_pred)
        f1 = f1_score(y_test_inliers_true, y_test_inliers_pred)
        auc_score = roc_auc

        self.log("Accuracy: " + acc.__str__())
        self.log("Precision: " + precision.__str__())
        self.log("Recall: " + recall.__str__())
        self.log("F1: " + f1.__str__())
        self.log("AUC: " + auc_score.__str__())

        self.log("TP: " + tp.__str__())
        self.log("TN: " + tn.__str__())
        self.log("FP: " + fp.__str__())
        self.log("FN: " + fn.__str__())

        return acc, precision, recall, f1, auc_score


