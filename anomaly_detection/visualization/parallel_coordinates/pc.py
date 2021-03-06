import matplotlib.pyplot as plt
from datetime import datetime
from utils.preprocessing import Preprocessing
from utils.logger import Logger
from pandas.plotting import parallel_coordinates

class PC:
    datasets_path = None
    output_path = None
    analyzed_features = []
    preprocessing = Preprocessing()

    def __init__(self, datasets_path, output_path, analyzed_features):
        self.datasets_path = datasets_path
        self.output_path = output_path
        self.analyzed_features = analyzed_features

    def draw(self, filename):
        logger = Logger(self.output_path + "\pc-" + filename + datetime.now().strftime("%Y-%m-%d_%H-%M").__str__() + ".log")
        logger.log("Drawing sample [" + filename + "]")
        logger.log("Data transforming starts")

        original_dataset = self.preprocessing.read_file(self.datasets_path + filename)

        # target dataset
        X = original_dataset[:]

        self.preprocessing.filter_by_column(X, "Dir", ["<?>", "<->", "<-"])

        self.preprocessing.transform_labels_with_normals(X)
        self.preprocessing.transform_non_numerical_column(X, "DstAddr")
        self.preprocessing.transform_non_numerical_column(X, "SrcAddr")
        self.preprocessing.transform_non_numerical_column(X, "Dport")
        self.preprocessing.transform_non_numerical_column(X, "Sport")
        self.preprocessing.transform_non_numerical_column(X, "Proto")
        self.preprocessing.transform_non_numerical_column(X, "Dir")
        self.preprocessing.transform_non_numerical_column(X, "State")

        X = X.dropna(axis=1, how='any')

        for i in range(0, self.analyzed_features.__len__()):
         self.preprocessing.quantile_standarization(X, self.analyzed_features[i])  # features normalized to range [0,1], so they may be plotted

        self.analyzed_features.append('inlier')
        X_normal = X[:]
        X_anomaly = X[:]

        # filter out anomalies to have two distinct plots
        X_normal = self.preprocessing.filter_by_column(X_normal, "inlier", [1])
        X_anomaly = self.preprocessing.filter_by_column(X_anomaly, "inlier", [-1])

        # let's leave only relevant features
        X_normal = X_normal[self.analyzed_features]
        X_anomaly = X_anomaly[self.analyzed_features]

        logger.log("Data transforming finished, drawing anomaly plot first")

        fig = plt.figure(1)
        parallel_coordinates(X_anomaly, 'inlier')
        fig.set_size_inches(19.2, 10.8)
        fig.savefig(self.output_path + "\parallel-coordinates_" + filename + "_anomaly.png", dpi=100, bbox_inches='tight')

        logger.log("Anomalies drawn, drawing regular data plot")

        fig1 = plt.figure(2)
        parallel_coordinates(X_normal, 'inlier')
        fig1.set_size_inches(19.2, 10.8)
        fig1.savefig(self.output_path + "\parallel-coordinates_" + filename + "_regular.png", dpi=100, bbox_inches='tight')

        logger.log("Drawing sample [" + filename + "] finished")
