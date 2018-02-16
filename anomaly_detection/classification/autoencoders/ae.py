from anomaly_detection.classification.autoencoders.autoencoder import Autoencoder
from multiprocessing import Pool
import numpy as np

version = "2.0.1"

filenames = [
             "scenario_6.binetflow",
             "scenario_2.binetflow",
             "scenario_9.binetflow",
             "scenario_1.binetflow",
             "scenario_8.binetflow"
             ]

numerical_features = [
    "Dur",
    # "SrcBytesRate",
    "TotPkts",
    # "TotBytesRate",
    "TotBytes",
    "SrcBytes"
    # "PacketOverhead"
]

categorical_features = [
    "SrcAddr",
    "DstAddr",
    "Dport",
    "Sport",
    "Proto",
    "Dir",
    "State"
]

binary_features = [
    # 'is_email'
]

PROCESSES_NUMBER = 4
CV_LOOP = 4
PCA_TURNED_ON = False
epochs=100

NL = "\n"


def do_ae(filename):
    ae = Autoencoder(filename, version, numerical_features, categorical_features, binary_features)
    ae.log("Autoencoder initialized")
    accs = []
    precisions = []
    recalls = []
    f1s = []
    auc_scores = []

    for loop in range(1, CV_LOOP+1):
        ae.log("Loop " + loop.__str__() + "/" + (CV_LOOP).__str__())
        acc, precision, recall, f1, auc_score = ae.perform_ae(nb_epoch=epochs)

        ae.log("Current best scores:")
        accs = np.append(accs, acc)
        precisions = np.append(precisions, precision)
        recalls = np.append(recalls, recall)
        f1s = np.append(f1s, f1)
        auc_scores = np.append(auc_scores, auc_score)

        ae.log('ACC, PRECISION, RECALL, F1, AUC MEAN\n' + np.mean(accs).__str__() + '\n' + np.mean(
            precisions).__str__() + '\n' + np.mean(recalls).__str__() + '\n' + np.mean(f1s).__str__() + '\n' + np.mean(
            auc_scores).__str__())
        ae.log('ACC, PRECISION, RECALL, F1, AUC STD\n' + np.std(accs).__str__() + '\n' + np.std(
            precisions).__str__() + '\n' + np.std(recalls).__str__() + '\n' + np.std(f1s).__str__() + '\n' + np.std(
            auc_scores).__str__())

if __name__ == '__main__':
    pool = Pool(processes=PROCESSES_NUMBER)
    pool.map(do_ae, filenames)
    pool.close()
    pool.join()
