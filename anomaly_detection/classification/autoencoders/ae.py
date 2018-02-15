from anomaly_detection.classification.autoencoders.autoencoder import Autoencoder
from multiprocessing import Pool

version = "2.0.0"

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
    # "SrcAddr",
    # "DstAddr",
    "Dport",
    "Sport",
    "Proto",
    "Dir",
    "State"
]

binary_features = [
    # 'is_email'
]

PROCESSES_NUMBER = 1
PCA_TURNED_ON = False


def do_ae(filename):
    ae = Autoencoder(filename, version, numerical_features, categorical_features, binary_features)
    ae.perform_ae()


if __name__ == '__main__':
    pool = Pool(processes=PROCESSES_NUMBER)
    pool.map(do_ae, filenames)
    pool.close()
    pool.join()
