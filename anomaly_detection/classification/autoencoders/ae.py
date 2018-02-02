from anomaly_detection.classification.autoencoders.autoencoder import Autoencoder

version = "2.0.0"

filenames = [
             # "scenario_6.binetflow"
             "scenario_2.binetflow"
             # "scenario_9.binetflow",
             # "scenario_1.binetflow",
             # "scenario_8.binetflow"
             ]

numerical_features = [
    # "Dur",
    "SrcBytesRate",
    # "TotPkts",
    "TotBytesRate",
    "TotBytes"
]

categorical_features = [
    "DstAddr",
    # "SrcAddr",
    # "Proto",
    # "Dport",
    # "Sport",
    # "Dir",
    # "State"
]

ae = Autoencoder(filenames[0], version, numerical_features, categorical_features)

ae.perform_ae()