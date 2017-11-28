from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from utils.preprocessing import Preprocessing
import seaborn as sns

preprocessing = Preprocessing()

datasets_path = "../../../datasets/"

filenames = ["scenario_1.binetflow", "scenario_2.binetflow", "scenario_6.binetflow", "scenario_8.binetflow", "scenario_9.binetflow"]


print("[" + datetime.now().__str__() + "]: " + 'Script starting')

cols = [
        "Dur",
        "TotPkts",
        "TotBytes",
        "SrcBytes"
]

cols_with_labels = cols[:]
cols_with_labels.append('attack')

original_dataset = pd.read_csv(datasets_path + filenames[0])

X = original_dataset[:]

preprocessing.transform_labels_text(X)

# preprocessing.normalize_columns(X, cols)
cols.append("Label")

X = X[['TotBytes', 'Label']]

fig, (ax) = plt.subplots(1)
sns.boxplot(data=(X[(X < 1000)].dropna(axis=1, how='all')), x="Label", y="TotBytes", ax=ax)

# ax.set_yscale("log")

fig.savefig("test.png", dpi=100, bbox_inches='tight')

print("[" + datetime.now().__str__() + "]: " + 'Drawing ended')
