import pandas as pd
from utils.preprocessing import Preprocessing


class DataframeUtils:
    pp = Preprocessing()

    def merge_results(self, train_set, train_labels, predictions):
        df = pd.DataFrame()
        df = self.pp.add_dataframes(df, train_set)
        df['inlier'] = train_labels
        df['prediction'] = pd.DataFrame(predictions, index=train_labels.index.values)[0]
        return df
