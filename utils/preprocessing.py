from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

class Preprocessing:
    le = LabelEncoder()
    oneHotEncoder = OneHotEncoder()

    def read_file(self, path):
        return pd.read_csv(path)

    def transform_labels(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "attack"] = 1
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False) == False, "attack"] = -1

    def transform_labels_text(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "Label"] = "Anomaly"
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False) == False, "Label"] = "Regular"

    def transform_non_numerical_column(self, dataset, *column_names):
        for column in column_names:
            series = pd.factorize(dataset[column])[0]
            result = self.le.fit_transform(series)
            dataset.loc[[True] * dataset[column].shape[0], column] = result

    def filter_by_column(self, dataset, column, values):
        return dataset.loc[dataset[column].isin(values)]

    def normalize_columns(self, dataset, column_names):
        for column in column_names:
            X = dataset[column]
            X_norm = (X - X.min()) / (X.max() - X.min())
            dataset.loc[[True] * dataset[column].shape[0], column] = X_norm

    def get_not_present_values(self, df1, df2):
        return df2[~df2.index.isin(df1.index)]
