from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Preprocessing:
    le = LabelEncoder()
    oneHotEncoder = OneHotEncoder()
    q_ts = {}
    std_scs = {}
    std_ns = {}
    std_min_maxs = {}

    def read_file(self, path):
        return pd.read_csv(path)

    def transform_labels(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "inlier"] = -1
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False) == False, "inlier"] = 1

    def transform_labels_invert(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "inlier"] = 1
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False) == False, "inlier"] = 0

    def transform_labels_with_normals(self, original_dataset):
        original_dataset.loc[
            original_dataset['Label'].str.contains(pat="background", case=False), "inlier"] = 0
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "inlier"] = -1
        original_dataset.loc[original_dataset['Label'].str.contains(pat="normal", case=False), "inlier"] = 1

    def transform_labels_text(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False) == False, "Label"] = "Regular"
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "Label"] = "Anomaly"

    def transform_labels_text_detailed(self, original_dataset):
        original_dataset.loc[original_dataset['Label'].str.contains(pat="background", case=False), "Label"] = "Background"
        original_dataset.loc[original_dataset['Label'].str.contains(pat="botnet", case=False), "Label"] = "Anomaly"
        original_dataset.loc[original_dataset['Label'].str.contains(pat="normal", case=False), "Label"] = "Normal"

    def transform_non_numerical_column(self, dataset, column):
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

    def get_not_present_values(self, df1, df2=None):
        if df2 is None:
            return df1
        return df1[~df1.index.isin(df2.index)]

    def add_dataframes(self, df1, df2):
        columns = list(df2.columns.values)
        for i in range(0, columns.__len__()):
            column = columns[i]
            df1[column] = df2[column]

        return df1

    def feature_selection_chi2(self, X, Y, result_feature_count):
        # Create and fit selector
        selector = SelectKBest(chi2, k=result_feature_count)
        selector.fit(X, Y)
        # Get idxs of columns to keep
        idxs_selected = X.columns[selector.get_support()]
        # Create new dataframe with only desired columns, or overwrite existing
        return X[idxs_selected]

    def feature_selection_random_forest(self, X, Y, result_feature_count):
        imp = self.get_rf_feat_importances(X, Y)
        sorted_indexes = np.flip(np.argsort(imp), axis=0)
        result_string = '\n'
        original_order_string = '\n'
        headers_line = ''
        scores_line = ''
        for s in sorted_indexes:
            result_string += (X.columns[s].__str__()) + '\t' + imp[s].__str__() + '\n'
        for i in range(0, imp.__len__()):
            headers_line += (X.columns[i].__str__()) + '\t'
            scores_line += (imp[i].__str__()) + '\t'
        idx = []
        for i in range(0, result_feature_count):
            idx.append(sorted_indexes[i])
        original_order_string = '\n' + headers_line + '\n' + scores_line
        return X[X.columns[idx]], result_string, original_order_string

    def get_rf_feat_importances(self, X, Y):
        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(X, Y)
        return rf.feature_importances_

    def feature_selection_mutual_info(self, X, Y, result_feature_count, discrete_mask):
        # Create and fit selector
        selector = SelectKBest(mutual_info_classif, k=result_feature_count)
        selector.fit(X, Y)
        # Get idxs of columns to keep
        idxs_selected = X.columns[selector.get_support()]
        # Create new dataframe with only desired columns, or overwrite existing
        return X[idxs_selected]

    def quantile_standarization(self, dataset, column):

        if column not in self.q_ts:
            q_t = QuantileTransformer()
            self.q_ts[column] = q_t
            result = q_t.fit_transform(dataset[column].values.reshape(-1, 1))
        else:
            q_t = self.q_ts[column]
            result = q_t.transform(dataset[column].values.reshape(-1, 1))
        dataset[column] = result

    def normalization(self, dataset, column):
        if column not in self.std_ns:
            std_n = Normalizer()
            self.std_ns[column] = std_n
            result = std_n.fit_transform(dataset[column].values.reshape(-1, 1))
        else:
            std_n = self.std_ns[column]
            result = std_n.transform(dataset[column].values.reshape(-1, 1))
        dataset[column] = result

    def standard_scaler(self, dataset, column):
        if column not in self.std_scs:
            std_sc = StandardScaler()
            self.std_scs[column] = std_sc
            result = std_sc.fit_transform(dataset[column].values.reshape(-1, 1))
        else:
            std_sc = self.std_scs[column]
            result = std_sc.transform(dataset[column].values.reshape(-1, 1))
        dataset[column] = result

    def min_max(self, dataset, column):
        if column not in self.std_scs:
            std_min_max = MinMaxScaler()
            self.std_min_maxs[column] = std_min_max
            result = std_min_max.fit_transform(dataset[column].values.reshape(-1, 1))
        else:
            std_min_max = self.std_min_maxs[column]
            result = std_min_max.transform(dataset[column].values.reshape(-1, 1))
        dataset[column] = result


    def transform_column_to_dummies(self, dataset, column):
        # uniques = dataset[column].domain.nunique()
        dummies = pd.get_dummies(dataset[column], prefix=column)
        dataset = dataset.drop(column, axis=1)
        return pd.concat([dataset, pd.DataFrame(data=dummies, index=dataset.index.values)], axis=1), list(dummies.head())
