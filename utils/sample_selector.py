import datetime
from sklearn import model_selection
from utils.preprocessing import Preprocessing
import pandas as pd

class SampleSelector:
    dataset = None
    pp = Preprocessing()

    def __init__(self, dataset):
        self.dataset=dataset

    def strategy_25min(self, test_size=0.2, cv_size=0.0):
        prm = 'StartTime'
        self.dataset[prm] = pd.to_datetime(self.dataset[prm])
        sorted = self.dataset.sort_values(by=[prm])
        min_date = sorted[prm].min()
        max_date = min_date + datetime.timedelta(minutes=25)

        piece_25min = sorted[(sorted[prm] >= min_date) & (sorted[prm] <= max_date)]

        # sanity check - if there are anomalies in first 25 minutes, the idea behind this strategy is wrong
        piece_25_min_without_anomalies = self.pp.filter_by_column(piece_25min, 'attack', [-1])

        if piece_25min.__len__() != piece_25_min_without_anomalies.__len__():
            columns = list(self.dataset.columns.values)
            empty_dt = pd.DataFrame(columns=columns)
            return empty_dt, empty_dt

        value_counts = self.dataset['attack'].value_counts()
        anomalies_ratio = self.__count_anomaly_ratio(value_counts)

        cv_df, test_df = self.__produce_datasets(excluded_rows=piece_25min,
                                                 anomalies_ratio=anomalies_ratio,
                                                 test_size=test_size, cv_size=cv_size)

        return piece_25min, cv_df, test_df

    def novelty_detection_random(self, train_size=0.6, test_size=0.2, cv_size=0.0):
        regular_dataset = self.dataset[:]
        regular_dataset = self.pp.filter_by_column(regular_dataset, 'attack', [-1])

        expected_train_df = train_size

        if train_size < 1 & train_size > 0:
            expected_train_df = round(self.dataset.__len__() * train_size, 0)

        train_df = regular_dataset.sample(n=expected_train_df)

        value_counts = self.dataset['attack'].value_counts()
        anomalies_ratio = self.__count_anomaly_ratio(value_counts)

        cv_df, test_df = self.__produce_datasets(excluded_rows=train_df,
                                                 anomalies_ratio=anomalies_ratio,
                                                 test_size=test_size, cv_size=cv_size)

        return train_df, cv_df, test_df

    def __count_anomaly_ratio(self, counts):
        anomalies = counts[1]
        normal = counts[-1]
        all = normal + anomalies

        return anomalies/all

    def __produce_datasets(self, excluded_rows=None, anomalies_ratio=0.05, test_size=0.2, cv_size=0.0):
        rem_dataset = self.pp.get_not_present_values(self.dataset, excluded_rows)
        whole_dataset_size = self.dataset.shape[0]

        cv_expected_anomalies_count = int(round(cv_size * whole_dataset_size * anomalies_ratio, 0))
        cv_expected_regularities_count = int(round(cv_size * whole_dataset_size * (1-anomalies_ratio), 0))
        test_expected_anomalies_count = int(round(test_size * whole_dataset_size * anomalies_ratio, 0))
        test_expected_regularities_count = int(round(test_size * whole_dataset_size * (1-anomalies_ratio), 0))

        rem_anomalies = self.pp.filter_by_column(rem_dataset, 'attack', [1])
        rem_regularities = self.pp.filter_by_column(rem_dataset, 'attack', [-1])

        test_anomalies = rem_anomalies.sample(n=test_expected_anomalies_count)
        rem_anomalies = self.pp.get_not_present_values(rem_anomalies, test_anomalies)
        cv_anomalies = rem_anomalies.sample(n=cv_expected_anomalies_count)

        test_regularities = rem_regularities.sample(n=test_expected_regularities_count)
        rem_regularities = self.pp.get_not_present_values(rem_regularities, test_regularities)

        cv_regularities = rem_regularities.sample(n=cv_expected_regularities_count)

        cv_set = cv_anomalies.append(cv_regularities)
        test_set = test_anomalies.append(test_regularities)

        return cv_set, test_set