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
        piece_25_min_without_anomalies = self.pp.filter_by_column(piece_25min, 'inlier', [1])

        if piece_25min.__len__() != piece_25_min_without_anomalies.__len__():
            columns = list(self.dataset.columns.values)
            empty_dt = pd.DataFrame(columns=columns)
            return empty_dt, empty_dt

        value_counts = self.dataset['attack'].value_counts()
        anomalies_ratio = self.__count_anomaly_ratio(value_counts)

        expected_test_df = test_size

        if (test_size < 1.0) & (test_size > 0.0):
            expected_test_df = round(self.dataset.__len__() * test_size, 0)

        expected_cv_df = cv_size

        if (cv_size < 1.0) & (cv_size > 0.0):
            expected_cv_df = round(self.dataset.__len__() * cv_size, 0)

        cv_df, test_df = self.__produce_datasets(excluded_rows=piece_25min,
                                                 anomalies_ratio=anomalies_ratio,
                                                 test_size=expected_test_df, cv_size=expected_cv_df)

        return piece_25min, cv_df, test_df

    def novelty_detection_normal_heavy(self, train_size=0.6, test_size=0.2, cv_size=0.0):
        normal_dataset = self.pp.filter_by_column(self.dataset, 'inlier', [1])
        all_normal = len(normal_dataset)
        background_dataset = self.pp.filter_by_column(self.dataset, 'inlier', [0])
        botnet_dataset = self.pp.filter_by_column(self.dataset, 'inlier', [-1])

        expected_train_df = train_size

        if (train_size < 1.0) & (train_size > 0.0):
            expected_train_df = round(self.dataset.__len__() * train_size, 0)

        expected_test_df = test_size

        if (test_size < 1.0) & (test_size > 0.0):
            expected_test_df = round(self.dataset.__len__() * test_size, 0)

        expected_cv_df = cv_size

        if (cv_size < 1.0) & (cv_size > 0.0):
            expected_cv_df = round(self.dataset.__len__() * cv_size, 0)

        train_normal_ratio = expected_train_df / (expected_train_df + expected_test_df + expected_cv_df)
        test_normal_ratio = expected_test_df / (expected_train_df + expected_test_df + expected_cv_df)
        cv_normal_ratio = expected_cv_df / (expected_train_df + expected_test_df + expected_cv_df)

        desired_normal_count = (expected_train_df + 0.95 * expected_train_df + 0.95 * expected_cv_df)

        normal_count = all_normal if all_normal < desired_normal_count else desired_normal_count

        train_normal_size = int(round(train_normal_ratio*normal_count, 0))
        test_normal_size = int(round(test_normal_ratio*normal_count*0.95, 0))
        cv_normal_size = int(round(cv_normal_ratio*normal_count*0.95, 0))

        train_normal_df = normal_dataset.sample(n=train_normal_size)
        test_normal_df = normal_dataset.sample(n=test_normal_size)
        cv_normal_df = normal_dataset.sample(n=cv_normal_size)

        test_bot, botnet_dataset = self.__get_rows__(botnet_dataset, [], 0.05 * expected_test_df)
        cv_bot, botnet_dataset = self.__get_rows__(botnet_dataset, [test_bot], 0.05 * expected_cv_df)

        train_bg, background_dataset = self.__get_rows__(background_dataset, [], expected_train_df - len(train_normal_df) if expected_train_df - len(train_normal_df) > 0 else 0)
        test_bg, background_dataset = self.__get_rows__(background_dataset, [train_bg], expected_test_df - len(test_normal_df) - len(test_bot) if expected_test_df - len(test_normal_df) - len(test_bot) > 0 else 0)
        cv_bg, background_dataset = self.__get_rows__(background_dataset, [train_bg, test_bg], expected_cv_df - len(cv_normal_df) - len(cv_bot) if expected_cv_df - len(cv_normal_df) - len(cv_bot) > 0 else 0)

        train_normal_df = train_normal_df.append(train_bg)
        test_normal_df = test_normal_df.append(test_bg)
        test_normal_df = test_normal_df.append(test_bot)
        cv_normal_df = cv_normal_df.append(cv_bg)
        cv_normal_df = cv_normal_df.append(cv_bot)

        self.pp.transform_labels(train_normal_df)
        self.pp.transform_labels(test_normal_df)
        self.pp.transform_labels(cv_normal_df)

        if len(train_normal_df) != expected_train_df and len(test_normal_df) != expected_test_df and len(cv_normal_df) != expected_cv_df:
            raise RuntimeError("Wrong size of dataframe")
        return train_normal_df, cv_normal_df, test_normal_df

    def __get_rows__(self, rows, excluded_rows, count):
        rem_dataset = rows
        for var in range(0, excluded_rows.__len__()):
            rem_dataset = self.pp.get_not_present_values(rem_dataset, excluded_rows[var])
        df = rem_dataset.sample(n=int(count))
        return df, self.pp.get_not_present_values(rem_dataset, df)

    def novelty_detection_random(self, train_size=0.6, test_size=0.2, cv_size=0.0):
        regular_dataset = self.dataset[:]
        regular_dataset = self.pp.filter_by_column(regular_dataset, 'inlier', [1])

        expected_train_df = train_size

        if (train_size < 1.0) & (train_size > 0.0):
            expected_train_df = round(self.dataset.__len__() * train_size, 0)

        expected_test_df = test_size

        if (test_size < 1.0) & (test_size > 0.0):
            expected_test_df = round(self.dataset.__len__() * test_size, 0)

        expected_cv_df = cv_size

        if (cv_size < 1.0) & (cv_size > 0.0):
            expected_cv_df = round(self.dataset.__len__() * cv_size, 0)

        train_df = regular_dataset.sample(n=expected_train_df)

        value_counts = self.dataset['inlier'].value_counts()
        anomalies_ratio = self.__count_anomaly_ratio(value_counts)

        cv_df, test_df = self.__produce_datasets(excluded_rows=train_df,
                                                 anomalies_ratio=anomalies_ratio,
                                                 test_size=expected_test_df, cv_size=expected_cv_df)

        return train_df, cv_df, test_df

    def __count_anomaly_ratio(self, counts):
        anomalies = counts[-1]
        normal = counts[1]
        all = normal + anomalies

        return anomalies/all

    def __produce_datasets(self, excluded_rows=None, anomalies_ratio=0.05, test_size=0.2, cv_size=0.0):
        rem_dataset = self.pp.get_not_present_values(self.dataset, excluded_rows)
        whole_dataset_size = self.dataset.shape[0]

        cv_df_size = cv_size if cv_size > 1.0 else cv_size * whole_dataset_size
        test_df_size = test_size if test_size > 1.0 else test_size * whole_dataset_size

        cv_expected_anomalies_count = int(round(cv_df_size * anomalies_ratio, 0))
        cv_expected_regularities_count = int(round(cv_df_size * (1-anomalies_ratio), 0))
        test_expected_anomalies_count = int(round(test_df_size * anomalies_ratio, 0))
        test_expected_regularities_count = int(round(test_df_size * (1-anomalies_ratio), 0))

        rem_anomalies = self.pp.filter_by_column(rem_dataset, 'inlier', [-1])
        rem_regularities = self.pp.filter_by_column(rem_dataset, 'inlier', [1])

        test_anomalies = rem_anomalies.sample(n=test_expected_anomalies_count)
        rem_anomalies = self.pp.get_not_present_values(rem_anomalies, test_anomalies)
        cv_anomalies = rem_anomalies.sample(n=cv_expected_anomalies_count)

        test_regularities = rem_regularities.sample(n=test_expected_regularities_count)
        rem_regularities = self.pp.get_not_present_values(rem_regularities, test_regularities)

        cv_regularities = rem_regularities.sample(n=cv_expected_regularities_count)

        cv_set = cv_anomalies.append(cv_regularities)
        test_set = test_anomalies.append(test_regularities)

        return cv_set, test_set