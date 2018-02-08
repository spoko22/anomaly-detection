import numpy as np


class TechnicalFeatures:
     def rates(self, orig_dataset, column, new_column=None, duration_col="Dur"):
        dataset = orig_dataset[:]
        if new_column is not None:
            dataset[new_column] = (dataset[column]/dataset[duration_col]).replace([np.inf, -np.inf, np.nan], 0)
        else:
            result = dataset[column]/dataset[duration_col]
            dataset[column] = result.replace([np.inf, -np.inf, np.nan], 0)
        return dataset

     def subtraction(self, orig_dataset, column1, column2, new_column=None):
         dataset = orig_dataset[:]
         if new_column is not None:
             dataset[new_column] = dataset[column1] - dataset[column2]
         else:
             result = dataset[column1] - dataset[column2]
             dataset[column1] = result
         return dataset

     def add_unreachable(self, dataset):
         dataset.loc[self.__is_unreachable__(dataset), "is_unreachable"] = 1
         dataset.loc[~self.__is_unreachable__(dataset), "is_unreachable"] = 0

     def add_reset(self, dataset):
         dataset.loc[~self.__is_reset__(dataset), "reset"] = 0
         dataset.loc[self.__is_reset__(dataset), "reset"] = 1

     def add_is_http(self, dataset, new_column):
         dataset.loc[self.__is_http__(dataset), new_column] = 1
         dataset.loc[~self.__is_http__(dataset), new_column] = 0

     def add_is_email(self, dataset, new_column):
         dataset.loc[self.__is_email__(dataset), new_column] = 1
         dataset.loc[~self.__is_email__(dataset), new_column] = 0

     def add_is_irc(self, dataset, new_column):
         dataset.loc[self.__is_irc__(dataset), new_column] = 1
         dataset.loc[~self.__is_irc__(dataset), new_column] = 0

     def __is_http__(self, dataset):
         return dataset['Dport'].isin(['80', '8080', '443']) | dataset['Sport'].isin(['80', '8080', '443'])

     def __is_email__(self, dataset):
         return (dataset['Dport'].isin(['25', '465', '110', '995', '143', '993', '587']) | dataset['Sport'].isin(['25', '465', '110', '995', '143', '993', '587'])) & dataset['Proto'].str.lower().isin(['tcp'])

     def __is_irc__(self, dataset):
         return (dataset['Dport'].isin(['6697', '6660', '6661', '6662', '6663', '6664', '6665', '6666', '6667', '6668', '6669', '7000']) | dataset['Sport'].isin(['6697', '6660', '6661', '6662', '6663', '6664', '6665', '6666', '6667', '6668', '6669', '7000'])) & dataset['Proto'].str.lower().isin(['tcp'])

     def __is_reset__(self, dataset):
         return (dataset['State'].str.contains(pat="R", case=False)) & (dataset['Proto'].str.lower().isin(['tcp']))

     def __is_unreachable__(self, dataset):
         return dataset['State'].str.contains(pat="UR", case=False)