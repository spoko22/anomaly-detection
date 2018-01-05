import numpy as np


class TechnicalFeatures:
     def rates(self, orig_dataset, column, new_column=None, duration_col="Dur"):
        dataset = orig_dataset[:]
        if new_column is not None:
            dataset[new_column] = (dataset[column]/dataset[duration_col]).replace([np.inf, -np.inf], 0)
        else:
            result = dataset[column]/dataset[duration_col]
            dataset[column] = result.replace([np.inf, -np.inf], 0)
        return dataset

