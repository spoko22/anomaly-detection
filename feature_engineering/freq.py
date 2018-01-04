class FrequencyIndicator:
    reference_set = None
    median_cols = {}

    def __init__(self, reference_set):
        self.reference_set = reference_set

    def using_median(self, dataset, column, new_column=None, remove_old=False):
        if column not in self.median_cols:
            self.median_cols[column] = self.reference_set[column].value_counts()

        vals = self.median_cols[column]
        median = vals.median()
        new = dataset[:]

        if new_column is None:
            new[column] = new[column].apply(lambda x: vals[x]/median)
        else:
            new[new_column] = new[column].apply(lambda x: vals[x]/median)
            if remove_old:
                return new.drop(column, axis=1)
        return new
